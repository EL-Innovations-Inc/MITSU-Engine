"""TorchMitsuLM: the numpy engine's MitsuLM, written with PyTorch modules.

Same LMConfig, same parameter names, same checkpoint files, same generation procedure.
The tests load identical weights into both and require the outputs, the loss, every
gradient, the cached generation path and the generated tokens to agree.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .. import config as engine_config
from ..io import read_checkpoint, write_checkpoint
from ..models import MitsuLM, LMConfig, sample_token
from .convert import engine_state, load_engine_state, lm_layout
from .modules import Block, RMSNorm, TiedHead


def _torch_dtype(array):
    return torch.float64 if array.dtype == np.float64 else torch.float32


class TorchMitsuLM(nn.Module):
    """Decoder-only language model: token embedding, n_layers x [attention, MitsuGLU] with
    RMSNorm and rotary positions, RMSNorm, output layer tied to the token table.

    TorchMitsuLM(cfg)                    initial weights drawn by the numpy engine (see `rng`)
    TorchMitsuLM.from_engine(model)      copy of a numpy MitsuLM
    TorchMitsuLM.load("model.npz")       from a checkpoint written by either engine

    rng    None, an int or a numpy Generator, with the engine's meaning: the weights are
           the ones MitsuLM(cfg, rng=rng) would have, so both engines start identically.
           The model takes the engine's precision (float32 unless me.set_dtype says float64).
    init   False skips initialisation (the weights are about to be overwritten).
    glu    the feed-forward class; defaults to twin.MitsuGLU. mitsu_torch.MitsuGLU from the
           main MITSU repository has the same constructor and can be passed here.
    """

    def __init__(self, cfg, rng=None, init=True, glu=None, **kwargs):
        super().__init__()
        cfg = cfg if isinstance(cfg, LMConfig) else LMConfig(cfg, **kwargs)
        if cfg.d_model % cfg.n_heads or (cfg.d_model // cfg.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even whole number")
        self.config = cfg
        hidden = cfg.ffn_hidden or int(round(8 * cfg.d_model / 3))
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, hidden, cfg.gate_mult, cfg.dropout, cfg.rope_base, cfg.context, glu)
                                     for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = TiedHead(self.tok) if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if init:
            state = MitsuLM(cfg, rng=rng).state_dict()
            self.to(_torch_dtype(state["tok.weight"]))          # adopt the engine's precision (me.set_dtype)
            load_engine_state(self, state)

    @property
    def engine_layout(self):
        """[(name, parameter, transposed, decay)] in the engine's parameter order."""
        return lm_layout(self)

    # ---- training path --------------------------------------------------
    def forward(self, idx):
        """Integer tokens (batch, time) -> logits (batch, time, vocab)."""
        if idx.shape[1] > self.config.context:
            raise ValueError(f"sequence length {idx.shape[1]} exceeds the context of {self.config.context}")
        x = self.drop(self.tok(idx))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def loss(self, idx, targets):
        """Mean cross-entropy (nats per token) of predicting `targets` from `idx`. The
        logits are converted to at least float32 first, so the loss is safe under autocast."""
        logits = self.forward(idx)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    # ---- generation path ------------------------------------------------
    @torch.no_grad()
    def step(self, idx_new, caches=None):
        """Logits for the new tokens `idx_new` (batch, new) given `caches` from earlier calls.
        Returns (logits, caches). No dropout, no gradients."""
        caches = caches or [None] * len(self.blocks)
        x = self.tok(idx_new)
        new = []
        for blk, cache in zip(self.blocks, caches):
            x, cache = blk.step(x, cache)
            new.append(cache)
        return self.head(self.norm(x)), new

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, rng=None, use_cache=True, stop_token=None):
        """Continue the prompt `idx` (1-D or (batch, time) integers) by `max_new_tokens`.

        The procedure is the engine's, line for line: the same window policy when the text
        outgrows the context, and the same sampler (the engine's own `sample_token`, on the
        last position's logits). With the same weights, prompt and `rng`, both engines
        return the same tokens. Returns a numpy int64 array, as the engine does."""
        rng = engine_config.get_rng(rng)
        device = self.tok.weight.device
        idx = np.atleast_2d(np.asarray(idx.cpu() if torch.is_tensor(idx) else idx, dtype=np.int64))
        context, keep = self.config.context, max(self.config.context // 2, 1)
        was_training = self.training
        self.eval()
        start, caches = 0, None
        done = np.zeros(len(idx), dtype=bool)
        for _ in range(max_new_tokens):
            if idx.shape[1] - start > context:            # prompt longer than the context
                start, caches = idx.shape[1] - context, None
            if use_cache:
                fresh = idx[:, start:] if caches is None else idx[:, -1:]
                logits, caches = self.step(torch.from_numpy(np.ascontiguousarray(fresh)).to(device), caches)
            else:
                logits = self.forward(torch.from_numpy(np.ascontiguousarray(idx[:, start:])).to(device))
            last = logits[:, -1, :]
            if last.dtype in (torch.float16, torch.bfloat16):
                last = last.float()
            nxt = sample_token(last.cpu().numpy(), temperature, top_k, top_p, rng)
            idx = np.concatenate([idx, nxt[:, None]], axis=1)
            if stop_token is not None:
                done |= nxt == stop_token
                if done.all():
                    break
            if idx.shape[1] - start >= context:           # window full: restart from its recent half
                start, caches = idx.shape[1] - keep, None
        self.train(was_training)
        return idx

    # ---- the bridge to the engine ---------------------------------------
    @classmethod
    def from_engine(cls, model, device=None, dtype=None, glu=None):
        """A PyTorch copy of a numpy MitsuLM, holding the same weights. `dtype` defaults to
        the engine model's own precision."""
        net = cls(model.config, init=False, glu=glu)
        net.to(device=device, dtype=dtype or _torch_dtype(model.tok.weight.data))
        load_engine_state(net, model.state_dict())
        return net

    def to_engine(self):
        """A numpy MitsuLM holding these weights, in the engine's current precision. Building
        it does not draw from the engine's global random generator."""
        model = MitsuLM(self.config, rng=0)
        model.load_state_dict(engine_state(self))
        return model

    def save(self, path, **meta):
        """Write the weights and the config in the engine's checkpoint format. The file is
        the same one MitsuLM.save writes: the numpy engine loads it with no PyTorch installed."""
        write_checkpoint(path, engine_state(self), meta={"model": "MitsuLM", "config": self.config.to_dict(), **meta})

    @classmethod
    def load(cls, path, device=None, dtype=torch.float32, glu=None):
        """Rebuild a model from a checkpoint written by MitsuLM.save or TorchMitsuLM.save."""
        params, _, meta = read_checkpoint(path)
        if "config" not in meta:
            raise ValueError(f"{path} has no model config; it was not written by MitsuLM.save")
        net = cls(LMConfig.from_dict(meta["config"]), init=False, glu=glu).to(device=device, dtype=dtype)
        load_engine_state(net, params)
        return net

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        c = self.config
        return f"vocab={c.vocab_size}, context={c.context}, d_model={c.d_model}, layers={c.n_layers}, heads={c.n_heads}, tied={c.tie_embeddings}"
