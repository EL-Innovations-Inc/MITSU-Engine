"""Ready-made networks.

    MLP            plain Linear -> activation stack. For shallow networks (up to about 8 layers).
    DeepMitsuNet   normalised residual blocks. Trains at any depth.
    MitsuGPT       small decoder-only transformer: learned positions, LayerNorm.
    MitsuLM        the language-model architecture: rotary positions, RMSNorm, MitsuGLU,
                   tied embeddings, cached generation. Described by an LMConfig.
"""
import math
from dataclasses import dataclass, asdict, fields
import numpy as np
from . import config, functional as F, init as initlib
from .module import Module, Sequential, Parameter
from .layers import Linear, Embedding, LayerNorm, RMSNorm, TiedHead, Dropout
from .activations import get_activation
from .blocks import mitsu_block, TransformerBlock
from .io import save_checkpoint, load_checkpoint, read_meta


class MLP(Sequential):
    """sizes = [d_in, h1, ..., d_out]. Linear layers with an activation between them.

    A plain stack of MITSU layers has no normalisation and no skip paths. It trains up to
    about 8 layers and fails at 16 (see init.py), so use DeepMitsuNet when you want depth.
    `init` is the weight gain: 'he' by default; a larger number such as 3.0 speeds up
    learning on small inputs because MITSU is a quiet activation."""

    def __init__(self, sizes, activation="mitsu", init="he", dropout=0.0, rng=None):
        rng = config.get_rng(rng)
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(Linear(a, b, init=init, rng=rng))
            if i < len(sizes) - 2:
                layers.append(get_activation(activation if isinstance(activation, str) else type(activation)))
                if dropout > 0:
                    layers.append(Dropout(dropout, rng=rng))
        super().__init__(*layers)


class DeepMitsuNet(Module):
    """Deep feed-forward network for vectors:

        stem Linear(d_in, width)
        depth x  [ x + Branch(LayerNorm(x)) ]        Branch = 'mlp' or 'glu' (MitsuGLU)
        LayerNorm -> Linear(width, d_out)

    Each branch is initialised to add variance branch_scale^2 / depth, so the signal is
    stable at any depth. Parameters: about depth * 2 * width * hidden.
    """

    def __init__(self, d_in, d_out, width=256, depth=16, hidden=None, block="mlp", activation="mitsu",
                 dropout=0.0, branch_scale=1.0, rng=None):
        rng = config.get_rng(rng)
        self.d_in, self.d_out, self.width, self.depth = d_in, d_out, width, depth
        self.stem = Linear(d_in, width, init="lecun", rng=rng)
        scale = branch_scale / math.sqrt(max(depth, 1))
        self.blocks = [mitsu_block(width, hidden, block, activation, dropout, scale, rng=rng) for _ in range(depth)]
        self.norm = LayerNorm(width)
        self.head = Linear(width, d_out, init="lecun", rng=rng)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def backward(self, grad):
        grad = self.norm.backward(self.head.backward(grad))
        for blk in reversed(self.blocks):
            grad = blk.backward(grad)
        return self.stem.backward(grad)

    def extra_repr(self):
        return f"d_in={self.d_in}, d_out={self.d_out}, width={self.width}, depth={self.depth}"


class MitsuGPT(Module):
    """Decoder-only transformer language model.

        token embedding + learned position embedding
        n_layers x TransformerBlock (causal attention + MITSU feed-forward)
        LayerNorm -> Linear(d_model, vocab_size)

    forward(idx) takes integer tokens (batch, time <= context) and returns logits
    (batch, time, vocab_size).
    """

    def __init__(self, vocab_size, context, d_model=128, n_layers=4, n_heads=4, ffn="glu", activation="mitsu",
                 hidden=None, dropout=0.0, branch_scale=1.0, rng=None):
        rng = config.get_rng(rng)
        self.vocab_size, self.context, self.d_model = vocab_size, context, d_model
        self.tok = Embedding(vocab_size, d_model, std=0.02, rng=rng)
        self.pos = Parameter(initlib.normal((context, d_model), 0.02, rng), decay=False)
        self.drop = Dropout(dropout, rng=rng)
        scale = branch_scale / math.sqrt(2 * max(n_layers, 1))
        self.blocks = [TransformerBlock(d_model, n_heads, ffn, activation, hidden, True, dropout, scale, rng=rng) for _ in range(n_layers)]
        self.norm = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, bias=False, init="lecun", rng=rng)
        self._c_T = None

    def forward(self, idx):
        idx = np.asarray(idx)
        T = idx.shape[1]
        if T > self.context:
            raise ValueError(f"sequence length {T} exceeds the context of {self.context}")
        self._c_T = T
        x = self.drop(self.tok(idx) + self.pos.data[:T])
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def backward(self, grad):
        grad = self.norm.backward(self.head.backward(grad))
        for blk in reversed(self.blocks):
            grad = blk.backward(grad)
        grad = self.drop.backward(grad)
        self.pos.grad[: self._c_T] += grad.sum(0)
        self.tok.backward(grad)
        return None

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, rng=None):
        """Sample `max_new_tokens` tokens after the prompt `idx` (1-D or (1, T) integers)."""
        rng = config.get_rng(rng)
        idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
        was_training = self.training
        self.eval()
        with config.no_grad():
            for _ in range(max_new_tokens):
                nxt = sample_token(self.forward(idx[:, -self.context:])[:, -1, :], temperature, top_k, None, rng)
                idx = np.concatenate([idx, nxt[:, None]], axis=1)
        self.train(was_training)
        return idx

    def extra_repr(self):
        return f"vocab={self.vocab_size}, context={self.context}, d_model={self.d_model}, layers={len(self.blocks)}"


def sample_token(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    """Pick one token per row of `logits` (batch, vocab).

    temperature <= 0   take the most likely token (deterministic)
    top_k              keep only the k most likely tokens
    top_p              keep the smallest set of tokens whose probability adds up to p
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        return logits.argmax(-1)
    rng = config.get_rng(rng)
    logits = logits / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = np.sort(logits, -1)[:, -top_k][:, None]
        logits = np.where(logits < kth, -np.inf, logits)
    p = F.softmax(logits, -1)
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-p, -1)
        sorted_p = np.take_along_axis(p, order, -1)
        drop = np.cumsum(sorted_p, -1) - sorted_p >= top_p        # tokens after the mass is reached
        sorted_p[drop] = 0.0
        p = np.zeros_like(p)
        np.put_along_axis(p, order, sorted_p, -1)
        p /= p.sum(-1, keepdims=True)
    return np.array([rng.choice(p.shape[-1], p=row) for row in p])


@dataclass
class LMConfig:
    """Everything needed to rebuild a MitsuLM. Stored inside every saved model.

    vocab_size       number of tokens
    context          longest sequence the model is trained on
    d_model          width of the residual stream
    n_layers         number of transformer blocks
    n_heads          attention heads per block (d_model / n_heads must be even)
    ffn_hidden       MitsuGLU hidden size; None means round(8/3 * d_model)
    gate_mult        MitsuGLU gate gain (3.04 restores a SiLU-sized gate; 1.0 is the raw formula)
    dropout          dropout rate on embeddings and both residual branches
    tie_embeddings   share the token table between the input and the output layer
    rope_base        base of the rotary position frequencies
    """
    vocab_size: int
    context: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_hidden: int = None
    gate_mult: float = initlib.GATE_MULT
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_base: float = 10000.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class MitsuLM(Module):
    """Decoder-only language model in the modern (Llama-style) layout, with MITSU gates:

        token embedding
        n_layers x [ x + Attention(RMSNorm(x))      causal, rotary positions
                     x + MitsuGLU(RMSNorm(x)) ]
        RMSNorm -> output layer (tied to the token embedding by default)

    No bias terms and no position table: position enters only through the rotation inside
    attention. forward(idx) maps integer tokens (batch, time) to logits (batch, time, vocab).
    """

    def __init__(self, config_or_vocab_size, rng=None, **kwargs):
        cfg = config_or_vocab_size if isinstance(config_or_vocab_size, LMConfig) else LMConfig(config_or_vocab_size, **kwargs)
        if cfg.d_model % cfg.n_heads or (cfg.d_model // cfg.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even whole number")
        rng = config.get_rng(rng)
        self.config = cfg
        self.tok = Embedding(cfg.vocab_size, cfg.d_model, std=0.02, rng=rng)
        self.drop = Dropout(cfg.dropout, rng=rng)
        scale = 1.0 / math.sqrt(2 * max(cfg.n_layers, 1))
        self.blocks = [TransformerBlock(cfg.d_model, cfg.n_heads, "glu", hidden=cfg.ffn_hidden, causal=True, dropout=cfg.dropout,
                                        out_scale=scale, norm="rms", rope=True, rope_base=cfg.rope_base, gate_mult=cfg.gate_mult, rng=rng)
                       for _ in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.d_model)
        if cfg.tie_embeddings:
            self.head = TiedHead(self.tok)
        else:
            self.head = Linear(cfg.d_model, cfg.vocab_size, bias=False, init=0.02 * math.sqrt(cfg.d_model), rng=rng)

    # ---- training path --------------------------------------------------
    def forward(self, idx):
        idx = np.asarray(idx)
        if idx.shape[1] > self.config.context:
            raise ValueError(f"sequence length {idx.shape[1]} exceeds the context of {self.config.context}")
        x = self.drop(self.tok(idx))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def backward(self, grad):
        grad = self.norm.backward(self.head.backward(grad))
        for blk in reversed(self.blocks):
            grad = blk.backward(grad)
        self.tok.backward(self.drop.backward(grad))
        return None

    # ---- generation path ------------------------------------------------
    def step(self, idx_new, caches=None):
        """Logits for the new tokens `idx_new` (batch, new) given `caches` from earlier calls.
        Returns (logits (batch, new, vocab), caches). Evaluation mode, no gradients."""
        caches = caches or [None] * len(self.blocks)
        with config.no_grad():
            x = self.tok(np.asarray(idx_new))
            new = []
            for blk, cache in zip(self.blocks, caches):
                x, cache = blk.step(x, cache)
                new.append(cache)
            return self.head(self.norm(x)), new

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, rng=None, use_cache=True, stop_token=None):
        """Continue the prompt `idx` (1-D or (batch, time) integers) by `max_new_tokens` tokens.

        With use_cache=True each new token costs one incremental step. When the text
        outgrows the context, the window restarts from its most recent half. use_cache=False
        recomputes the same window from scratch every step: slower, identical output.
        Generation stops early if every row has produced `stop_token`."""
        rng = config.get_rng(rng)
        idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))
        context, keep = self.config.context, max(self.config.context // 2, 1)
        was_training = self.training
        self.eval()
        start, caches, logits = 0, None, None
        done = np.zeros(len(idx), dtype=bool)
        for _ in range(max_new_tokens):
            if idx.shape[1] - start > context:            # prompt longer than the context
                start, caches = idx.shape[1] - context, None
            if use_cache:
                fresh = idx[:, start:] if caches is None else idx[:, -1:]
                logits, caches = self.step(fresh, caches)
            else:
                with config.no_grad():
                    logits = self.forward(idx[:, start:])
            nxt = sample_token(logits[:, -1, :], temperature, top_k, top_p, rng)
            idx = np.concatenate([idx, nxt[:, None]], axis=1)
            if stop_token is not None:
                done |= nxt == stop_token
                if done.all():
                    break
            if idx.shape[1] - start >= context:           # window full: restart from its recent half
                start, caches = idx.shape[1] - keep, None
        self.train(was_training)
        return idx

    # ---- persistence ----------------------------------------------------
    def save(self, path, **meta):
        """Write the weights and the config to one .npz file. Extra keyword arguments are stored as metadata."""
        save_checkpoint(path, self, meta={"model": "MitsuLM", "config": self.config.to_dict(), **meta})

    @classmethod
    def load(cls, path, rng=None):
        """Rebuild a model saved with save(): reads the config, builds the network, loads the weights."""
        meta = read_meta(path)
        if "config" not in meta:
            raise ValueError(f"{path} has no model config; it was not written by MitsuLM.save")
        model = cls(LMConfig.from_dict(meta["config"]), rng=rng)
        load_checkpoint(path, model)
        return model

    def extra_repr(self):
        c = self.config
        return f"vocab={c.vocab_size}, context={c.context}, d_model={c.d_model}, layers={c.n_layers}, heads={c.n_heads}, tied={c.tie_embeddings}"
