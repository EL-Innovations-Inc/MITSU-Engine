"""Ready-made networks.

    MLP            plain Linear -> activation stack. For shallow networks (up to about 8 layers).
    DeepMitsuNet   normalised residual blocks. Trains at any depth.
    MitsuGPT       decoder-only transformer language model with MITSU feed-forwards.
"""
import math
import numpy as np
from . import config, functional as F, init as initlib
from .module import Module, Sequential, Parameter
from .layers import Linear, Embedding, LayerNorm, Dropout
from .activations import get_activation
from .blocks import mitsu_block, TransformerBlock


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
        self.blocks = [mitsu_block(width, hidden, block, activation, dropout, scale, rng) for _ in range(depth)]
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
        self.blocks = [TransformerBlock(d_model, n_heads, ffn, activation, hidden, True, dropout, scale, rng) for _ in range(n_layers)]
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
                logits = self.forward(idx[:, -self.context:])[:, -1, :].astype(np.float64)
                if temperature <= 0:
                    nxt = logits.argmax(-1)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        kth = np.sort(logits, -1)[:, -min(top_k, logits.shape[-1])][:, None]
                        logits = np.where(logits < kth, -np.inf, logits)
                    p = F.softmax(logits, -1)
                    nxt = np.array([rng.choice(p.shape[-1], p=row) for row in p])
                idx = np.concatenate([idx, nxt[:, None]], axis=1)
        self.train(was_training)
        return idx

    def extra_repr(self):
        return f"vocab={self.vocab_size}, context={self.context}, d_model={self.d_model}, layers={len(self.blocks)}"
