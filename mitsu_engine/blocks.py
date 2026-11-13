"""Building blocks: feed-forward branches, MitsuGLU, self-attention, residual blocks.

Initialisation convention for every branch that sits inside a Residual:
the input is LayerNorm-ed (unit variance), the first layer keeps it unit-variance,
and the LAST layer of the branch is scaled so the branch's output has variance
`out_scale`^2. Models pass out_scale = branch_scale / sqrt(number of branches), so
the residual stream's variance stays bounded however deep the network is.
"""
import math
import numpy as np
from . import config, functional as F, init as initlib
from .module import Module, Sequential, Residual, Parameter
from .layers import Linear, LayerNorm, Dropout
from .activations import MITSU, get_activation


class FeedForward(Module):
    """Linear -> activation -> Linear. The classic MLP branch."""

    def __init__(self, dim, hidden, activation="mitsu", bias=True, out_scale=1.0, d_out=None, rng=None):
        rng = config.get_rng(rng)
        self.fc = Linear(dim, hidden, bias=bias, init="lecun", rng=rng)
        self.act = get_activation(activation)
        self.out = Linear(hidden, d_out or dim, bias=bias, init=initlib.activation_gain(self.act), scale=out_scale, rng=rng)

    def forward(self, x):
        return self.out(self.act(self.fc(x)))

    def backward(self, grad):
        return self.fc.backward(self.act.backward(self.out.backward(grad)))


class MitsuGLU(Module):
    """SwiGLU with the SiLU gate replaced by MITSU:

        y = W2( gate_mult * MITSU(W1 x)  *  W3 x )

    gate_mult (default 3.04) restores the gate's signal size at initialisation; set it
    to 1.0 for the raw formula. With hidden = 8/3 * dim it has the parameter count of a
    4x feed-forward block.
    """

    def __init__(self, dim, hidden=None, gate_mult=initlib.GATE_MULT, bias=False, out_scale=1.0, c=2.0, rng=None):
        rng = config.get_rng(rng)
        hidden = hidden or int(round(8 * dim / 3))
        self.dim, self.hidden = dim, hidden
        self.w1 = Linear(dim, hidden, bias=bias, init="lecun", rng=rng)
        self.w3 = Linear(dim, hidden, bias=bias, init="lecun", rng=rng)
        self.gate = MITSU(c=c, gain=gate_mult)
        self.w2 = Linear(hidden, dim, bias=bias, init=initlib.activation_gain(self.gate), scale=out_scale, rng=rng)
        self._c = None

    def forward(self, x):
        g = self.gate(self.w1(x))
        v = self.w3(x)
        if config.grad_enabled():
            self._c = (g, v)
        return self.w2(g * v)

    def backward(self, grad):
        g, v = self._c
        gh = self.w2.backward(grad)
        return self.w1.backward(self.gate.backward(gh * v)) + self.w3.backward(gh * g)

    def extra_repr(self):
        return f"{self.dim} -> {self.hidden} -> {self.dim}"


class MultiHeadSelfAttention(Module):
    """Scaled dot-product self-attention over (batch, time, dim). `causal=True` stops
    each position from seeing later ones (language modelling)."""

    def __init__(self, dim, n_heads, causal=True, bias=False, out_scale=1.0, rng=None):
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        rng = config.get_rng(rng)
        self.dim, self.n_heads, self.causal = dim, n_heads, causal
        self.qkv = Linear(dim, 3 * dim, bias=bias, init="lecun", rng=rng)
        self.proj = Linear(dim, dim, bias=bias, init="lecun", scale=out_scale, rng=rng)
        self._c = None

    def _split(self, x, B, T):      # (B, T, D) -> (B, H, T, D/H)
        return x.reshape(B, T, self.n_heads, self.dim // self.n_heads).transpose(0, 2, 1, 3)

    def _merge(self, x, B, T):      # (B, H, T, D/H) -> (B, T, D)
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.dim)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (self._split(part, B, T) for part in np.split(self.qkv(x), 3, axis=-1))
        scale = 1.0 / math.sqrt(D // self.n_heads)
        att = (q @ k.transpose(0, 1, 3, 2)) * scale
        if self.causal:
            att = att + np.triu(np.full((T, T), -np.inf, dtype=att.dtype), 1)
        p = F.softmax(att, -1)
        if config.grad_enabled():
            self._c = (q, k, v, p, scale, B, T)
        return self.proj(self._merge(p @ v, B, T))

    def backward(self, grad):
        q, k, v, p, scale, B, T = self._c
        go = self._split(self.proj.backward(grad), B, T)
        gp = go @ v.transpose(0, 1, 3, 2)
        gv = p.transpose(0, 1, 3, 2) @ go
        gatt = p * (gp - (gp * p).sum(-1, keepdims=True)) * scale
        gq = gatt @ k
        gk = gatt.transpose(0, 1, 3, 2) @ q
        gqkv = np.concatenate([self._merge(g, B, T) for g in (gq, gk, gv)], axis=-1)
        return self.qkv.backward(gqkv)

    def extra_repr(self):
        return f"dim={self.dim}, heads={self.n_heads}, causal={self.causal}"


def residual_branch(dim, inner, dropout=0.0, rng=None):
    """Residual( LayerNorm -> inner [-> Dropout] ): the pre-norm residual unit."""
    layers = [LayerNorm(dim), inner]
    if dropout > 0:
        layers.append(Dropout(dropout, rng=rng))
    return Residual(Sequential(*layers))


def mitsu_block(dim, hidden=None, kind="mlp", activation="mitsu", dropout=0.0, out_scale=1.0, rng=None):
    """One pre-norm residual block for deep feed-forward networks.

    kind='mlp':  x + W2 act(W1 LN(x))          hidden defaults to 2 * dim
    kind='glu':  x + MitsuGLU(LN(x))           hidden defaults to 8/3 * dim
    """
    rng = config.get_rng(rng)
    if kind == "mlp":
        inner = FeedForward(dim, hidden or 2 * dim, activation, out_scale=out_scale, rng=rng)
    elif kind == "glu":
        inner = MitsuGLU(dim, hidden, out_scale=out_scale, rng=rng)
    else:
        raise ValueError("kind must be 'mlp' or 'glu'")
    return residual_branch(dim, inner, dropout, rng)


class TransformerBlock(Module):
    """Pre-norm transformer block: x + Attention(LN(x)), then x + FeedForward(LN(x)).

    ffn='glu' uses MitsuGLU (hidden 8/3 * dim); ffn='mlp' uses Linear-activation-Linear
    (hidden 4 * dim) with `activation`.
    """

    def __init__(self, dim, n_heads, ffn="glu", activation="mitsu", hidden=None, causal=True, dropout=0.0, out_scale=1.0, rng=None):
        rng = config.get_rng(rng)
        self.attn = residual_branch(dim, MultiHeadSelfAttention(dim, n_heads, causal, out_scale=out_scale, rng=rng), dropout, rng)
        if ffn == "glu":
            inner = MitsuGLU(dim, hidden, out_scale=out_scale, rng=rng)
        elif ffn == "mlp":
            inner = FeedForward(dim, hidden or 4 * dim, activation, bias=False, out_scale=out_scale, rng=rng)
        else:
            raise ValueError("ffn must be 'glu' or 'mlp'")
        self.ff = residual_branch(dim, inner, dropout, rng)

    def forward(self, x):
        return self.ff(self.attn(x))

    def backward(self, grad):
        return self.attn.backward(self.ff.backward(grad))
