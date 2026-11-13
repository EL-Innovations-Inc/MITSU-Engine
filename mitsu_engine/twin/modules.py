"""PyTorch modules that mirror the numpy engine's language-model layers.

Attribute names are the engine's, so a parameter has the same name in both. The only
difference in storage is that torch.nn.Linear keeps its weight as (out, in) and the
engine as (in, out); convert.py transposes on the way across.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..blocks import rope_tables as _engine_rope_tables
from ..init import GATE_MULT
from .functional import mitsu, rope_rotate, _work, LOW_PRECISION


class MITSU(nn.Module):
    """MITSU(u) = gain * s (s - 1) / (s + c), s = softplus(u). No parameters."""

    def __init__(self, c=2.0, gain=1.0):
        super().__init__()
        self.c, self.gain = float(c), float(gain)

    def forward(self, u):
        y = mitsu(u, self.c)
        return y if self.gain == 1.0 else y * self.gain

    def extra_repr(self):
        return f"c={self.c:g}, gain={self.gain:g}"


class MitsuGLU(nn.Module):
    """W2( gate_mult * MITSU(W1 x) * W3 x ): SwiGLU with the SiLU gate replaced by MITSU.

    Same constructor and parameter names as the MitsuGLU module in the main MITSU
    repository (mitsu_torch.MitsuGLU), so either can be used inside TorchMitsuLM."""

    def __init__(self, dim, hidden_dim, gate_mult=GATE_MULT, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.gate_mult = float(gate_mult)

    def forward(self, x):
        return self.w2(self.gate_mult * mitsu(self.w1(x)) * self.w3(x))

    def extra_repr(self):
        return f"gate_mult={self.gate_mult:g}"


class RMSNorm(nn.Module):
    """y = gain * x / sqrt(mean(x^2) + eps) on the last axis. Half-precision inputs are
    normalised in float32."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        w = _work(x)
        inv = 1.0 / torch.sqrt((w * w).mean(-1, keepdim=True) + self.eps)
        return (w * inv).to(x.dtype) * self.gain


class TiedHead(nn.Module):
    """Output layer that reuses an Embedding's table: logits = x E^T.

    It holds the Embedding module, not a copy of its Parameter, and owns no parameter of
    its own. Sharing one Parameter object between two modules also ties them, but some
    device moves replace Parameter objects and silently untie them; looking the table up
    through the Embedding on every call cannot come apart."""

    def __init__(self, embedding):
        super().__init__()
        self._embedding = (embedding,)             # in a tuple so it is not registered a second time

    @property
    def weight(self):
        return self._embedding[0].weight

    def forward(self, x):
        return F.linear(x, self.weight)


class Attention(nn.Module):
    """Multi-head self-attention. The defaults are the language model's: causal, rotary
    positions, no bias. causal, rope and bias mirror the engine's MultiHeadSelfAttention.

    forward(x) is the training path. step(x, cache) is the incremental path used for
    generation: it processes only the new tokens and reuses earlier keys and values."""

    def __init__(self, dim, n_heads, rope_base=10000.0, context=256, causal=True, rope=True, bias=False):
        super().__init__()
        if dim % n_heads or (rope and (dim // n_heads) % 2):
            raise ValueError("dim must be divisible by n_heads, and dim / n_heads must be even for rotary positions")
        self.dim, self.n_heads, self.rope_base, self.context, self.causal, self.rope = dim, n_heads, rope_base, context, causal, rope
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self._tables = {}          # (device, dtype) -> (cos, sin); derived data, not saved

    def _rope(self, length, offset, like):
        """cos and sin for positions offset .. offset + length - 1, on `like`'s device and
        dtype (float32 when `like` is half precision). The angles come from the engine's own
        function in float64 and are then rounded, exactly as the engine does."""
        dtype = torch.float32 if like.dtype in LOW_PRECISION else like.dtype
        key = (like.device, dtype)
        tables = self._tables.get(key)
        if tables is None or tables[0].shape[0] < offset + length:
            n = max(self.context, offset + length)
            cos, sin = _engine_rope_tables(n, self.dim // self.n_heads, self.rope_base, 0, np.float64)
            tables = tuple(torch.from_numpy(t).to(device=like.device, dtype=dtype) for t in (cos, sin))
            self._tables[key] = tables
        return tables[0][offset:offset + length], tables[1][offset:offset + length]

    def _qkv(self, x, offset):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)   # each (B, H, T, D/H)
        if not self.rope:
            return q, k, v
        cos, sin = self._rope(T, offset, x)
        return rope_rotate(q, cos, sin), rope_rotate(k, cos, sin), v

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self._qkv(x, 0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    @torch.no_grad()
    def step(self, x, cache=None):
        """Attend from the new tokens x (batch, new, dim) to everything seen so far.
        `cache` is (keys, values) of the earlier tokens. Returns (output, new cache)."""
        B, T, D = x.shape
        past = 0 if cache is None else cache[0].shape[2]
        q, k, v = self._qkv(x, past)
        if past:
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        mask = None
        if self.causal and T > 1:      # new token i sits at position past + i and may see positions <= past + i
            mask = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(past)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(y.transpose(1, 2).reshape(B, T, D)), (k, v)

    def extra_repr(self):
        return f"dim={self.dim}, heads={self.n_heads}, rope_base={self.rope_base:g}"


class Block(nn.Module):
    """x = x + Dropout(Attention(RMSNorm(x)));  x = x + Dropout(MitsuGLU(RMSNorm(x)))"""

    def __init__(self, dim, n_heads, hidden, gate_mult=GATE_MULT, dropout=0.0, rope_base=10000.0, context=256, glu=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, rope_base, context)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = (glu or MitsuGLU)(dim, hidden, gate_mult=gate_mult, bias=False)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        return x + self.drop2(self.ffn(self.norm2(x)))

    @torch.no_grad()
    def step(self, x, cache=None):
        a, cache = self.attn.step(self.norm1(x), cache)
        x = x + a
        return x + self.ffn(self.norm2(x)), cache
