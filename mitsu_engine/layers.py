"""Basic layers: Linear, Embedding, LayerNorm, Dropout, Flatten."""
import numpy as np
from . import config, init as initlib
from .module import Module, Parameter


class Linear(Module):
    """y = x W + b on the last axis. W has shape (d_in, d_out)."""

    def __init__(self, d_in, d_out, bias=True, init="lecun", scale=1.0, rng=None):
        self.d_in, self.d_out = d_in, d_out
        self.weight = Parameter(initlib.linear_weight(d_in, d_out, init, scale, rng))
        self.bias = Parameter(np.zeros(d_out), decay=False) if bias else None
        self._c_x = None

    def forward(self, x):
        if config.grad_enabled():
            self._c_x = x
        y = x @ self.weight.data
        if self.bias is not None:
            y += self.bias.data
        return y

    def backward(self, grad):
        x2 = self._c_x.reshape(-1, self.d_in)
        g2 = grad.reshape(-1, self.d_out)
        self.weight.grad += x2.T @ g2
        if self.bias is not None:
            self.bias.grad += g2.sum(0)
        return grad @ self.weight.data.T

    def extra_repr(self):
        return f"{self.d_in} -> {self.d_out}, bias={self.bias is not None}"


class Embedding(Module):
    """Integer indices -> rows of a table. backward returns None: indices have no gradient."""

    def __init__(self, num_embeddings, dim, std=0.02, rng=None):
        self.num_embeddings, self.dim = num_embeddings, dim
        self.weight = Parameter(initlib.normal((num_embeddings, dim), std, rng), decay=False)
        self._c_idx = None

    def forward(self, idx):
        idx = np.asarray(idx)
        if config.grad_enabled():
            self._c_idx = idx
        return self.weight.data[idx]

    def backward(self, grad):
        np.add.at(self.weight.grad, self._c_idx.reshape(-1), grad.reshape(-1, self.dim))
        return None

    def extra_repr(self):
        return f"{self.num_embeddings} x {self.dim}"


class LayerNorm(Module):
    """Normalise the last axis to zero mean and unit variance, then scale and shift."""

    def __init__(self, dim, eps=1e-5):
        self.dim, self.eps = dim, eps
        self.gain = Parameter(np.ones(dim), decay=False)
        self.bias = Parameter(np.zeros(dim), decay=False)
        self._c = None

    def forward(self, x):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        inv = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + self.eps)
        xhat = xc * inv
        if config.grad_enabled():
            self._c = (xhat, inv)
        return xhat * self.gain.data + self.bias.data

    def backward(self, grad):
        xhat, inv = self._c
        g2 = grad.reshape(-1, self.dim)
        self.gain.grad += (g2 * xhat.reshape(-1, self.dim)).sum(0)
        self.bias.grad += g2.sum(0)
        gh = grad * self.gain.data
        return (gh - gh.mean(-1, keepdims=True) - xhat * (gh * xhat).mean(-1, keepdims=True)) * inv

    def extra_repr(self):
        return f"{self.dim}"


class Dropout(Module):
    """Zero a fraction p of the values while training and rescale the rest. Off in eval mode."""

    def __init__(self, p=0.1, rng=None):
        if not 0.0 <= p < 1.0:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.rng = config.get_rng(rng)
        self._c_mask = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self._c_mask = None
            return x
        mask = (self.rng.random(x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        self._c_mask = mask
        return x * mask

    def backward(self, grad):
        return grad if self._c_mask is None else grad * self._c_mask

    def extra_repr(self):
        return f"p={self.p}"


class Flatten(Module):
    """(N, ...) -> (N, everything else)."""

    def __init__(self):
        self._c_shape = None

    def forward(self, x):
        self._c_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._c_shape)
