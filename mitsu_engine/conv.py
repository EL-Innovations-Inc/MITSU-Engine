"""Convolution, pooling and group normalisation for images, in (batch, channels, height, width) layout.

The array kernels at the top (conv2d_forward and its two backward halves, the pooling
routes) are plain functions on numpy arrays. The layers below and the automatic
differentiation operations in autograd.py both call them, so there is one implementation
of the arithmetic. Each layer's backward pass is checked against finite differences.
"""
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from . import config, init as initlib
from .module import Module, Parameter


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def _windows(x, kh, kw, stride):
    """Every kh x kw patch of x (N, C, H, W) at the given stride, as a view of shape
    (N, C, H_out, W_out, kh, kw). No data is copied."""
    return sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]


def _pad(x, p):
    return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x


def out_size(size, kernel, stride=1, padding=0):
    """Output length of a convolution or pooling along one axis."""
    return (size + 2 * padding - kernel) // stride + 1


def conv2d_forward(x, w, b=None, stride=1, padding=0):
    """Cross-correlate x (N, C, H, W) with w (C_out, C, kh, kw). Returns (y, col).

    `col` holds every input patch as one row, shape (N * H_out * W_out, C * kh * kw): the
    convolution is then a single matrix product, and conv2d_backward reuses the same rows."""
    N, co = x.shape[0], w.shape[0]
    win = _windows(_pad(x, padding), w.shape[2], w.shape[3], stride)                  # (N, C, H_out, W_out, kh, kw), a view
    Ho, Wo = win.shape[2], win.shape[3]
    col = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, -1)                    # the one copy of the patches
    y = col @ w.reshape(co, -1).T
    if b is not None:
        y += b
    return np.ascontiguousarray(y.reshape(N, Ho, Wo, co).transpose(0, 3, 1, 2)), col


def conv2d_backward(grad, col, w, x_shape, stride=1, padding=0, need_input=True):
    """Gradients of a convolution given `grad` = dL/dy and the patch rows from the forward
    pass. Returns (dL/dx or None, dL/dw, dL/db)."""
    co, C, kh, kw = w.shape
    N, _, Ho, Wo = grad.shape
    g = grad.transpose(0, 2, 3, 1).reshape(-1, co)
    gw = (g.T @ col).reshape(w.shape)
    gb = g.sum(axis=0)
    gx = None
    if need_input:
        H, W = x_shape[2], x_shape[3]
        # gradient of every patch entry, arranged so each kernel tap is one contiguous block
        taps = np.ascontiguousarray((g @ w.reshape(co, -1)).reshape(N, Ho, Wo, C, kh, kw).transpose(4, 5, 0, 3, 1, 2))
        gxp = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=grad.dtype)
        for i in range(kh):
            for j in range(kw):                                                       # each tap adds to a strided slice of the input
                gxp[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride] += taps[i, j]
        gx = np.ascontiguousarray(gxp[:, :, padding:padding + H, padding:padding + W]) if padding else gxp
    return gx, gw, gb


def max_pool_forward(x, kernel, stride):
    """Returns (y, index of the winning position inside each window)."""
    win = _windows(x, kernel, kernel, stride)
    flat = win.reshape(*win.shape[:4], kernel * kernel)
    idx = flat.argmax(-1)
    return np.take_along_axis(flat, idx[..., None], -1)[..., 0], idx


def max_pool_backward(grad, idx, x_shape, kernel, stride):
    gx = np.zeros(x_shape, dtype=grad.dtype)
    Ho, Wo = grad.shape[2], grad.shape[3]
    for i in range(kernel):
        for j in range(kernel):
            gx[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride] += grad * (idx == i * kernel + j)
    return gx


def avg_pool_forward(x, kernel, stride):
    return _windows(x, kernel, kernel, stride).mean(axis=(4, 5))


def avg_pool_backward(grad, x_shape, kernel, stride):
    gx = np.zeros(x_shape, dtype=grad.dtype)
    Ho, Wo = grad.shape[2], grad.shape[3]
    g = grad / (kernel * kernel)
    for i in range(kernel):
        for j in range(kernel):
            gx[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride] += g
    return gx


class Conv2d(Module):
    """2-D convolution over (batch, channels, height, width). The weight has shape
    (c_out, c_in, kernel, kernel), the layout PyTorch uses.

    init, scale   as in Linear: weights are N(0, (gain * scale)^2 / fan_in) with
                  fan_in = c_in * kernel^2; `init` is a name or a number (the gain).
    """

    def __init__(self, c_in, c_out, kernel=3, stride=1, padding=0, bias=True, init="lecun", scale=1.0, rng=None):
        self.c_in, self.c_out, self.kernel, self.stride, self.padding = c_in, c_out, kernel, stride, padding
        fan = c_in * kernel * kernel
        self.weight = Parameter(initlib.normal((c_out, c_in, kernel, kernel), initlib.gain(init) * scale / math.sqrt(fan), rng))
        self.bias = Parameter(np.zeros(c_out), decay=False) if bias else None
        self._c = None

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.c_in:
            raise ValueError(f"Conv2d expects (batch, {self.c_in}, height, width), got {x.shape}")
        y, col = conv2d_forward(x, self.weight.data, None if self.bias is None else self.bias.data, self.stride, self.padding)
        if config.grad_enabled():
            self._c = (col, x.shape)
        return y

    def backward(self, grad):
        col, x_shape = self._c
        gx, gw, gb = conv2d_backward(grad, col, self.weight.data, x_shape, self.stride, self.padding)
        self.weight.grad += gw
        if self.bias is not None:
            self.bias.grad += gb
        return gx

    def extra_repr(self):
        return f"{self.c_in} -> {self.c_out}, kernel={self.kernel}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None}"


class MaxPool2d(Module):
    """The largest value in each kernel x kernel window. stride defaults to the kernel size."""

    def __init__(self, kernel=2, stride=None):
        self.kernel, self.stride = kernel, stride or kernel
        self._c = None

    def forward(self, x):
        y, idx = max_pool_forward(x, self.kernel, self.stride)
        if config.grad_enabled():
            self._c = (idx, x.shape)
        return y

    def backward(self, grad):
        idx, x_shape = self._c
        return max_pool_backward(grad, idx, x_shape, self.kernel, self.stride)

    def extra_repr(self):
        return f"kernel={self.kernel}, stride={self.stride}"


class AvgPool2d(Module):
    """The mean of each kernel x kernel window. stride defaults to the kernel size."""

    def __init__(self, kernel=2, stride=None):
        self.kernel, self.stride = kernel, stride or kernel
        self._c_shape = None

    def forward(self, x):
        self._c_shape = x.shape
        return avg_pool_forward(x, self.kernel, self.stride)

    def backward(self, grad):
        return avg_pool_backward(grad, self._c_shape, self.kernel, self.stride)

    def extra_repr(self):
        return f"kernel={self.kernel}, stride={self.stride}"


class GlobalAvgPool2d(Module):
    """(batch, channels, height, width) -> (batch, channels): the mean over each feature map."""

    def __init__(self):
        self._c_shape = None

    def forward(self, x):
        self._c_shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, grad):
        N, C, H, W = self._c_shape
        return np.broadcast_to(grad[:, :, None, None] / (H * W), self._c_shape).copy()


class GroupNorm(Module):
    """Normalise each group of channels of each image to zero mean and unit variance, then
    scale and shift per channel. It has no batch statistics, so it behaves the same in
    training and evaluation and at any batch size. groups=1 normalises a whole image."""

    def __init__(self, groups, channels, eps=1e-5):
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.groups, self.channels, self.eps = groups, channels, eps
        self.gain = Parameter(np.ones(channels), decay=False)
        self.bias = Parameter(np.zeros(channels), decay=False)
        self._c = None

    def forward(self, x):
        N = x.shape[0]
        xg = x.reshape(N, self.groups, -1)
        xc = xg - xg.mean(-1, keepdims=True)
        inv = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + self.eps)
        xhat = (xc * inv).reshape(x.shape)
        if config.grad_enabled():
            self._c = (xhat, inv)
        return xhat * self.gain.data[None, :, None, None] + self.bias.data[None, :, None, None]

    def backward(self, grad):
        xhat, inv = self._c
        self.gain.grad += (grad * xhat).sum(axis=(0, 2, 3))
        self.bias.grad += grad.sum(axis=(0, 2, 3))
        N = grad.shape[0]
        gh = (grad * self.gain.data[None, :, None, None]).reshape(N, self.groups, -1)
        xh = xhat.reshape(N, self.groups, -1)
        return ((gh - gh.mean(-1, keepdims=True) - xh * (gh * xh).mean(-1, keepdims=True)) * inv).reshape(grad.shape)

    def extra_repr(self):
        return f"groups={self.groups}, channels={self.channels}"
