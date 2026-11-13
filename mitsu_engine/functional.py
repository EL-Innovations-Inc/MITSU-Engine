"""Stateless maths: MITSU and the reference activations, with exact derivatives.

    mother function   f(x, y, z) = (x^(y+z) + y^(x+z) + z^(x+y)) / (x + y + z)
    MITSU(u)          f(s, 1, 1) - 1 = s (s - 1) / (s + 2),   s = softplus(u)

`mitsu(u, c)` is the one-parameter family s (s - 1) / (s + c); c = 2 is MITSU.
"""
import numpy as np

SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def softplus(u):
    """log(1 + e^u), computed without overflow."""
    return np.maximum(u, 0) + np.log1p(np.exp(-np.abs(u)))


def sigmoid(u):
    e = np.exp(-np.abs(u))
    return np.where(u >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def mitsu(u, c=2.0):
    """s (s - 1) / (s + c). The division happens before the multiplication, so the
    intermediate s * (s - 1) never overflows in low precision."""
    s = softplus(u)
    return s * ((s - 1.0) / (s + c))


def mitsu_grad(u, c=2.0):
    """d mitsu / du = (1 - c (c + 1) / (s + c)^2) * sigmoid(u), written as two ratios so
    (s + c)^2 is never formed (it overflows in half precision)."""
    s = softplus(u)
    return (1.0 - (c / (s + c)) * ((c + 1.0) / (s + c))) * sigmoid(u)


def relu(u):
    return np.maximum(u, 0)


def relu_grad(u):
    return (u > 0).astype(u.dtype)


def silu(u):
    return u * sigmoid(u)


def silu_grad(u):
    sg = sigmoid(u)
    return sg * (1.0 + u * (1.0 - sg))


def gelu(u):
    """GELU, tanh approximation (the form used by GPT-2). numpy has no erf."""
    return 0.5 * u * (1.0 + np.tanh(SQRT_2_OVER_PI * (u + 0.044715 * u ** 3)))


def gelu_grad(u):
    inner = SQRT_2_OVER_PI * (u + 0.044715 * u ** 3)
    t = np.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * u * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3 * 0.044715 * u * u)


def tanh(u):
    return np.tanh(u)


def tanh_grad(u):
    return 1.0 - np.square(np.tanh(u))


def log_softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
