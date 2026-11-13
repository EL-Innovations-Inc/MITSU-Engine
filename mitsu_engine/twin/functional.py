"""MITSU and rotary positions for PyTorch tensors, written to follow the numpy engine
operation for operation.

Half precision. float16 and bfloat16 inputs are computed in float32 and cast back, and
the division happens before the multiplication, so MITSU itself cannot overflow in half
precision (float16 would overflow at u >= 257 if s * (s - 1) were formed first). This
makes the activation safe under autocast. It does not make a whole network safe: the
product gate * value and the layers around it can still overflow in float16.
"""
import torch

LOW_PRECISION = (torch.float16, torch.bfloat16)


def _work(x):
    return x.float() if x.dtype in LOW_PRECISION else x


def softplus(u):
    """log(1 + e^u) without overflow, and without the linear shortcut above 20 that
    torch.nn.functional.softplus takes: this is the same function the engine computes."""
    return torch.logaddexp(u, torch.zeros((), dtype=u.dtype, device=u.device))


def mitsu(u, c=2.0):
    """s (s - 1) / (s + c), s = softplus(u). c = 2 is MITSU. Output dtype matches the input."""
    if not u.is_floating_point():
        raise TypeError("MITSU expects a floating-point tensor")
    s = softplus(_work(u))
    return (s * ((s - 1.0) / (s + c))).to(u.dtype)


def rope_rotate(x, cos, sin):
    """Rotate each consecutive pair of features of x (..., time, head_dim) by its position's
    angle. cos and sin have shape (time, head_dim / 2)."""
    w = _work(x)
    x1, x2 = w[..., 0::2], w[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).to(x.dtype)
