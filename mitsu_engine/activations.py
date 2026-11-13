"""Activation layers. MITSU is the default everywhere; the rest are for comparison."""
from . import config, functional as F
from .module import Module


class Activation(Module):
    """Elementwise activation y = gain * fn(u), with an exact derivative."""

    fn = None
    dfn = None

    def __init__(self, gain=1.0):
        self.gain = float(gain)
        self._c_u = None

    def forward(self, u):
        if config.grad_enabled():
            self._c_u = u
        y = type(self).fn(u)
        return y if self.gain == 1.0 else y * self.gain

    def backward(self, grad):
        d = type(self).dfn(self._c_u)
        return grad * d if self.gain == 1.0 else grad * d * self.gain

    def extra_repr(self):
        return "" if self.gain == 1.0 else f"gain={self.gain:g}"


class MITSU(Module):
    """MITSU(u) = s (s - 1) / (s + c), s = softplus(u), with c = 2.

    c:     the seat-weight constant. 2 is MITSU as published; other values are the
           family tested in experiment 06 (smaller opens earlier, larger later).
    gain:  multiplies the output. MITSU is about 3.04 times quieter than SiLU, so a
           gain near 3 restores a SiLU-sized signal. It also multiplies the slope.
    """

    def __init__(self, c=2.0, gain=1.0):
        if c <= 0:
            raise ValueError("c must be positive")
        self.c, self.gain = float(c), float(gain)
        self._c_u = None

    def forward(self, u):
        if config.grad_enabled():
            self._c_u = u
        y = F.mitsu(u, self.c)
        return y if self.gain == 1.0 else y * self.gain

    def backward(self, grad):
        d = F.mitsu_grad(self._c_u, self.c)
        return grad * d if self.gain == 1.0 else grad * d * self.gain

    def extra_repr(self):
        parts = ([] if self.c == 2.0 else [f"c={self.c:g}"]) + ([] if self.gain == 1.0 else [f"gain={self.gain:g}"])
        return ", ".join(parts)


class ReLU(Activation):
    fn, dfn = staticmethod(F.relu), staticmethod(F.relu_grad)


class GELU(Activation):
    fn, dfn = staticmethod(F.gelu), staticmethod(F.gelu_grad)


class SiLU(Activation):
    fn, dfn = staticmethod(F.silu), staticmethod(F.silu_grad)


class Tanh(Activation):
    fn, dfn = staticmethod(F.tanh), staticmethod(F.tanh_grad)


_REGISTRY = {"mitsu": MITSU, "relu": ReLU, "gelu": GELU, "silu": SiLU, "tanh": Tanh}


def get_activation(spec="mitsu"):
    """'mitsu' | 'relu' | 'gelu' | 'silu' | 'tanh', an Activation class, or an instance."""
    if isinstance(spec, Module):
        return spec
    if isinstance(spec, type) and issubclass(spec, Module):
        return spec()
    if spec not in _REGISTRY:
        raise ValueError(f"unknown activation '{spec}'; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[spec]()
