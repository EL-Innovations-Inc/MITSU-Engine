"""Parameter, Module and the two containers everything else is built from.

Every Module implements
    forward(x)      -> y, remembering whatever backward needs
    backward(grad)  -> gradient with respect to x, adding parameter gradients to .grad
Backpropagation is written by hand for each layer and verified by gradcheck.
"""
import numpy as np
from . import config


class Parameter:
    """A trainable array with its gradient. `decay` marks it for weight decay
    (weights yes; biases, gains and embeddings' norms no)."""

    __slots__ = ("data", "grad", "decay")

    def __init__(self, data, decay=True):
        self.data = np.ascontiguousarray(data, dtype=config.get_dtype())
        self.grad = np.zeros_like(self.data)
        self.decay = decay

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def zero_grad(self):
        self.grad.fill(0)

    def __repr__(self):
        return f"Parameter(shape={self.data.shape}, dtype={self.data.dtype.name})"


class Module:
    """Base class. Sub-modules and Parameters are found by walking attributes
    (including lists and tuples of them) in the order they were assigned."""

    training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    # ---- tree walking -------------------------------------------------
    def children(self):
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield f"{name}.{i}", item

    def modules(self):
        yield self
        for _, child in self.children():
            yield from child.modules()

    def named_parameters(self, prefix=""):
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                yield prefix + name, value
        for name, child in self.children():
            yield from child.named_parameters(prefix + name + ".")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    # ---- state --------------------------------------------------------
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state, strict=True):
        own = dict(self.named_parameters())
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        if strict and (missing or unexpected):
            raise KeyError(f"state does not match the model. missing: {missing[:5]}  unexpected: {unexpected[:5]}")
        for name, p in own.items():
            if name in state:
                value = np.asarray(state[name])
                if value.shape != p.data.shape:
                    raise ValueError(f"shape mismatch for {name}: model {p.data.shape}, state {value.shape}")
                p.data[...] = value
        return self

    def free_caches(self):
        """Drop everything forward() stored for backward(). Call after training if memory matters."""
        for m in self.modules():
            for key in [k for k in m.__dict__ if k == "_c" or k.startswith("_c_")]:
                m.__dict__[key] = None

    # ---- description --------------------------------------------------
    def extra_repr(self):
        return ""

    def __repr__(self):
        kids = list(self.children())
        head = f"{type(self).__name__}({self.extra_repr()}"
        if not kids:
            return head + ")"
        body = "\n".join("  " + f"({n}): " + repr(c).replace("\n", "\n  ") for n, c in kids)
        return head + ("\n" if not self.extra_repr() else ",\n") + body + "\n)"

    def summary(self):
        """A table of every parameter with its shape and size."""
        rows = [(n, "x".join(map(str, p.shape)) or "scalar", p.size) for n, p in self.named_parameters()]
        width = max([len(r[0]) for r in rows] + [9])
        lines = [f"{'parameter':{width}s}  {'shape':>14s}  {'count':>12s}", "-" * (width + 30)]
        lines += [f"{n:{width}s}  {s:>14s}  {c:12,d}" for n, s, c in rows]
        lines += ["-" * (width + 30), f"{'total':{width}s}  {'':>14s}  {self.num_parameters():12,d}"]
        return "\n".join(lines)


class Identity(Module):
    def forward(self, x):
        return x

    def backward(self, grad):
        return grad


class Sequential(Module):
    """Layers applied one after another."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def append(self, layer):
        self.layers.append(layer)
        return self

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]


class Residual(Module):
    """y = x + fn(x). The skip path is what lets very deep stacks train."""

    def __init__(self, fn):
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

    def backward(self, grad):
        return grad + self.fn.backward(grad)
