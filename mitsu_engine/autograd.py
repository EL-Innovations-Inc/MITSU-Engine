"""Reverse-mode automatic differentiation on numpy arrays.

    import mitsu_engine as me
    from mitsu_engine import autograd as ag

    x = ag.Tensor([0.5, 1.5, 2.5], requires_grad=True)
    y = (ag.mitsu(x) * x).sum()
    y.backward()
    x.grad                                   # dy/dx

A Tensor wraps an array. Every operation records how to pass a gradient back to its
inputs; backward() walks those records from the result to the leaves. The layers in this
engine have hand-written backward passes and do not need this. It is here for three things:

  * writing a new layer without deriving its gradient: subclass AutoModule and write
    compute(x) with Tensor operations (see AutoModule);
  * differentiating any formula directly, such as the mother function with all three
    seats live;
  * a second, independent check of every hand-written backward pass (tests/test_autograd.py).

Engine Parameters can be used directly in Tensor expressions (`x @ self.weight`); the
gradient is added to Parameter.grad, where the optimisers expect it. Inside
`me.no_grad()` nothing is recorded. First derivatives only.
"""
import builtins as _builtins
import numpy as np
from . import config, functional as F, conv as _conv
from .module import Module, Parameter

builtins_max, builtins_any = _builtins.max, _builtins.any      # this module defines its own sum, max, abs and pow


class Tensor:
    """An array that remembers how it was computed.

    data            the numpy array
    requires_grad   True for leaves you want a gradient for, and for everything computed from them
    grad            after backward(): the gradient with respect to this leaf (None before)
    """

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "_param")
    __array_ufunc__ = None                     # numpy_array + Tensor defers to Tensor.__radd__

    def __init__(self, data, requires_grad=False):
        self.data = data.data if isinstance(data, Tensor) else np.asarray(data)
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._parents, self._backward, self._param = (), None, None

    # ---- description ----------------------------------------------------
    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    size = property(lambda self: self.data.size)
    dtype = property(lambda self: self.data.dtype)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor({np.array2string(self.data, precision=4, threshold=8)}{', requires_grad=True' if self.requires_grad else ''})"

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def detach(self):
        """The same values with no history."""
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    # ---- the backward pass ----------------------------------------------
    def backward(self, grad=None):
        """Send a gradient from this tensor back to every leaf it was computed from.

        With no argument this tensor must hold a single number (a loss). Otherwise `grad`
        is the gradient of the final loss with respect to this tensor. Gradients are added
        to leaf.grad for plain leaves and to Parameter.grad for engine Parameters."""
        if not self.requires_grad:
            raise RuntimeError("this tensor does not depend on anything that requires a gradient")
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() with no argument needs a single number; pass the gradient with respect to this tensor")
            grad = np.ones_like(self.data)
        order, seen, stack = [], set(), [(self, False)]
        while stack:                                            # depth-first, without recursion
            node, finished = stack.pop()
            if finished:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents if p.requires_grad and id(p) not in seen)
        grads = {id(self): np.asarray(grad, dtype=self.data.dtype).reshape(self.data.shape)}
        for node in reversed(order):                            # each node after everything that used it
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:                          # a leaf
                if node._param is not None:
                    node._param.grad += g
                else:
                    node.grad = np.array(g) if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._backward(g)):
                if pg is not None and parent.requires_grad:
                    key = id(parent)
                    grads[key] = pg if key not in grads else grads[key] + pg

    # ---- operators ------------------------------------------------------
    def __add__(self, other): return add(self, other)
    def __radd__(self, other): return add(other, self)
    def __sub__(self, other): return sub(self, other)
    def __rsub__(self, other): return sub(other, self)
    def __mul__(self, other): return mul(self, other)
    def __rmul__(self, other): return mul(other, self)
    def __truediv__(self, other): return div(self, other)
    def __rtruediv__(self, other): return div(other, self)
    def __pow__(self, other): return pow(self, other)
    def __rpow__(self, other): return pow(other, self)
    def __matmul__(self, other): return matmul(self, other)
    def __rmatmul__(self, other): return matmul(other, self)
    def __neg__(self): return neg(self)
    def __getitem__(self, index): return getitem(self, index)

    # ---- methods mirroring the functions below ---------------------------
    def sum(self, axis=None, keepdims=False): return sum(self, axis, keepdims)
    def mean(self, axis=None, keepdims=False): return mean(self, axis, keepdims)
    def max(self, axis=None, keepdims=False): return max(self, axis, keepdims)
    def reshape(self, *shape): return reshape(self, *shape)
    def transpose(self, *axes): return transpose(self, *axes)
    def swapaxes(self, a, b): return swapaxes(self, a, b)
    def flatten(self, start=1): return reshape(self, *self.shape[:start], -1)
    def exp(self): return exp(self)
    def log(self): return log(self)
    def sqrt(self): return sqrt(self)
    def tanh(self): return tanh(self)

    @property
    def T(self):
        return transpose(self)


# ---- plumbing ------------------------------------------------------------------------

def as_tensor(x, like=None):
    """Tensor -> itself. Parameter -> a leaf whose gradient goes to Parameter.grad.
    Anything else -> a constant. A Python number takes the precision of `like`."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, Parameter):
        t = Tensor(x.data, requires_grad=config.grad_enabled())
        t._param = x
        return t
    if like is not None and np.isscalar(x) and np.issubdtype(like.data.dtype, np.floating):
        return Tensor(np.asarray(x, dtype=like.data.dtype))
    return Tensor(x)


def _two(a, b):
    if isinstance(a, Tensor) or isinstance(a, Parameter):
        a = as_tensor(a)
        return a, as_tensor(b, a)
    b = as_tensor(b)
    return as_tensor(a, b), b


def _result(data, parents, backward):
    out = Tensor(data)
    if config.grad_enabled() and any(p.requires_grad for p in parents):
        out.requires_grad, out._parents, out._backward = True, parents, backward
    return out


def _unbroadcast(g, shape):
    """Sum a gradient back down to the shape of an operand that numpy broadcast up."""
    if g.shape == shape:
        return g
    extra = g.ndim - len(shape)
    if extra:
        g = g.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and g.shape[i] != 1)
    return g.sum(axis=axes, keepdims=True) if axes else g


# ---- arithmetic ----------------------------------------------------------------------

def add(a, b):
    a, b = _two(a, b)
    return _result(a.data + b.data, (a, b), lambda g: (_unbroadcast(g, a.shape), _unbroadcast(g, b.shape)))


def sub(a, b):
    a, b = _two(a, b)
    return _result(a.data - b.data, (a, b), lambda g: (_unbroadcast(g, a.shape), _unbroadcast(-g, b.shape)))


def mul(a, b):
    a, b = _two(a, b)
    return _result(a.data * b.data, (a, b), lambda g: (_unbroadcast(g * b.data, a.shape), _unbroadcast(g * a.data, b.shape)))


def div(a, b):
    a, b = _two(a, b)
    out = a.data / b.data
    return _result(out, (a, b), lambda g: (_unbroadcast(g / b.data, a.shape), _unbroadcast(-g * out / b.data, b.shape)))


def neg(a):
    a = as_tensor(a)
    return _result(-a.data, (a,), lambda g: (-g,))


def pow(a, b):
    """a ** b. The exponent may be a number or a tensor; a tensor exponent needs a > 0,
    because its gradient is a^b * log(a). This is what lets the mother function
    x^(y+z) + y^(x+z) + z^(x+y) be differentiated with all three seats live."""
    if np.isscalar(b):
        a = as_tensor(a)
        return _result(a.data ** b, (a,), lambda g: (g * b * a.data ** (b - 1),))
    a, b = _two(a, b)
    out = a.data ** b.data
    return _result(out, (a, b), lambda g: (_unbroadcast(g * b.data * a.data ** (b.data - 1), a.shape),
                                           _unbroadcast(g * out * np.log(a.data), b.shape)))


def matmul(a, b):
    """Matrix product on the last two axes, with numpy's broadcasting over leading axes."""
    a, b = _two(a, b)
    if a.ndim == 1:
        return reshape(matmul(reshape(a, 1, -1), b), *b.shape[:-2], *b.shape[-1:]) if b.ndim > 1 else sum(mul(a, b))
    if b.ndim == 1:
        return reshape(matmul(a, reshape(b, -1, 1)), *a.shape[:-1])
    return _result(a.data @ b.data, (a, b), lambda g: (_unbroadcast(g @ np.swapaxes(b.data, -1, -2), a.shape),
                                                       _unbroadcast(np.swapaxes(a.data, -1, -2) @ g, b.shape)))


# ---- elementwise functions -----------------------------------------------------------

def _unary(fn, dfn):
    def op(a):
        a = as_tensor(a)
        return _result(fn(a.data), (a,), lambda g: (g * dfn(a.data),))
    return op


def exp(a):
    a = as_tensor(a)
    out = np.exp(a.data)
    return _result(out, (a,), lambda g: (g * out,))


def log(a):
    a = as_tensor(a)
    return _result(np.log(a.data), (a,), lambda g: (g / a.data,))


def sqrt(a):
    a = as_tensor(a)
    out = np.sqrt(a.data)
    return _result(out, (a,), lambda g: (g * 0.5 / out,))


def tanh(a):
    a = as_tensor(a)
    out = np.tanh(a.data)
    return _result(out, (a,), lambda g: (g * (1.0 - out * out),))


def sigmoid(a):
    a = as_tensor(a)
    out = F.sigmoid(a.data)
    return _result(out, (a,), lambda g: (g * out * (1.0 - out),))


def mitsu(a, c=2.0):
    """MITSU(u) = s (s - 1) / (s + c), s = softplus(u), with its exact derivative."""
    a = as_tensor(a)
    return _result(F.mitsu(a.data, c), (a,), lambda g: (g * F.mitsu_grad(a.data, c),))


softplus = _unary(F.softplus, F.sigmoid)
relu = _unary(F.relu, F.relu_grad)
silu = _unary(F.silu, F.silu_grad)
gelu = _unary(F.gelu, F.gelu_grad)
abs = _unary(np.abs, np.sign)


def maximum(a, b):
    a, b = _two(a, b)
    first = a.data >= b.data
    return _result(np.maximum(a.data, b.data), (a, b), lambda g: (_unbroadcast(g * first, a.shape), _unbroadcast(g * ~first, b.shape)))


def minimum(a, b):
    a, b = _two(a, b)
    first = a.data <= b.data
    return _result(np.minimum(a.data, b.data), (a, b), lambda g: (_unbroadcast(g * first, a.shape), _unbroadcast(g * ~first, b.shape)))


def where(condition, a, b):
    """a where `condition` (a boolean array) is true, b elsewhere."""
    condition = np.asarray(condition.data if isinstance(condition, Tensor) else condition, dtype=bool)
    a, b = _two(a, b)
    return _result(np.where(condition, a.data, b.data), (a, b),
                   lambda g: (_unbroadcast(g * condition, a.shape), _unbroadcast(g * ~condition, b.shape)))


# ---- reductions ----------------------------------------------------------------------

def _expand(g, shape, axis, keepdims):
    if axis is not None and not keepdims:
        g = np.expand_dims(g, axis)
    return np.broadcast_to(g, shape)


def sum(a, axis=None, keepdims=False):
    a = as_tensor(a)
    return _result(a.data.sum(axis=axis, keepdims=keepdims), (a,), lambda g: (_expand(g, a.shape, axis, keepdims),))


def mean(a, axis=None, keepdims=False):
    a = as_tensor(a)
    out = a.data.mean(axis=axis, keepdims=keepdims)
    count = a.data.size / builtins_max(out.size, 1)
    return _result(out, (a,), lambda g: (_expand(g, a.shape, axis, keepdims) / count,))


def max(a, axis=None, keepdims=False):
    """The largest value along `axis`. Where several entries tie, they share the gradient."""
    a = as_tensor(a)
    top = a.data.max(axis=axis, keepdims=True)
    share = a.data == top
    share = share / share.sum(axis=axis, keepdims=True)
    out = top if keepdims else a.data.max(axis=axis)
    return _result(out, (a,), lambda g: (_expand(g, a.shape, axis, keepdims) * share,))


# ---- shape ---------------------------------------------------------------------------

def reshape(a, *shape):
    a = as_tensor(a)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return _result(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),))


def transpose(a, *axes):
    """Permute the axes; with no arguments, reverse them."""
    a = as_tensor(a)
    if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        axes = tuple(axes[0])
    axes = axes or tuple(range(a.ndim))[::-1]
    return _result(a.data.transpose(axes), (a,), lambda g: (g.transpose(np.argsort(axes)),))


def swapaxes(a, i, j):
    a = as_tensor(a)
    return _result(np.swapaxes(a.data, i, j), (a,), lambda g: (np.swapaxes(g, i, j),))


def getitem(a, index):
    """a[index], with numpy's rules. Indexing with integer arrays (an embedding lookup) may
    pick the same entry several times; their gradients add up."""
    a = as_tensor(a)
    index = tuple(i.data if isinstance(i, Tensor) else i for i in index) if isinstance(index, tuple) else (index.data if isinstance(index, Tensor) else index)
    parts = index if isinstance(index, tuple) else (index,)
    fancy = builtins_any(isinstance(i, (np.ndarray, list)) for i in parts)

    def backward(g):
        z = np.zeros(a.shape, dtype=g.dtype)
        if fancy:
            np.add.at(z, index, g)
        else:
            z[index] += g
        return (z,)
    return _result(a.data[index], (a,), backward)


def concatenate(tensors, axis=0):
    tensors = tuple(as_tensor(t) for t in tensors)
    cuts = np.cumsum([t.shape[axis] for t in tensors])[:-1]
    return _result(np.concatenate([t.data for t in tensors], axis=axis), tensors, lambda g: tuple(np.split(g, cuts, axis=axis)))


def stack(tensors, axis=0):
    tensors = tuple(as_tensor(t) for t in tensors)
    return _result(np.stack([t.data for t in tensors], axis=axis), tensors, lambda g: tuple(np.moveaxis(g, axis, 0)))


# ---- probabilities and losses --------------------------------------------------------

def softmax(a, axis=-1):
    a = as_tensor(a)
    out = F.softmax(a.data, axis)
    return _result(out, (a,), lambda g: (out * (g - (g * out).sum(axis=axis, keepdims=True)),))


def log_softmax(a, axis=-1):
    a = as_tensor(a)
    out = F.log_softmax(a.data, axis)
    return _result(out, (a,), lambda g: (g - np.exp(out) * g.sum(axis=axis, keepdims=True),))


def cross_entropy(logits, targets):
    """Mean softmax cross-entropy. logits (..., classes); targets (...) integer class ids."""
    logits = as_tensor(logits)
    t = np.asarray(targets.data if isinstance(targets, Tensor) else targets).reshape(-1)
    logp = F.log_softmax(logits.data.reshape(-1, logits.shape[-1]), -1)
    rows = np.arange(len(t))

    def backward(g):
        d = np.exp(logp)
        d[rows, t] -= 1.0
        return ((d * (g / len(t))).reshape(logits.shape).astype(logits.data.dtype, copy=False),)
    return _result(np.asarray(-logp[rows, t].mean(), dtype=logits.data.dtype), (logits,), backward)


def mse(prediction, target):
    """Mean squared error over every element."""
    d = sub(prediction, target)
    return mean(mul(d, d))


def dropout(a, p, training=True, rng=None):
    """Zero a fraction p of the values and rescale the rest. The identity when not training."""
    a = as_tensor(a)
    if not training or p == 0.0:
        return a
    mask = (config.get_rng(rng).random(a.shape) >= p).astype(a.data.dtype) / (1.0 - p)
    return _result(a.data * mask, (a,), lambda g: (g * mask,))


# ---- images --------------------------------------------------------------------------

def conv2d(x, weight, bias=None, stride=1, padding=0):
    """2-D convolution of x (N, C, H, W) with weight (C_out, C, kh, kw)."""
    x, weight = as_tensor(x), as_tensor(weight)
    bias = None if bias is None else as_tensor(bias)
    y, col = _conv.conv2d_forward(x.data, weight.data, None if bias is None else bias.data, stride, padding)

    def backward(g):
        gx, gw, gb = _conv.conv2d_backward(g, col, weight.data, x.shape, stride, padding, need_input=x.requires_grad)
        return (gx, gw) if bias is None else (gx, gw, gb)
    return _result(y, (x, weight) if bias is None else (x, weight, bias), backward)


def max_pool2d(x, kernel=2, stride=None):
    x, stride = as_tensor(x), stride or kernel
    y, idx = _conv.max_pool_forward(x.data, kernel, stride)
    return _result(y, (x,), lambda g: (_conv.max_pool_backward(g, idx, x.shape, kernel, stride),))


def avg_pool2d(x, kernel=2, stride=None):
    x, stride = as_tensor(x), stride or kernel
    return _result(_conv.avg_pool_forward(x.data, kernel, stride), (x,), lambda g: (_conv.avg_pool_backward(g, x.shape, kernel, stride),))


# ---- helpers -------------------------------------------------------------------------

def grad(fn, *arrays):
    """Gradient of the scalar fn(*tensors) with respect to each array. Returns one array
    for one argument, a list for several."""
    tensors = [Tensor(np.asarray(a), requires_grad=True) for a in arrays]
    fn(*tensors).backward()
    grads = [t.grad if t.grad is not None else np.zeros_like(t.data) for t in tensors]
    return grads[0] if len(grads) == 1 else grads


def check(fn, *arrays, n_checks=24, eps=1e-6, rng=0):
    """Compare the automatic gradient of fn with central finite differences, in float64.

    fn takes Tensors and returns a Tensor; a non-scalar result is reduced to a number with
    a fixed random weighting. Returns the worst error over up to `n_checks` entries of each
    argument, measured as |numeric - automatic| / max(|numeric| + |automatic|, 1e-4 * largest
    gradient entry)."""
    rng = config.get_rng(rng)
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    weights = None

    def scalar(*tensors):
        nonlocal weights
        out = as_tensor(fn(*tensors))
        if out.size == 1:
            return sum(out)
        if weights is None:
            weights = rng.standard_normal(out.shape)
        return sum(mul(out, weights))

    tensors = [Tensor(a, requires_grad=True) for a in arrays]
    scalar(*tensors).backward()
    grads = [t.grad if t.grad is not None else np.zeros_like(t.data) for t in tensors]
    floor = 1e-4 * builtins_max([float(np.abs(g).max()) for g in grads] + [1e-30])
    value = lambda: float(scalar(*[Tensor(a) for a in arrays]).data)
    worst = 0.0
    for a, g in zip(arrays, grads):
        flat, gflat = a.reshape(-1), g.reshape(-1)
        for i in rng.choice(flat.size, size=min(n_checks, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps; up = value()
            flat[i] = old - eps; down = value()
            flat[i] = old
            numeric = (up - down) / (2 * eps)
            worst = builtins_max(worst, _builtins.abs(numeric - gflat[i]) / builtins_max(floor, _builtins.abs(numeric) + _builtins.abs(gflat[i])))
    return worst


# ---- the bridge to the module system -------------------------------------------------

class AutoModule(Module):
    """A layer whose backward pass is derived automatically.

    Subclass it, create Parameters in __init__, and write compute(x) with Tensor
    operations. Use the Parameters directly in the expressions:

        class Affine(me.AutoModule):
            def __init__(self, d_in, d_out):
                self.weight = me.Parameter(me.init.linear_weight(d_in, d_out))
                self.bias = me.Parameter(np.zeros(d_out), decay=False)

            def compute(self, x):
                return x @ self.weight + self.bias

    The result is an ordinary Module: it works inside Sequential and Residual, with the
    optimisers, the Trainer, checkpoints and gradcheck. Sub-modules may be called inside
    compute; other AutoModules can be called any number of times, a layer with a
    hand-written backward pass once per forward pass (it keeps one set of caches).
    """

    def compute(self, x):
        raise NotImplementedError

    def forward(self, x):
        x = np.asarray(x)
        track = config.grad_enabled()
        xt = Tensor(x, requires_grad=track and np.issubdtype(x.dtype, np.floating))
        out = as_tensor(self.compute(xt))
        self._c_graph = (xt, out) if track else None
        return out.data

    def backward(self, grad):
        xt, out = self._c_graph
        if out.requires_grad:
            out.backward(grad)
        return xt.grad


def call_module(module, x):
    """What Module.__call__ does when it is handed a Tensor: an AutoModule continues the
    graph; any other layer becomes one node whose backward step is its own backward()."""
    if isinstance(module, AutoModule):
        return module.compute(x)
    out = Tensor(module.forward(x.data))
    if config.grad_enabled() and (x.requires_grad or next(iter(module.named_parameters()), None) is not None):
        out.requires_grad, out._parents, out._backward = True, (x,), lambda g: (module.backward(g),)
    return out
