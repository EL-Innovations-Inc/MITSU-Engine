"""Numerical gradient checking: the test every hand-written backward must pass."""
import numpy as np
from . import config


def gradcheck(module, x, n_checks=24, eps=1e-6, rng=0):
    """Compare backward() with central finite differences.

    A random projection R turns the output into the scalar L = sum(module(x) * R).
    The analytic gradients of L (input and every parameter) are compared with
    (L(+eps) - L(-eps)) / (2 eps) at up to `n_checks` random entries each.
    Returns (worst error, where it occurred). The error of one entry is
    |numeric - analytic| / max(|numeric| + |analytic|, 1e-4 * largest gradient in the model),
    so entries whose true gradient is zero (for example a bias added to every attention key)
    are judged on an absolute scale instead of dividing rounding noise by nothing.
    Requires float64 parameters: call set_dtype(numpy.float64) before building the module.
    """
    rng = config.get_rng(rng)
    params = list(module.named_parameters())
    if any(p.data.dtype != np.float64 for _, p in params):
        raise TypeError("gradcheck needs float64 parameters: call set_dtype(np.float64) before building the module")
    was_training = module.training
    module.eval()
    is_float_input = np.issubdtype(np.asarray(x).dtype, np.floating)
    if is_float_input:
        x = np.array(x, dtype=np.float64)
    y = module(x)
    R = rng.standard_normal(y.shape)
    loss = lambda: float((module(x) * R).sum())
    module.zero_grad()
    gx = module.backward(R.copy())
    worst, where = 0.0, ""
    scale = max([float(np.abs(p.grad).max()) for _, p in params] + ([float(np.abs(gx).max())] if (is_float_input and gx is not None) else []) + [1e-30])
    floor = 1e-4 * scale

    def probe(array, grad, label):
        nonlocal worst, where
        flat, gflat = array.reshape(-1), grad.reshape(-1)
        for i in rng.choice(flat.size, size=min(n_checks, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps; up = loss()
            flat[i] = old - eps; down = loss()
            flat[i] = old
            numeric, analytic = (up - down) / (2 * eps), gflat[i]
            err = abs(numeric - analytic) / max(floor, abs(numeric) + abs(analytic))
            if err > worst:
                worst, where = err, label

    if is_float_input and gx is not None:
        probe(x, gx, "input")
    for name, p in params:
        probe(p.data, p.grad.copy(), name)
    module.zero_grad()
    module.train(was_training)
    return worst, where
