"""Global settings: floating-point type, random generator, gradient switch."""
import numpy as np

_state = {"dtype": np.float32, "rng": np.random.default_rng(0), "grad": True}


def set_dtype(dtype):
    """Set the floating-point type new parameters are created with (float32 by default).
    Use float64 for gradient checking."""
    dtype = np.dtype(dtype).type
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")
    _state["dtype"] = dtype


def get_dtype():
    return _state["dtype"]


def manual_seed(seed):
    """Reseed the global generator used for weight initialisation, dropout and shuffling
    whenever a component is not given its own `rng`."""
    _state["rng"] = np.random.default_rng(seed)


def get_rng(rng=None):
    """Return `rng` as a numpy Generator: None -> the global one, an int -> a fresh seeded one."""
    if rng is None:
        return _state["rng"]
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


def get_rng_state(rng=None):
    """The state of a generator (the global one by default) as a JSON-serialisable dict."""
    return get_rng(rng).bit_generator.state


def set_rng_state(state, rng=None):
    """Restore a state captured by get_rng_state, so random draws continue exactly."""
    get_rng(rng).bit_generator.state = state


def grad_enabled():
    return _state["grad"]


class no_grad:
    """Context manager: inside it, layers do not keep the caches backward() needs.
    Use it for evaluation and generation to save memory and time."""

    def __enter__(self):
        self._prev = _state["grad"]
        _state["grad"] = False
        return self

    def __exit__(self, *exc):
        _state["grad"] = self._prev
        return False
