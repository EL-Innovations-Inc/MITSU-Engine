"""Weight initialisation.

Measured facts about MITSU that the defaults rest on (z ~ N(0, 1)):
    mean of MITSU(z)      0.010
    std of MITSU(z)       0.184        (SiLU: 0.56, about 3.04 times larger)
    MITSU(0)             -0.079

MITSU is a quiet activation. Two consequences:
  * a branch that ends in MITSU needs its next layer scaled up by about
    1 / 0.184 = 5.43 to hand on a unit-variance signal (`MITSU_GAIN`);
  * a plain stack of Linear -> MITSU with no normalisation does not hold a signal at
    initialisation: with ordinary gains every unit drifts to the constant MITSU(0) within
    about five layers, and with the variance-preserving gain the signal grows without
    limit. Training can rescue a plain stack of about 8 layers; at 16 it stays stuck
    (benchmarks/depth_sweep.py). Deep MITSU networks therefore use normalised residual
    blocks (models.DeepMitsuNet), which train at 256 blocks.
"""
import math
import numpy as np
from . import config

MITSU_STD = 0.1841       # std of MITSU(z), z ~ N(0, 1)
MITSU_GAIN = 5.431       # 1 / MITSU_STD
GATE_MULT = 3.04         # std(SiLU(z)) / std(MITSU(z)): MitsuGLU's default gate gain

_GAINS = {"lecun": 1.0, "xavier": 1.0, "he": math.sqrt(2.0), "mitsu": MITSU_GAIN}


def gain(name):
    if isinstance(name, (int, float)):
        return float(name)
    if name not in _GAINS:
        raise ValueError(f"unknown init '{name}'; choose from {sorted(_GAINS)} or pass a number")
    return _GAINS[name]


def normal(shape, std, rng=None):
    return (config.get_rng(rng).standard_normal(shape) * std).astype(config.get_dtype())


def linear_weight(d_in, d_out, init="lecun", scale=1.0, rng=None):
    """Weights of shape (d_in, d_out) drawn from N(0, (gain * scale)^2 / fan), where fan is
    d_in, or the mean of d_in and d_out for 'xavier'. `init` may also be a number (the gain)."""
    fan = 0.5 * (d_in + d_out) if init == "xavier" else d_in
    return normal((d_in, d_out), gain(init) * scale / math.sqrt(fan), rng)


def activation_gain(activation):
    """1 / sqrt(E[act(z)^2]) for z ~ N(0, 1), by Gauss-Hermite quadrature.

    Initialising the layer AFTER an activation with this gain makes that layer's output
    unit-variance when the activation's input is unit-variance. It works for any
    activation object, including MITSU with a non-default c or gain."""
    x, w = np.polynomial.hermite_e.hermegauss(120)
    w = w / w.sum()
    with config.no_grad():
        y = np.asarray(activation(x.astype(np.float64)), dtype=np.float64)
    return float(1.0 / math.sqrt(float((w * y * y).sum())))
