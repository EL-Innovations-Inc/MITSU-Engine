"""The maths: MITSU against its definition, its limits, and numerical safety."""
import math
import numpy as np
import pytest
from mitsu_engine import functional as F

U = np.linspace(-20.0, 20.0, 4001)


def test_mitsu_is_the_mother_function_with_two_seats_at_one():
    x, y, z = F.softplus(U), 1.0, 1.0
    f = (x ** (y + z) + y ** (x + z) + z ** (x + y)) / (x + y + z)
    assert np.allclose(f - 1.0, F.mitsu(U), atol=1e-12)


@pytest.mark.parametrize("i,j,k", [(1, 1, 1), (2, 1, 1), (1, 3, 1), (8, 1, 1), (1, 4, 4)])
def test_c_family_is_the_weighted_mother_function(i, j, k):
    s = F.softplus(U)
    f = (i * s ** 2 + j + k) / (i * s + j + k)
    assert np.allclose(f - 1.0, F.mitsu(U, c=(j + k) / i), atol=1e-12)


def test_dip_and_tail():
    s_star = math.sqrt(6) - 2
    u_star = math.log(math.expm1(s_star))
    fine = np.linspace(-2, 1, 300001)
    assert abs(F.mitsu(fine).min() - s_star * (s_star - 1) / (s_star + 2)) < 1e-9
    assert abs(fine[F.mitsu(fine).argmin()] - u_star) < 1e-4
    assert abs(F.mitsu(np.array(-40.0))) < 1e-12
    assert abs(F.mitsu(np.array(1e6)) - (1e6 - 3)) < 1e-5


@pytest.mark.parametrize("fn,dfn", [(F.mitsu, F.mitsu_grad), (lambda u: F.mitsu(u, 0.5), lambda u: F.mitsu_grad(u, 0.5)),
                                     (F.silu, F.silu_grad), (F.gelu, F.gelu_grad), (F.tanh, F.tanh_grad)])
def test_derivatives_match_finite_differences(fn, dfn):
    h = 1e-5
    assert np.abs((fn(U + h) - fn(U - h)) / (2 * h) - dfn(U)).max() < 1e-8


def test_mitsu_slope_is_bounded():
    d = F.mitsu_grad(np.linspace(-60, 60, 240001))
    assert -0.05 < d.min() and d.max() < 1.0


def test_no_overflow_at_extremes_or_in_half_precision():
    extreme = np.array([-1e30, -1e4, -745.0, 0.0, 745.0, 1e4, 1e30])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for f in (F.softplus, F.sigmoid, F.mitsu, F.mitsu_grad, F.silu, F.silu_grad):
            assert np.isfinite(f(extreme)).all()
        half = np.array([-60000, -300, 0, 255, 256, 257, 258, 1000, 60000], dtype=np.float16)
        assert np.isfinite(F.mitsu(half)).all() and np.isfinite(F.mitsu_grad(half)).all()
    assert abs(float(F.mitsu(np.array([257], dtype=np.float16))[0]) - 254.0) < 1.0


def test_softmax_and_log_softmax_agree_and_survive_large_logits():
    z = np.random.default_rng(0).standard_normal((5, 9)) * 500
    p = F.softmax(z)
    assert np.allclose(p.sum(-1), 1.0) and np.isfinite(F.log_softmax(z)).all()
    assert np.allclose(np.exp(F.log_softmax(z)), p)
