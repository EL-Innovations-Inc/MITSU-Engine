"""Optimisers, clipping and schedules."""
import numpy as np
import pytest
import mitsu_engine as me


def quadratic_problem():
    r = np.random.default_rng(0)
    X = r.standard_normal((64, 6)).astype(np.float32)
    T = X @ r.standard_normal((6, 3)).astype(np.float32)
    return X, T


@pytest.mark.parametrize("make", [
    lambda p: me.SGD(p, lr=0.05), lambda p: me.SGD(p, lr=0.02, momentum=0.9), lambda p: me.SGD(p, lr=0.02, momentum=0.9, nesterov=True),
    lambda p: me.Adam(p, lr=0.03), lambda p: me.AdamW(p, lr=0.03, weight_decay=0.0)], ids=["sgd", "momentum", "nesterov", "adam", "adamw"])
def test_optimisers_solve_a_linear_regression(make):
    X, T = quadratic_problem()
    model = me.Linear(6, 3, rng=1)
    opt, loss_fn = make(model.parameters()), me.MSE()
    first = None
    for _ in range(400):
        opt.zero_grad()
        loss, g = loss_fn(model(X), T)
        first = first or loss
        model.backward(g)
        opt.step()
    assert loss < 1e-3 * first


def test_adam_first_step_has_size_lr():
    p = me.Parameter(np.zeros(5))
    p.grad[:] = [3.0, -0.1, 7.0, -20.0, 0.5]
    me.Adam([p], lr=0.01).step()
    assert np.allclose(np.abs(p.data), 0.01, rtol=1e-4)


def test_adamw_decays_weights_but_not_biases():
    lin = me.Linear(4, 4, rng=0)
    lin.bias.data[:] = 1.0
    w0 = lin.weight.data.copy()
    me.AdamW(lin.parameters(), lr=0.1, weight_decay=0.5).step()      # gradients are zero
    assert np.allclose(lin.weight.data, w0 * (1 - 0.1 * 0.5)) and np.allclose(lin.bias.data, 1.0)


def test_clip_grad_norm():
    a, b = me.Parameter(np.zeros(3)), me.Parameter(np.zeros(4))
    a.grad[:] = 3.0; b.grad[:] = 4.0
    before = me.clip_grad_norm([a, b], 1.0)
    assert abs(before - np.sqrt(3 * 9 + 4 * 16)) < 1e-5 and abs(me.grad_norm([a, b]) - 1.0) < 1e-5
    assert abs(me.clip_grad_norm([a, b], 10.0) - 1.0) < 1e-5 and abs(me.grad_norm([a, b]) - 1.0) < 1e-5


def test_schedules():
    s = me.WarmupCosine(1.0, warmup=10, total=110, min_lr=0.1)
    assert abs(s(0) - 0.1) < 1e-9 and abs(s(9) - 1.0) < 1e-9 and abs(s(60) - 0.55) < 1e-9 and abs(s(110) - 0.1) < 1e-9 and abs(s(10 ** 6) - 0.1) < 1e-9
    assert me.ConstantLR(0.3)(123) == 0.3 and abs(me.StepLR(1.0, 10, 0.5)(25) - 0.25) < 1e-12


@pytest.mark.parametrize("make", [lambda p: me.SGD(p, lr=0.05, momentum=0.9), lambda p: me.AdamW(p, lr=0.01)], ids=["sgd", "adamw"])
def test_optimiser_state_round_trip_continues_identically(make):
    X, T = quadratic_problem()

    def run(model, opt, n):
        for _ in range(n):
            opt.zero_grad()
            _, g = me.MSE()(model(X), T)
            model.backward(g)
            opt.step()

    a = me.Linear(6, 3, rng=1); oa = make(a.parameters()); run(a, oa, 10)
    b = me.Linear(6, 3, rng=9); b.load_state_dict(a.state_dict()); ob = make(b.parameters()); ob.load_state_dict(oa.state_dict())
    run(a, oa, 5); run(b, ob, 5)
    assert np.array_equal(a.weight.data, b.weight.data) and oa.steps == ob.steps == 15
