"""The module system: parameter discovery, state, modes, seeding."""
import numpy as np
import pytest
import mitsu_engine as me


def test_parameter_counts():
    net = me.DeepMitsuNet(10, 3, width=16, depth=5, hidden=32)
    per_block = 2 * 16 + (16 * 32 + 32) + (32 * 16 + 16)
    assert net.num_parameters() == (10 * 16 + 16) + 5 * per_block + 2 * 16 + (16 * 3 + 3)
    gpt = me.MitsuGPT(20, 8, d_model=12, n_layers=2, n_heads=3, ffn="glu", hidden=32)
    per_block = 2 * 12 + 12 * 36 + 12 * 12 + 2 * 12 + 3 * 12 * 32
    assert gpt.num_parameters() == 20 * 12 + 8 * 12 + 2 * per_block + 2 * 12 + 12 * 20


def test_named_parameters_are_unique_and_ordered():
    net = me.DeepMitsuNet(4, 2, width=8, depth=3)
    names = [n for n, _ in net.named_parameters()]
    assert len(names) == len(set(names)) and names[0] == "stem.weight" and names[-1] == "head.bias"
    assert names == [n for n, _ in net.named_parameters()]


def test_state_dict_round_trip_and_errors():
    a, b = me.MLP([4, 6, 3], rng=1), me.MLP([4, 6, 3], rng=2)
    x = np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32)
    assert not np.allclose(a(x), b(x))
    b.load_state_dict(a.state_dict())
    assert np.array_equal(a(x), b(x))
    with pytest.raises(KeyError):
        me.MLP([4, 6, 6, 3]).load_state_dict(a.state_dict())
    with pytest.raises(ValueError):
        me.MLP([4, 7, 3]).load_state_dict(a.state_dict())


def test_same_seed_same_weights():
    a, b, c = me.DeepMitsuNet(4, 2, 8, 2, rng=7), me.DeepMitsuNet(4, 2, 8, 2, rng=7), me.DeepMitsuNet(4, 2, 8, 2, rng=8)
    assert all(np.array_equal(p.data, q.data) for p, q in zip(a.parameters(), b.parameters()))
    assert not np.array_equal(a.stem.weight.data, c.stem.weight.data)
    me.manual_seed(3); d = me.Linear(5, 5)
    me.manual_seed(3); e = me.Linear(5, 5)
    assert np.array_equal(d.weight.data, e.weight.data)


def test_train_eval_and_dropout():
    drop = me.Dropout(0.5, rng=0)
    x = np.ones((200, 50), dtype=np.float32)
    y = drop(x)
    assert 0.35 < (y == 0).mean() < 0.65 and abs(y.mean() - 1.0) < 0.1
    assert np.array_equal(drop.backward(np.ones_like(x)), y)
    net = me.Sequential(me.Linear(50, 50), drop).eval()
    assert not drop.training and np.array_equal(drop(x), x)
    net.train()
    assert drop.training


def test_no_grad_keeps_no_caches_and_free_caches_clears_them():
    lin = me.Linear(4, 4)
    x = np.zeros((2, 4), dtype=np.float32)
    with me.no_grad():
        lin(x)
    assert lin._c_x is None
    lin(x)
    assert lin._c_x is not None
    lin.free_caches()
    assert lin._c_x is None
    net = me.DeepMitsuNet(4, 2, width=8, depth=2, block="glu")
    net(np.zeros((2, 4), dtype=np.float32))
    net.free_caches()
    assert all(v is None for m in net.modules() for k, v in m.__dict__.items() if k == "_c" or k.startswith("_c_"))


def test_zero_grad_and_accumulation():
    lin = me.Linear(3, 2)
    x = np.ones((4, 3), dtype=np.float32)
    lin(x); lin.backward(np.ones((4, 2), dtype=np.float32))
    first = lin.weight.grad.copy()
    lin(x); lin.backward(np.ones((4, 2), dtype=np.float32))
    assert np.allclose(lin.weight.grad, 2 * first)
    lin.zero_grad()
    assert not lin.weight.grad.any() and not lin.bias.grad.any()


def test_dtype_setting():
    assert me.Linear(2, 2).weight.data.dtype == np.float32
    me.set_dtype(np.float64)
    assert me.Linear(2, 2).weight.data.dtype == np.float64
    with pytest.raises(ValueError):
        me.set_dtype(np.int32)


def test_repr_and_summary():
    net = me.DeepMitsuNet(4, 2, width=8, depth=2)
    assert "DeepMitsuNet" in repr(net) and "MITSU" in repr(net)
    assert f"{net.num_parameters():,d}" in net.summary()


def test_branch_initialisation_keeps_the_stream_bounded_at_depth():
    x = np.random.default_rng(0).standard_normal((256, 32)).astype(np.float32)
    for block in ("mlp", "glu"):
        net = me.DeepMitsuNet(32, 4, width=64, depth=200, block=block, rng=0)
        with me.no_grad():
            h = net.stem(x)
            start = np.sqrt((h ** 2).mean())
            for blk in net.blocks:
                h = blk(h)
        assert 0.5 * start < np.sqrt((h ** 2).mean()) < 3.0 * start


def test_activation_gain_matches_monte_carlo():
    z = np.random.default_rng(0).standard_normal(2_000_000)
    assert abs(me.init.activation_gain(me.MITSU()) - 1 / np.sqrt((me.functional.mitsu(z) ** 2).mean())) < 0.02
    assert abs(me.init.activation_gain(me.ReLU()) - np.sqrt(2)) < 1e-3
    assert abs(np.std(me.functional.silu(z)) / np.std(me.functional.mitsu(z)) - me.init.GATE_MULT) < 0.02
