"""Automatic differentiation: every operation against finite differences, and against the
engine's hand-written backward passes."""
import math
import numpy as np
import pytest
import mitsu_engine as me
from mitsu_engine import autograd as ag, functional as F
from mitsu_engine.blocks import rope_tables

TOL = 1e-6


@pytest.fixture(autouse=True)
def float64():
    me.set_dtype(np.float64)
    me.manual_seed(0)
    yield
    me.set_dtype(np.float32)


R = np.random.default_rng(3)
f = lambda *shape, scale=1.0: R.standard_normal(shape) * scale
pos = lambda *shape: R.uniform(0.5, 2.0, shape)

OPS = [
    ("add, broadcast", lambda a, b: a + b, [f(3, 4), f(4)]),
    ("sub, broadcast", lambda a, b: a - b, [f(3, 1), f(1, 4)]),
    ("mul, broadcast", lambda a, b: a * b, [f(2, 3, 4), f(3, 1)]),
    ("div", lambda a, b: a / b, [f(3, 4), pos(3, 4)]),
    ("number on the left", lambda a: 2.0 - a + 3.0 * a - 1.0 / (a * a + 1.0), [f(3, 4)]),
    ("neg", lambda a: -a, [f(3, 4)]),
    ("pow, number exponent", lambda a: a ** 3, [f(3, 4)]),
    ("pow, tensor exponent", lambda a, b: a ** b, [pos(3, 4), f(3, 4)]),
    ("pow, number base", lambda a: 2.0 ** a, [f(3, 4)]),
    ("matmul", lambda a, b: a @ b, [f(3, 4), f(4, 5)]),
    ("matmul, batched by matrix", lambda a, b: a @ b, [f(2, 3, 4), f(4, 5)]),
    ("matmul, batched both", lambda a, b: a @ b, [f(2, 3, 4), f(2, 4, 5)]),
    ("matmul, matrix by vector", lambda a, b: a @ b, [f(3, 4), f(4)]),
    ("matmul, vector by matrix", lambda a, b: a @ b, [f(4), f(4, 5)]),
    ("exp", ag.exp, [f(3, 4)]),
    ("log", ag.log, [pos(3, 4)]),
    ("sqrt", ag.sqrt, [pos(3, 4)]),
    ("tanh", ag.tanh, [f(3, 4)]),
    ("sigmoid", ag.sigmoid, [f(3, 4, scale=2)]),
    ("softplus", ag.softplus, [f(3, 4, scale=3)]),
    ("mitsu", ag.mitsu, [f(3, 4, scale=3)]),
    ("mitsu c=0.7", lambda a: ag.mitsu(a, 0.7), [f(3, 4, scale=3)]),
    ("relu", ag.relu, [f(3, 4) + 0.2]),
    ("silu", ag.silu, [f(3, 4, scale=2)]),
    ("gelu", ag.gelu, [f(3, 4, scale=2)]),
    ("abs", ag.abs, [f(3, 4) + 0.3]),
    ("maximum", ag.maximum, [f(3, 4), f(3, 4)]),
    ("minimum, broadcast", ag.minimum, [f(3, 4), f(4)]),
    ("where", lambda a, b: ag.where(np.arange(12).reshape(3, 4) % 2 == 0, a, b), [f(3, 4), f(3, 4)]),
    ("sum, all", lambda a: a.sum(), [f(3, 4)]),
    ("sum, axis", lambda a: a.sum(axis=1), [f(2, 3, 4)]),
    ("sum, axes, keepdims", lambda a: a.sum(axis=(0, 2), keepdims=True), [f(2, 3, 4)]),
    ("mean, all", lambda a: a.mean(), [f(3, 4)]),
    ("mean, axis", lambda a: a.mean(axis=-1, keepdims=True) * a, [f(2, 3, 4)]),
    ("max, axis", lambda a: a.max(axis=1), [f(3, 5)]),
    ("max, all", lambda a: a.max(), [f(3, 5)]),
    ("reshape", lambda a: a.reshape(4, 6) @ a.reshape(6, 4), [f(2, 3, 4)]),
    ("transpose", lambda a: a.transpose(2, 0, 1) * 2.0, [f(2, 3, 4)]),
    ("T and swapaxes", lambda a: a.T @ a.swapaxes(0, 1).T, [f(3, 4)]),
    ("flatten", lambda a: a.flatten().sum(axis=1), [f(2, 3, 4)]),
    ("slice", lambda a: a[1:, ::2] * a[:-1, 1::2], [f(4, 6)]),
    ("ellipsis slice", lambda a: a[..., 0::2] - a[..., 1::2], [f(2, 3, 4)]),
    ("integer-array index with repeats", lambda a: a[np.array([0, 2, 2, 1, 0])], [f(3, 4)]),
    ("two integer arrays", lambda a: a[np.arange(3), np.array([1, 0, 3])], [f(3, 4)]),
    ("concatenate", lambda a, b: ag.concatenate([a, b, a], axis=1), [f(2, 3), f(2, 2)]),
    ("stack", lambda a, b: ag.stack([a, b], axis=-1).reshape(2, 6), [f(2, 3), f(2, 3)]),
    ("softmax", lambda a: ag.softmax(a, axis=-1), [f(3, 5)]),
    ("softmax, other axis", lambda a: ag.softmax(a, axis=0), [f(3, 5)]),
    ("log_softmax", ag.log_softmax, [f(3, 5)]),
    ("cross_entropy", lambda a: ag.cross_entropy(a, np.array([[1, 0, 4], [2, 2, 3]])), [f(2, 3, 5)]),
    ("mse", lambda a, b: ag.mse(a, b), [f(3, 4), f(3, 4)]),
    ("conv2d", lambda x, w, b: ag.conv2d(x, w, b, padding=1), [f(2, 3, 5, 5), f(4, 3, 3, 3), f(4)]),
    ("conv2d, stride 2, no bias", lambda x, w: ag.conv2d(x, w, stride=2), [f(2, 2, 7, 7), f(3, 2, 3, 3)]),
    ("max_pool2d", lambda x: ag.max_pool2d(x, 2), [f(2, 3, 6, 6)]),
    ("avg_pool2d", lambda x: ag.avg_pool2d(x, 3, 2), [f(2, 3, 7, 7)]),
    ("a small network", lambda x, w1, w2: ag.cross_entropy(ag.mitsu(x @ w1) @ w2, np.array([0, 2, 1, 1])), [f(4, 5), f(5, 6), f(6, 3)]),
]


@pytest.mark.parametrize("name,fn,arrays", OPS, ids=[o[0] for o in OPS])
def test_operation_against_finite_differences(name, fn, arrays):
    assert ag.check(fn, *arrays) < TOL


def test_values_match_numpy():
    a, b = f(3, 4), pos(3, 4)
    x, y = ag.Tensor(a), ag.Tensor(b)
    np.testing.assert_allclose((x * y + x / y - x ** 2).data, a * b + a / b - a ** 2)
    np.testing.assert_allclose(ag.mitsu(x).data, F.mitsu(a))
    np.testing.assert_allclose((x @ y.T).data, a @ b.T)
    np.testing.assert_allclose(ag.softmax(x).data.sum(-1), 1.0)
    assert x.shape == (3, 4) and x.ndim == 2 and x.size == 12 and len(x) == 3 and "Tensor" in repr(x)


def test_a_value_used_twice_gets_both_gradients():
    x = ag.Tensor(np.array([1.5, -2.0]), requires_grad=True)
    (x * x + x).sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)
    x.zero_grad()
    assert x.grad is None


def test_numpy_array_on_the_left_stays_in_the_graph():
    x = ag.Tensor(f(3), requires_grad=True)
    y = np.ones(3) * x + np.arange(3.0) - x
    assert isinstance(y, ag.Tensor) and y.requires_grad


def test_python_numbers_do_not_change_the_precision():
    x = ag.Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    y = (x * 0.5 + 1.0) / 3.0 - 2.0
    assert y.dtype == np.float32 and ag.mitsu(y).dtype == np.float32
    y.sum().backward()
    assert x.grad.dtype == np.float32


def test_no_grad_records_nothing():
    x = ag.Tensor(f(3), requires_grad=True)
    with me.no_grad():
        y = ag.exp(x) * x
    assert not y.requires_grad and y._parents == ()
    assert not x.detach().requires_grad


def test_backward_needs_a_scalar_or_a_gradient():
    x = ag.Tensor(f(3), requires_grad=True)
    with pytest.raises(ValueError):
        (x * 2).backward()
    (x * 2).backward(np.ones(3))
    np.testing.assert_allclose(x.grad, 2.0)
    with pytest.raises(RuntimeError):
        ag.Tensor(f(3)).sum().backward()


def test_max_shares_the_gradient_between_ties():
    x = ag.Tensor(np.array([[1.0, 3.0, 3.0], [2.0, 0.0, 1.0]]), requires_grad=True)
    x.max(axis=1).sum().backward()
    np.testing.assert_allclose(x.grad, [[0, 0.5, 0.5], [1, 0, 0]])


def test_dropout():
    x = ag.Tensor(np.ones((50, 50)), requires_grad=True)
    y = ag.dropout(x, 0.4, rng=0)
    assert 0.5 < (y.data == 0).mean() / 0.4 < 1.5 and abs(y.data.mean() - 1.0) < 0.1
    y.sum().backward()
    np.testing.assert_allclose(x.grad, y.data)
    assert ag.dropout(x, 0.4, training=False) is x


def test_grad_helper():
    g = ag.grad(lambda x: (x ** 2).sum(), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(g, [2, 4, 6])
    ga, gb = ag.grad(lambda a, b: (a * b).sum(), np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(ga, [3, 4]), np.testing.assert_allclose(gb, [1, 2])


# ---- the mother function -----------------------------------------------------------------

def mother(x, y, z):
    return (x ** (y + z) + y ** (x + z) + z ** (x + y)) / (x + y + z)


def test_mother_function_with_three_live_seats():
    x, y, z = pos(8), pos(8), pos(8)
    assert ag.check(mother, x, y, z) < TOL
    gx, gy, gz = ag.grad(lambda a, b, c: mother(a, b, c).sum(), x, y, z)
    # the function is symmetric: swapping two seats swaps their gradients
    hx, hy, hz = ag.grad(lambda a, b, c: mother(a, b, c).sum(), y, x, z)
    np.testing.assert_allclose(gx, hy, rtol=1e-12), np.testing.assert_allclose(gy, hx, rtol=1e-12), np.testing.assert_allclose(gz, hz, rtol=1e-12)


def test_mitsu_is_the_mother_function_with_two_seats_at_one():
    u = f(200, scale=3)
    t = ag.Tensor(u, requires_grad=True)
    y = mother(ag.softplus(t), 1.0, 1.0) - 1.0                # f(s, 1, 1) - 1, differentiated automatically
    y.sum().backward()
    np.testing.assert_allclose(y.data, F.mitsu(u), rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(t.grad, F.mitsu_grad(u), rtol=1e-9, atol=1e-13)   # the hand-derived derivative


# ---- parameters and modules --------------------------------------------------------------

def test_parameters_receive_their_gradient():
    w = me.Parameter(f(4, 3))
    x = f(5, 4)
    ((ag.Tensor(x) @ w) * (w.data.sum() + ag.sum(w))).sum().backward()     # the parameter is used twice
    expected = ag.grad(lambda t: ((ag.Tensor(x) @ t) * (w.data.sum() + ag.sum(t))).sum(), w.data)
    np.testing.assert_allclose(w.grad, expected, rtol=1e-12)


class Affine(me.AutoModule):
    """Linear, with the gradient left to the engine."""

    def __init__(self, d_in, d_out, rng=None):
        self.weight = me.Parameter(me.init.linear_weight(d_in, d_out, rng=rng))
        self.bias = me.Parameter(np.zeros(d_out), decay=False)

    def compute(self, x):
        return x @ self.weight + self.bias


class AutoGLU(me.AutoModule):
    def __init__(self, dim, hidden, rng=None):
        self.w1, self.w3, self.w2 = (me.Parameter(me.init.linear_weight(a, b, rng=rng)) for a, b in ((dim, hidden), (dim, hidden), (hidden, dim)))

    def compute(self, x):
        return ((ag.mitsu(x @ self.w1) * me.init.GATE_MULT) * (x @ self.w3)) @ self.w2


def test_automodule_matches_the_hand_written_layer():
    auto, hand = Affine(5, 4, rng=1), me.Linear(5, 4, rng=1)
    auto.bias.data[:] = hand.bias.data[:] = f(4)
    assert np.array_equal(auto.weight.data, hand.weight.data)
    x, g = f(3, 5), f(3, 4)
    np.testing.assert_allclose(auto(x), hand(x), rtol=1e-13)
    np.testing.assert_allclose(auto.backward(g), hand.backward(g), rtol=1e-12)
    np.testing.assert_allclose(auto.weight.grad, hand.weight.grad, rtol=1e-12)
    np.testing.assert_allclose(auto.bias.grad, hand.bias.grad, rtol=1e-12)
    assert me.gradcheck(auto, x)[0] < 1e-5 and me.gradcheck(AutoGLU(5, 7), f(2, 3, 5))[0] < 1e-5


def test_automodule_glu_matches_mitsuglu():
    hand = me.MitsuGLU(6, 8, rng=2)
    auto = AutoGLU(6, 8)
    for name in ("w1", "w2", "w3"):
        getattr(auto, name).data[...] = getattr(hand, name).weight.data
    x, g = f(2, 3, 6), f(2, 3, 6)
    np.testing.assert_allclose(auto(x), hand(x), rtol=1e-12)
    np.testing.assert_allclose(auto.backward(g), hand.backward(g), rtol=1e-10)
    for name in ("w1", "w2", "w3"):
        np.testing.assert_allclose(getattr(auto, name).grad, getattr(hand, name).weight.grad, rtol=1e-10)


def test_automodule_composes_with_hand_written_layers():
    class Mixed(me.AutoModule):                              # hand-written layers called inside compute
        def __init__(self):
            self.inner, self.norm, self.scale = me.Linear(5, 6, rng=0), me.LayerNorm(6), me.Parameter(np.ones(6))

        def compute(self, x):
            return ag.mitsu(self.norm(self.inner(x))) * self.scale + x.sum(axis=-1, keepdims=True)

    net = me.Sequential(me.Linear(4, 5), me.Residual(Affine(5, 5)), Mixed(), AutoGLU(6, 4), me.Linear(6, 3))
    assert [n for n, _ in net.named_parameters()][:3] == ["layers.0.weight", "layers.0.bias", "layers.1.fn.weight"]
    assert me.gradcheck(net, f(3, 4))[0] < 1e-5
    net.free_caches()
    assert net.layers[2]._c_graph is None


def test_automodule_with_integer_input():
    class Lookup(me.AutoModule):
        def __init__(self):
            self.table = me.Parameter(f(7, 4))

        def compute(self, idx):
            return ag.as_tensor(self.table)[idx.data] * 2.0

    m = Lookup()
    idx = np.array([[1, 1, 3], [0, 6, 1]])
    out = m(idx)
    assert m.backward(np.ones_like(out)) is None
    np.testing.assert_allclose(m.table.grad[1], 6.0), np.testing.assert_allclose(m.table.grad[2], 0.0)
    assert me.gradcheck(m, idx)[0] < 1e-5


def test_automodule_trains_and_saves(tmp_path):
    me.set_dtype(np.float32)
    X, y = me.make_spirals(200, classes=2, rng=0)
    build = lambda: me.Sequential(Affine(2, 32, rng=0), me.MITSU(gain=3.0), Affine(32, 32, rng=1), me.MITSU(gain=3.0), Affine(32, 2, rng=2))
    net = build()
    trainer = me.Trainer(net, me.AdamW(net.parameters(), lr=1e-2), log=None)
    trainer.fit(me.DataLoader(X, y, 64, rng=0), epochs=120)
    acc = trainer.evaluate(me.DataLoader(X, y, 200, shuffle=False))["metric"]
    assert acc > 0.9
    me.save_checkpoint(str(tmp_path / "auto.npz"), net)
    other = build()
    me.load_checkpoint(str(tmp_path / "auto.npz"), other)
    np.testing.assert_array_equal(other(X[:5]), net(X[:5]))


# ---- hand-written backward passes against automatic differentiation ------------------------

def rms(x, gain, eps=1e-6):
    return x / ag.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * gain


def test_normalisation_layers_against_autograd():
    x, g = f(2, 3, 6, scale=2), f(2, 3, 6)
    for layer, formula in [
        (me.RMSNorm(6), lambda t, l: rms(t, l.gain)),
        (me.LayerNorm(6), lambda t, l: (t - t.mean(axis=-1, keepdims=True)) / ag.sqrt(((t - t.mean(axis=-1, keepdims=True)) ** 2).mean(axis=-1, keepdims=True) + l.eps) * l.gain + l.bias),
    ]:
        layer.gain.data[:] = pos(6)
        layer(x)
        gx = layer.backward(g)
        hand = {n: p.grad.copy() for n, p in layer.named_parameters()}
        layer.zero_grad()
        t = ag.Tensor(x, requires_grad=True)
        out = formula(t, layer)
        out.backward(g)
        np.testing.assert_allclose(out.data, layer(x), rtol=1e-12)
        np.testing.assert_allclose(t.grad, gx, rtol=1e-9, atol=1e-13)
        for n, p in layer.named_parameters():
            np.testing.assert_allclose(p.grad, hand[n], rtol=1e-9, atol=1e-13)


def lm_logits(model, idx):
    """MitsuLM's forward pass written with Tensor operations: no hand-written gradient anywhere."""
    cfg = model.config
    B, T = idx.shape
    D, H = cfg.d_model, cfg.n_heads
    cos, sin = rope_tables(T, D // H, cfg.rope_base, 0, np.float64)
    mask = np.triu(np.full((T, T), -np.inf), 1)

    def rotate(t):
        a, b = t[..., 0::2], t[..., 1::2]
        return ag.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(t.shape)

    table = ag.as_tensor(model.tok.weight)
    x = table[idx]
    for blk in model.blocks:
        h = rms(x, blk.norm1.gain)
        qkv = h @ blk.attn.qkv.weight
        q, k, v = (qkv[..., i * D:(i + 1) * D].reshape(B, T, H, D // H).transpose(0, 2, 1, 3) for i in range(3))
        att = ag.softmax((rotate(q) @ rotate(k).transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(D // H)) + mask)
        x = x + (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D) @ blk.attn.proj.weight
        h = rms(x, blk.norm2.gain)
        x = x + ((ag.mitsu(h @ blk.ffn.w1.weight) * cfg.gate_mult) * (h @ blk.ffn.w3.weight)) @ blk.ffn.w2.weight
    x = rms(x, model.norm.gain)
    return x @ (table.T if cfg.tie_embeddings else ag.as_tensor(model.head.weight))


@pytest.mark.parametrize("tied", [True, False])
def test_the_language_model_differentiated_two_ways(tied):
    """The whole MitsuLM: hand-written backward against automatic differentiation."""
    model = me.MitsuLM(me.LMConfig(vocab_size=37, context=12, d_model=16, n_layers=2, n_heads=2, tie_embeddings=tied), rng=0)
    idx, tgt = R.integers(0, 37, (3, 12)), R.integers(0, 37, (3, 12))
    logits = model(idx)
    loss, grad = me.CrossEntropy()(logits, tgt)
    model.backward(grad)
    hand = {n: p.grad.copy() for n, p in model.named_parameters()}
    model.zero_grad()

    auto_logits = lm_logits(model, idx)
    auto_loss = ag.cross_entropy(auto_logits, tgt)
    auto_loss.backward()
    np.testing.assert_allclose(auto_logits.data, logits, rtol=1e-11, atol=1e-13)
    assert abs(auto_loss.item() - loss) < 1e-13
    for n, p in model.named_parameters():
        assert np.abs(p.grad - hand[n]).max() <= 1e-11 * max(np.abs(hand[n]).max(), 1e-30) + 1e-15, n


def test_conv_layers_against_autograd():
    x, g = f(2, 3, 6, 6), None
    conv, norm = me.Conv2d(3, 4, 3, stride=1, padding=1, rng=0), me.GroupNorm(2, 4)
    norm.gain.data[:], norm.bias.data[:] = pos(4), f(4)
    net = me.Sequential(conv, norm, me.MITSU(), me.MaxPool2d(2))
    y = net(x)
    g = f(*y.shape)
    gx = net.backward(g)
    hand = {n: p.grad.copy() for n, p in net.named_parameters()}
    net.zero_grad()

    t = ag.Tensor(x, requires_grad=True)
    h = ag.conv2d(t, conv.weight, conv.bias, padding=1)
    hg = h.reshape(2, 2, -1)
    hg = (hg - hg.mean(axis=-1, keepdims=True)) / ag.sqrt(((hg - hg.mean(axis=-1, keepdims=True)) ** 2).mean(axis=-1, keepdims=True) + norm.eps)
    h = hg.reshape(h.shape) * ag.as_tensor(norm.gain).reshape(1, 4, 1, 1) + ag.as_tensor(norm.bias).reshape(1, 4, 1, 1)
    out = ag.max_pool2d(ag.mitsu(h), 2)
    out.backward(g)
    np.testing.assert_allclose(out.data, y, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(t.grad, gx, rtol=1e-9, atol=1e-13)
    for n, p in net.named_parameters():
        np.testing.assert_allclose(p.grad, hand[n], rtol=1e-9, atol=1e-13)
