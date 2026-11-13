"""Every hand-written backward pass against finite differences (float64)."""
import numpy as np
import pytest
import mitsu_engine as me

TOL = 1e-5


@pytest.fixture(autouse=True)
def float64():
    me.set_dtype(np.float64)
    me.manual_seed(0)
    yield
    me.set_dtype(np.float32)


def cases():
    r = np.random.default_rng(1)
    f = lambda *shape, scale=1.0: r.standard_normal(shape) * scale
    return [
        ("Linear", lambda: me.Linear(5, 4), f(3, 5)),
        ("Linear on 3-D input, no bias", lambda: me.Linear(5, 4, bias=False), f(2, 3, 5)),
        ("LayerNorm", lambda: me.LayerNorm(6), f(2, 3, 6, scale=2)),
        ("MITSU", lambda: me.MITSU(), f(4, 5, scale=2)),
        ("MITSU c=0.5 gain=3", lambda: me.MITSU(c=0.5, gain=3.0), f(4, 5, scale=2)),
        ("ReLU", lambda: me.ReLU(), f(4, 5) + 0.3),
        ("GELU", lambda: me.GELU(), f(4, 5, scale=2)),
        ("SiLU", lambda: me.SiLU(), f(4, 5, scale=2)),
        ("Tanh", lambda: me.Tanh(), f(4, 5)),
        ("Sequential", lambda: me.Sequential(me.Linear(5, 6), me.MITSU(), me.Linear(6, 3)), f(4, 5)),
        ("Residual", lambda: me.Residual(me.Sequential(me.LayerNorm(5), me.Linear(5, 5), me.MITSU())), f(4, 5)),
        ("FeedForward", lambda: me.FeedForward(6, 9), f(3, 6)),
        ("MitsuGLU", lambda: me.MitsuGLU(6, 8), f(2, 3, 6)),
        ("MitsuGLU bias, raw gate", lambda: me.MitsuGLU(6, 8, bias=True, gate_mult=1.0), f(3, 6)),
        ("Attention causal", lambda: me.MultiHeadSelfAttention(8, 2, causal=True), f(2, 5, 8)),
        ("Attention full, bias", lambda: me.MultiHeadSelfAttention(8, 4, causal=False, bias=True), f(2, 5, 8)),
        ("TransformerBlock glu", lambda: me.TransformerBlock(8, 2, ffn="glu"), f(2, 5, 8)),
        ("TransformerBlock mlp", lambda: me.TransformerBlock(8, 2, ffn="mlp"), f(2, 5, 8)),
        ("mitsu_block mlp", lambda: me.mitsu_block(6, kind="mlp"), f(3, 6)),
        ("mitsu_block glu", lambda: me.mitsu_block(6, kind="glu"), f(3, 6)),
        ("MLP", lambda: me.MLP([5, 7, 6, 3]), f(4, 5)),
        ("DeepMitsuNet mlp x6", lambda: me.DeepMitsuNet(5, 3, width=8, depth=6), f(4, 5)),
        ("DeepMitsuNet glu x4", lambda: me.DeepMitsuNet(5, 3, width=9, depth=4, block="glu"), f(4, 5)),
        ("RMSNorm", lambda: me.RMSNorm(6), f(2, 3, 6, scale=2)),
        ("Attention rotary, causal", lambda: me.MultiHeadSelfAttention(8, 2, causal=True, rope=True), f(2, 5, 8)),
        ("Attention rotary, full, bias", lambda: me.MultiHeadSelfAttention(8, 2, causal=False, rope=True, bias=True), f(2, 5, 8)),
        ("TransformerBlock rms + rotary", lambda: me.TransformerBlock(8, 2, norm="rms", rope=True), f(2, 5, 8)),
        ("mitsu_block rms", lambda: me.mitsu_block(6, kind="glu", norm="rms"), f(3, 6)),
        ("MitsuLM tied", lambda: me.MitsuLM(13, context=6, d_model=8, n_layers=2, n_heads=2), r.integers(0, 13, (2, 6))),
        ("MitsuLM untied, short input", lambda: me.MitsuLM(13, context=6, d_model=8, n_layers=2, n_heads=2, tie_embeddings=False), r.integers(0, 13, (2, 5))),
        ("TiedHead", lambda: me.TiedHead(me.Embedding(11, 6)), f(3, 6)),
        ("Embedding", lambda: me.Embedding(11, 6), r.integers(0, 11, (3, 4))),
        ("MitsuGPT glu", lambda: me.MitsuGPT(13, 6, d_model=8, n_layers=2, n_heads=2), r.integers(0, 13, (2, 6))),
        ("MitsuGPT mlp, short input", lambda: me.MitsuGPT(13, 6, d_model=8, n_layers=2, n_heads=2, ffn="mlp"), r.integers(0, 13, (2, 5))),
    ]


@pytest.mark.parametrize("name,build,x", cases(), ids=[c[0] for c in cases()])
def test_backward_matches_finite_differences(name, build, x):
    err, where = me.gradcheck(build(), x, n_checks=40)
    assert err < TOL, f"{name}: worst error {err:.2e} at {where}"


@pytest.mark.parametrize("loss,shape", [(me.CrossEntropy(), (4, 7)), (me.CrossEntropy(label_smoothing=0.1), (2, 3, 7)),
                                         (me.CrossEntropy(ignore_index=2), (6, 7)), (me.MSE(), (4, 3))])
def test_loss_gradients(loss, shape):
    r = np.random.default_rng(2)
    z = r.standard_normal(shape)
    t = r.standard_normal(shape) if isinstance(loss, me.MSE) else r.integers(0, shape[-1], shape[:-1])
    _, g = loss(z, t)
    num = np.zeros_like(z)
    for i in np.ndindex(*shape):
        old = z[i]
        z[i] = old + 1e-6; up = loss(z, t)[0]
        z[i] = old - 1e-6; down = loss(z, t)[0]
        z[i] = old
        num[i] = (up - down) / 2e-6
    assert np.abs(num - g).max() < 1e-8


def test_gradcheck_refuses_float32():
    me.set_dtype(np.float32)
    with pytest.raises(TypeError):
        me.gradcheck(me.Linear(3, 2), np.zeros((1, 3)))
