"""Convolution, pooling, group normalisation and the convolutional network."""
import gzip
import numpy as np
import pytest
import mitsu_engine as me
from mitsu_engine.conv import out_size

R = np.random.default_rng(5)
f = lambda *shape, scale=1.0: R.standard_normal(shape) * scale

CASES = [
    ("Conv2d 3x3 padded", lambda: me.Conv2d(3, 4, 3, padding=1), (2, 3, 6, 6)),
    ("Conv2d stride 2, no bias", lambda: me.Conv2d(3, 4, 3, stride=2, padding=1, bias=False), (2, 3, 7, 7)),
    ("Conv2d 1x1", lambda: me.Conv2d(3, 5, 1), (2, 3, 4, 4)),
    ("Conv2d 5x5 stride 2", lambda: me.Conv2d(2, 3, 5, stride=2), (2, 2, 9, 9)),
    ("Conv2d 2x2 stride 3 padding 2", lambda: me.Conv2d(2, 3, 2, stride=3, padding=2), (2, 2, 8, 8)),
    ("MaxPool2d", lambda: me.MaxPool2d(2), (2, 3, 6, 6)),
    ("MaxPool2d overlapping", lambda: me.MaxPool2d(3, 2), (2, 3, 7, 7)),
    ("AvgPool2d", lambda: me.AvgPool2d(2), (2, 3, 6, 6)),
    ("AvgPool2d overlapping", lambda: me.AvgPool2d(3, 1), (2, 3, 6, 6)),
    ("GlobalAvgPool2d", lambda: me.GlobalAvgPool2d(), (2, 3, 5, 5)),
    ("GroupNorm", lambda: me.GroupNorm(2, 4), (2, 4, 5, 5)),
    ("GroupNorm, one group", lambda: me.GroupNorm(1, 3), (2, 3, 5, 5)),
    ("conv stack", lambda: me.Sequential(me.Conv2d(2, 4, 3, padding=1), me.GroupNorm(2, 4), me.MITSU(), me.MaxPool2d(2), me.Flatten(), me.Linear(36, 3)), (2, 2, 6, 6)),
    ("MitsuConvNet", lambda: me.MitsuConvNet(1, 5, channels=(4, 8), blocks=1, groups=4), (2, 1, 8, 8)),
    ("MitsuConvNet, two blocks, odd width", lambda: me.MitsuConvNet(2, 3, channels=(6, 9), blocks=2, groups=4), (2, 2, 7, 7)),
]


@pytest.mark.parametrize("name,make,shape", CASES, ids=[c[0] for c in CASES])
def test_backward_against_finite_differences(name, make, shape):
    me.set_dtype(np.float64)
    me.manual_seed(0)
    module = make()
    for n, p in module.named_parameters():                   # move gains and biases off their defaults
        if not p.decay:
            p.data += 0.3 * np.random.default_rng(1).standard_normal(p.data.shape)
    err, where = me.gradcheck(module, f(*shape))
    assert err < 1e-5, where


def test_convolution_matches_a_direct_loop():
    me.set_dtype(np.float64)
    conv = me.Conv2d(2, 3, 3, stride=2, padding=1, rng=0)
    conv.bias.data[:] = f(3)
    x = f(2, 2, 5, 6)
    y = conv(x)
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    assert y.shape == (2, 3, out_size(5, 3, 2, 1), out_size(6, 3, 2, 1)) == (2, 3, 3, 3)
    for n in range(2):
        for o in range(3):
            for i in range(3):
                for j in range(3):
                    direct = (xp[n, :, 2 * i:2 * i + 3, 2 * j:2 * j + 3] * conv.weight.data[o]).sum() + conv.bias.data[o]
                    assert abs(y[n, o, i, j] - direct) < 1e-12


def test_pooling_values():
    x = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)
    np.testing.assert_array_equal(me.MaxPool2d(2)(x)[0, 0], [[5, 7], [13, 15]])
    np.testing.assert_array_equal(me.AvgPool2d(2)(x)[0, 0], [[2.5, 4.5], [10.5, 12.5]])
    np.testing.assert_allclose(me.GlobalAvgPool2d()(x), [[7.5, 23.5]])
    pool = me.MaxPool2d(2)
    pool(x)
    g = pool.backward(np.ones((1, 2, 2, 2), dtype=np.float32))
    assert g.sum() == 8 and g[0, 0, 1, 1] == 1 and g[0, 0, 0, 0] == 0          # the gradient goes to each window's winner


def test_group_norm_statistics_and_batch_independence():
    norm = me.GroupNorm(2, 6)
    x = f(4, 6, 5, 5, scale=3).astype(np.float32) + 2
    y = norm(x)
    groups = y.reshape(4, 2, -1)
    assert np.abs(groups.mean(-1)).max() < 1e-5 and np.abs(groups.std(-1) - 1).max() < 1e-3
    np.testing.assert_allclose(norm(x[:1]), y[:1], rtol=1e-6)                   # one image alone gives the same result
    norm.eval()
    np.testing.assert_array_equal(norm(x), y)                                    # no separate evaluation behaviour
    with pytest.raises(ValueError):
        me.GroupNorm(4, 6)


def test_shapes_errors_and_description():
    net = me.MitsuConvNet(3, 7, channels=(8, 16, 32), blocks=2)
    assert net(f(5, 3, 16, 16).astype(np.float32)).shape == (5, 7)
    assert "MitsuConvNet" in repr(net) and "Conv2d" in repr(net) and net.num_parameters() == sum(p.size for p in net.parameters())
    with pytest.raises(ValueError):
        me.Conv2d(3, 4)(f(2, 5, 8, 8))
    with pytest.raises(ValueError):
        me.Conv2d(3, 4)(f(2, 3, 8))


def test_signal_stays_bounded_in_a_deep_conv_net():
    me.manual_seed(0)
    x = f(8, 3, 16, 16).astype(np.float32)
    for blocks in (1, 4, 12):
        out = me.MitsuConvNet(3, 10, channels=(16, 32), blocks=blocks)(x)
        assert np.isfinite(out).all() and 0.02 < out.std() < 5.0


def _bars(n, rng):
    """Images with a horizontal or a vertical bar at a random place: position cannot be
    read from single pixels, so a convolution has to find the orientation."""
    rng = np.random.default_rng(rng)
    x = rng.normal(0, 0.3, (n, 1, 12, 12)).astype(np.float32)
    y = rng.integers(0, 2, n)
    for i in range(n):
        r, c = rng.integers(1, 11), rng.integers(0, 6)
        if y[i]:
            x[i, 0, r, c:c + 6] += 2.0
        else:
            x[i, 0, c:c + 6, r] += 2.0
    return x, y


def test_conv_net_learns(tmp_path):
    me.manual_seed(0)
    X, y = _bars(600, 0)
    Xte, yte = _bars(300, 1)
    net = me.MitsuConvNet(1, 2, channels=(8, 16), blocks=1)
    trainer = me.Trainer(net, me.AdamW(net.parameters(), lr=3e-3), clip_norm=1.0, log=None)
    trainer.fit(me.DataLoader(X, y, 50), epochs=4)
    assert trainer.evaluate(me.DataLoader(Xte, yte, 300, shuffle=False))["metric"] > 0.95
    path = str(tmp_path / "conv.npz")
    me.save_checkpoint(path, net)
    other = me.MitsuConvNet(1, 2, channels=(8, 16), blocks=1)
    me.load_checkpoint(path, other)
    np.testing.assert_array_equal(other(Xte[:4]), net(Xte[:4]))


def test_load_mnist_as_images(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 6), ("t10k", 4)):
        with gzip.open(tmp_path / f"mnist-{split}-images-idx3-ubyte.gz", "wb") as fh:
            fh.write(bytes(16) + rng.integers(0, 256, n * 784, dtype=np.uint8).tobytes())
        with gzip.open(tmp_path / f"mnist-{split}-labels-idx1-ubyte.gz", "wb") as fh:
            fh.write(bytes(8) + rng.integers(0, 10, n, dtype=np.uint8).tobytes())
    Xtr, ytr, Xte, yte = me.load_mnist(str(tmp_path), download=False, flatten=False)
    flat = me.load_mnist(str(tmp_path), download=False)[0]
    assert Xtr.shape == (6, 1, 28, 28) and Xte.shape == (4, 1, 28, 28) and ytr.shape == (6,) and Xtr.dtype == np.float32
    np.testing.assert_array_equal(Xtr.reshape(6, 784), flat)
