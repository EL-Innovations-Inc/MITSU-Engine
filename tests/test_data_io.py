"""Data utilities and checkpoints."""
import numpy as np
import pytest
import mitsu_engine as me


def test_dataloader_covers_everything_once():
    X, y = np.arange(103)[:, None].astype(np.float32), np.arange(103)
    seen = np.concatenate([yb for _, yb in me.DataLoader(X, y, 16, rng=0)])
    assert sorted(seen.tolist()) == list(range(103)) and len(me.DataLoader(X, y, 16)) == 7
    assert sum(len(yb) for _, yb in me.DataLoader(X, y, 16, drop_last=True, rng=0)) == 96
    assert np.array_equal(np.concatenate([yb for _, yb in me.DataLoader(X, y, 16, shuffle=False)]), y)
    with pytest.raises(ValueError):
        me.DataLoader(X, y[:-1])


def test_split_is_a_partition():
    X, y = np.arange(100)[:, None], np.arange(100)
    a, b, c, d = me.train_val_split(X, y, 0.2, rng=0)
    assert len(a) == 80 and len(c) == 20 and sorted(np.concatenate([b, d]).tolist()) == list(range(100))


def test_standardizer_does_not_explode_on_constant_features():
    X = np.random.default_rng(0).standard_normal((50, 3))
    X[:, 1] = 0.0                                    # a feature that never varies in training
    sc = me.Standardizer().fit(X)
    Z = sc.transform(X)
    assert np.allclose(Z[:, [0, 2]].mean(0), 0, atol=1e-5) and np.allclose(Z[:, [0, 2]].std(0), 1, atol=1e-4)
    unseen = sc.transform(np.array([[0.0, 255.0, 0.0]]))
    assert abs(unseen[0, 1]) <= 255.0


def test_text_tools():
    text = "hello mitsu, hello world"
    tok = me.CharTokenizer(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text and tok.vocab_size == len(set(text))
    batch = me.TextBatcher(ids, context=5, batch_size=4, rng=0)
    x, y = batch()
    assert x.shape == y.shape == (4, 5) and np.array_equal(x[:, 1:], y[:, :-1])
    windows = list(batch.windows(batch_size=2))
    assert sum(len(a) for a, _ in windows) == (len(ids) - 1) // 5


def test_spirals():
    X, y = me.make_spirals(100, classes=3, rng=0)
    assert X.shape == (300, 2) and np.bincount(y).tolist() == [100, 100, 100] and np.abs(X).max() < 1.5


def test_checkpoint_round_trip(tmp_path):
    x = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    a = me.DeepMitsuNet(6, 3, width=8, depth=3, rng=1)
    oa = me.AdamW(a.parameters(), lr=1e-2)
    for _ in range(3):
        oa.zero_grad(); _, g = me.CrossEntropy()(a(x), np.array([0, 1, 2, 0])); a.backward(g); oa.step()
    path = str(tmp_path / "ckpt.npz")
    me.save_checkpoint(path, a, oa, meta={"note": "three steps", "acc": 0.5})
    b = me.DeepMitsuNet(6, 3, width=8, depth=3, rng=2)
    ob = me.AdamW(b.parameters(), lr=1e-2)
    meta = me.load_checkpoint(path, b, ob)
    assert meta == {"note": "three steps", "acc": 0.5} and ob.steps == 3
    assert np.array_equal(a(x), b(x)) and np.array_equal(oa.v[0], ob.v[0])
    with pytest.raises(KeyError):
        me.load_checkpoint(path, me.DeepMitsuNet(6, 3, width=8, depth=4))
