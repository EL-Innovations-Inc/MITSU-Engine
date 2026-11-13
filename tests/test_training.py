"""End to end: the engine actually learns, deterministically."""
import numpy as np
import pytest
import mitsu_engine as me


def spiral_data():
    X, y = me.make_spirals(300, classes=3, rng=0)
    return me.train_val_split(X, y, 0.2, rng=1)


def test_plain_mlp_learns_spirals():
    Xtr, ytr, Xva, yva = spiral_data()
    net = me.MLP([2, 64, 64, 3], rng=0)
    tr = me.Trainer(net, me.Adam(net.parameters(), lr=1e-2), log=None)
    tr.fit(me.DataLoader(Xtr, ytr, 64, rng=2), epochs=120)
    assert tr.evaluate(me.DataLoader(Xva, yva, 256, shuffle=False))["metric"] > 0.95


@pytest.mark.parametrize("block", ["mlp", "glu"])
def test_48_layer_network_learns_spirals(block):
    Xtr, ytr, Xva, yva = spiral_data()
    net = me.DeepMitsuNet(2, 3, width=32, depth=48, block=block, rng=0)
    tr = me.Trainer(net, me.AdamW(net.parameters(), lr=1e-3), clip_norm=1.0, scheduler=me.WarmupCosine(1e-3, 30, 20 * 12), log=None)
    tr.fit(me.DataLoader(Xtr, ytr, 64, rng=2), epochs=20)
    assert tr.evaluate(me.DataLoader(Xva, yva, 256, shuffle=False))["metric"] > 0.95


def test_gpt_memorises_a_pattern_and_generates_it():
    pattern = np.array([0, 1, 2, 3, 4, 5, 6, 7] * 40)
    gpt = me.MitsuGPT(8, 16, d_model=32, n_layers=2, n_heads=2, rng=0)
    batcher = me.TextBatcher(pattern, context=16, batch_size=16, rng=1)
    tr = me.Trainer(gpt, me.AdamW(gpt.parameters(), lr=3e-3), clip_norm=1.0, metric=None, log=None)
    tr.fit_steps(batcher, 150)
    x, y = batcher()
    loss, _ = me.CrossEntropy()(gpt(x), y)
    assert loss < 0.1
    out = gpt.generate([0, 1, 2], 13, temperature=0)
    assert out.shape == (1, 16) and out[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7] * 2
    sampled = gpt.generate([0], 20, temperature=1.0, top_k=3, rng=0)
    assert sampled.shape == (1, 21) and sampled.min() >= 0 and sampled.max() < 8


def test_training_is_deterministic():
    def run():
        Xtr, ytr, _, _ = spiral_data()
        net = me.DeepMitsuNet(2, 3, width=16, depth=4, dropout=0.1, rng=5)
        tr = me.Trainer(net, me.AdamW(net.parameters(), lr=1e-3), log=None)
        tr.fit(me.DataLoader(Xtr, ytr, 64, rng=6), epochs=2)
        return [h["train_loss"] for h in tr.history], net.head.weight.data.copy()

    (l1, w1), (l2, w2) = run(), run()
    assert l1 == l2 and np.array_equal(w1, w2)


def test_checkpoint_restores_best_and_early_stopping_stops(tmp_path):
    Xtr, ytr, Xva, yva = spiral_data()
    net = me.MLP([2, 32, 3], rng=0)
    tr = me.Trainer(net, me.Adam(net.parameters(), lr=0.3), log=None)       # far too high: validation gets worse
    hist = tr.fit(me.DataLoader(Xtr, ytr, 64, rng=2), me.DataLoader(Xva, yva, 256, shuffle=False), epochs=40,
                  checkpoint=str(tmp_path / "best.npz"), patience=3)
    assert len(hist) < 40
    best = min(h["val_loss"] for h in hist)
    assert abs(tr.evaluate(me.DataLoader(Xva, yva, 256, shuffle=False))["loss"] - best) < 1e-6


def test_non_finite_loss_raises():
    net = me.Linear(2, 2)
    tr = me.Trainer(net, me.SGD(net.parameters(), lr=0.1), loss_fn=me.MSE(), metric=None, log=None)
    with pytest.raises(FloatingPointError), np.errstate(invalid="ignore"):
        tr.train_step(np.array([[np.inf, 1.0]], dtype=np.float32), np.zeros((1, 2), dtype=np.float32))


def test_gpt_rejects_sequences_longer_than_its_context():
    with pytest.raises(ValueError):
        me.MitsuGPT(8, 4, d_model=8, n_layers=1, n_heads=1)(np.zeros((1, 5), dtype=np.int64))
