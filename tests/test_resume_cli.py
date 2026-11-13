"""Gradient accumulation, exact resume, logging, and the command line end to end."""
import json, math, os
import numpy as np
import pytest
import mitsu_engine as me
from mitsu_engine.cli import main


def test_accumulating_two_half_batches_equals_one_full_batch():
    me.set_dtype(np.float64)
    r = np.random.default_rng(0)
    X, y = r.standard_normal((32, 6)), r.integers(0, 3, 32)

    def run(accumulate):
        net = me.DeepMitsuNet(6, 3, width=8, depth=2, rng=1)
        tr = me.Trainer(net, me.SGD(net.parameters(), lr=0.1), log=None)
        if accumulate == 1:
            tr.train_step(X, y)
        else:
            halves = iter([(X[:16], y[:16]), (X[16:], y[16:])])
            tr.train_step(next_batch=lambda: next(halves), accumulate=2)
        return net.head.weight.data.copy(), net.stem.weight.data.copy()

    (h1, s1), (h2, s2) = run(1), run(2)
    assert np.allclose(h1, h2, atol=1e-12) and np.allclose(s1, s2, atol=1e-12)


def build(seed=0):
    me.manual_seed(seed)
    lm = me.MitsuLM(20, context=8, d_model=16, n_layers=2, n_heads=2, dropout=0.2)
    batcher = me.TextBatcher(np.arange(600) % 20, context=8, batch_size=4)
    trainer = me.Trainer(lm, me.AdamW(lm.parameters(), lr=1e-2), metric=None, clip_norm=1.0,
                         scheduler=me.WarmupCosine(1e-2, 3, 12), log=None)
    return lm, batcher, trainer


def test_resume_continues_bit_for_bit(tmp_path):
    lm_a, batch_a, tr_a = build()
    tr_a.fit_steps(batch_a, 12)

    path = str(tmp_path / "state.npz")
    lm_b, batch_b, tr_b = build()
    tr_b.fit_steps(batch_b, 7, state_path=path)
    lm_c, batch_c, tr_c = build(seed=123)                    # a fresh process would start from other random numbers
    tr_c.load_state(path)
    assert tr_c.step == 7
    tr_c.fit_steps(batch_c, 12)
    for (name, p), q in zip(lm_a.named_parameters(), lm_c.parameters()):
        assert np.array_equal(p.data, q.data), name
    assert tr_c.optimizer.steps == 12


def test_fit_steps_logging_state_and_total_semantics(tmp_path):
    lm, batcher, trainer = build()
    log, state = str(tmp_path / "log.jsonl"), str(tmp_path / "state.npz")
    trainer.fit_steps(batcher, 10, eval_fn=lambda: me.evaluate_lm(lm, batcher.windows(4, 3)), eval_every=5, log_every=2,
                      state_path=state, state_every=5, log_file=log)
    rows = [json.loads(line) for line in open(log)]
    assert [r["step"] for r in rows] == [2, 4, 5, 6, 8, 10]
    assert all("grad_norm_max" in r and r["grad_norm_max"] > 0 for r in rows)
    evals = [r for r in rows if "val_loss" in r]
    assert [r["step"] for r in evals] == [5, 10] and all(math.isclose(r["val_perplexity"], math.exp(r["val_loss"])) for r in evals)
    assert me.read_meta(state)["trainer"]["step"] == 10
    trainer.fit_steps(batcher, 10)                            # already at the total: nothing happens
    assert trainer.step == 10


@pytest.fixture()
def corpus(tmp_path):
    text = ("ROMEO: But soft, what light through yonder window breaks?\nJULIET: It is the east, and I am the sun.\n"
            "NURSE: Peace, I have done. God mark thee to his grace!\n") * 120
    path = tmp_path / "play.txt"
    path.write_text(text)
    data = str(tmp_path / "data")
    main(["prepare", "--text", str(path), "--out", data, "--vocab-size", "300"])
    return data


TINY = ["--d-model", "16", "--layers", "2", "--heads", "2", "--context", "16", "--batch", "4", "--eval-every", "10",
        "--log-every", "5", "--eval-windows", "8", "--dropout", "0.1"]


def weights(run):
    lm = me.MitsuLM.load(os.path.join(run, "model.npz"))
    return [p.data.copy() for p in lm.parameters()]


def test_command_line_end_to_end(corpus, tmp_path, capsys):
    straight, stopped = str(tmp_path / "straight"), str(tmp_path / "stopped")
    main(["train", "--data", corpus, "--out", straight, "--steps", "30"] + TINY)
    assert sorted(os.listdir(straight)) == ["log.jsonl", "model.npz", "run.json", "state.npz", "tokenizer.json"]

    main(["train", "--data", corpus, "--out", stopped, "--steps", "30", "--stop-after", "17"] + TINY)
    assert me.read_meta(os.path.join(stopped, "model.npz"))["step"] == 17
    main(["train", "--data", corpus, "--out", stopped, "--resume"])
    assert me.read_meta(os.path.join(stopped, "model.npz"))["step"] == 30
    assert all(np.array_equal(a, b) for a, b in zip(weights(straight), weights(stopped)))        # interrupted == uninterrupted

    main(["train", "--data", corpus, "--out", stopped, "--resume", "--steps", "40"])               # extend a finished run
    assert me.read_meta(os.path.join(stopped, "model.npz"))["step"] == 40

    capsys.readouterr()
    main(["generate", "--run", straight, "--prompt", "ROMEO:", "--tokens", "12", "--seed", "1"])
    first = capsys.readouterr().out
    main(["generate", "--run", straight, "--prompt", "ROMEO:", "--tokens", "12", "--seed", "1"])
    assert first.startswith("ROMEO:") and capsys.readouterr().out == first
    main(["info", "--run", straight])
    assert "last evaluation: step 30" in capsys.readouterr().out


def test_command_line_refuses_to_overwrite_or_resume_nothing(corpus, tmp_path):
    run = str(tmp_path / "run")
    with pytest.raises(SystemExit):
        main(["train", "--data", corpus, "--out", run, "--resume"])
    main(["train", "--data", corpus, "--out", run, "--steps", "5"] + TINY)
    with pytest.raises(SystemExit):
        main(["train", "--data", corpus, "--out", run, "--steps", "5"] + TINY)
