"""The PyTorch twin against the numpy engine: the same weights must give the same results.

Skipped as a whole when PyTorch is not installed; the core engine does not need it.
"""
import json, os, subprocess, sys
import numpy as np
import pytest

try:
    import torch
    torch.nn.Linear
except Exception:
    pytest.skip("PyTorch is not installed", allow_module_level=True)

import mitsu_engine as me
from mitsu_engine import functional as EF, twin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = {
    "tied": dict(vocab_size=61, context=24, d_model=32, n_layers=2, n_heads=4),
    "untied": dict(vocab_size=61, context=24, d_model=32, n_layers=2, n_heads=4, tie_embeddings=False),
    "one-head": dict(vocab_size=40, context=16, d_model=16, n_layers=1, n_heads=1, ffn_hidden=20),
    "raw-gate": dict(vocab_size=50, context=20, d_model=48, n_layers=3, n_heads=6, gate_mult=1.0, rope_base=500.0),
}


def pair(dtype=np.float64, seed=0, **kw):
    """A numpy MitsuLM and its PyTorch copy, in `dtype`."""
    me.set_dtype(dtype)
    ref = me.MitsuLM(me.LMConfig(**(kw or CONFIGS["tied"])), rng=seed)
    return ref, twin.TorchMitsuLM.from_engine(ref)


def weights_differ_by(ref, net):
    state = twin.engine_state(net)
    return max(float(np.abs(state[n] - p.data).max()) for n, p in ref.named_parameters())


# ---- the core stays numpy-only ---------------------------------------------------------

def test_importing_the_engine_does_not_import_torch():
    code = "import sys, mitsu_engine; import mitsu_engine.cli; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == "False"


# ---- the activation --------------------------------------------------------------------

def test_mitsu_matches_the_engine_formula_and_derivative():
    u = np.concatenate([np.linspace(-40, 40, 2001), [-745.0, -100.0, 0.0, 20.0, 20.5, 100.0, 700.0]])
    t = torch.tensor(u, dtype=torch.float64, requires_grad=True)
    for c in (2.0, 1.3, 3.5):
        y = twin.mitsu(t, c)
        (grad,) = torch.autograd.grad(y.sum(), t)
        np.testing.assert_allclose(y.detach().numpy(), EF.mitsu(u, c), rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(grad.numpy(), EF.mitsu_grad(u, c), rtol=1e-12, atol=1e-15)


def test_softplus_has_no_linear_shortcut():
    u = np.array([19.0, 20.0, 21.0, 25.0, 30.0])
    ours = twin.softplus(torch.tensor(u, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(ours, EF.softplus(u), rtol=1e-15)
    assert np.all(ours > u)                      # torch's built-in softplus returns u itself above 20


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mitsu_is_finite_in_half_precision(dtype):
    """float16 overflows at u >= 257 if s * (s - 1) is formed before the division."""
    u = torch.tensor([-60000.0, -300.0, -1.0, 0.0, 1.0, 256.0, 257.0, 1000.0, 30000.0, 60000.0], dtype=dtype, requires_grad=True)
    y = twin.mitsu(u)
    assert y.dtype == dtype and torch.isfinite(y).all()
    (grad,) = torch.autograd.grad(y.float().sum(), u)
    assert torch.isfinite(grad).all()
    big = u.detach().float() >= 256
    np.testing.assert_allclose(y.detach().float()[big].numpy(), (u.detach().float()[big] - 3.0).numpy(), rtol=1e-2)
    assert y[2].item() < 0 and abs(grad[-1].item() - 1.0) < 1e-2


def test_modules_match_engine_layers():
    me.set_dtype(np.float64)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5, 12))
    norm, tnorm = me.RMSNorm(12), twin.RMSNorm(12).double()
    norm.gain.data[:] = rng.standard_normal(12)
    with torch.no_grad():
        tnorm.gain.copy_(torch.from_numpy(norm.gain.data))
    np.testing.assert_allclose(tnorm(torch.from_numpy(x)).detach().numpy(), norm(x), rtol=1e-13, atol=1e-15)

    glu, tglu = me.MitsuGLU(12, 20, rng=1), twin.MitsuGLU(12, 20).double()
    with torch.no_grad():
        for name in ("w1", "w2", "w3"):
            getattr(tglu, name).weight.copy_(torch.from_numpy(getattr(glu, name).weight.data.T.copy()))
    np.testing.assert_allclose(tglu(torch.from_numpy(x)).detach().numpy(), glu(x), rtol=1e-12, atol=1e-14)

    act = twin.MITSU(c=1.5, gain=3.0)
    np.testing.assert_allclose(act(torch.from_numpy(x)).numpy(), me.MITSU(c=1.5, gain=3.0)(x), rtol=1e-13, atol=1e-15)
    with pytest.raises(TypeError):
        twin.mitsu(torch.arange(3))


# ---- names, order, and moving weights --------------------------------------------------

@pytest.mark.parametrize("name", list(CONFIGS))
def test_parameter_names_and_order_are_the_engines(name):
    ref, net = pair(np.float32, **CONFIGS[name])
    names = [n for n, _ in ref.named_parameters()]
    assert twin.engine_param_names(ref.config) == names
    assert list(twin.engine_state(net)) == names
    assert net.num_parameters() == ref.num_parameters()


@pytest.mark.parametrize("name", ["tied", "untied"])
def test_weights_round_trip_bit_for_bit(name):
    ref, net = pair(np.float32, **CONFIGS[name])
    assert weights_differ_by(ref, net) == 0.0
    back = net.to_engine()
    for (n, a), (_, b) in zip(ref.named_parameters(), back.named_parameters()):
        assert np.array_equal(a.data, b.data), n
    assert (net.head.weight is net.tok.weight) == (name == "tied")


def test_linear_weights_are_transposed_and_others_are_not():
    ref, net = pair(np.float32)
    assert tuple(net.blocks[0].ffn.w1.weight.shape) == ref.blocks[0].ffn.w1.weight.data.shape[::-1]
    assert np.array_equal(net.blocks[0].attn.qkv.weight.detach().numpy(), ref.blocks[0].attn.qkv.weight.data.T)
    assert np.array_equal(net.tok.weight.detach().numpy(), ref.tok.weight.data)
    assert twin.is_linear_weight("head.weight") and not twin.is_linear_weight("tok.weight") and not twin.is_linear_weight("norm.gain")


def test_load_engine_state_rejects_wrong_names_and_shapes():
    ref, net = pair(np.float32)
    state = ref.state_dict()
    with pytest.raises(KeyError):
        twin.load_engine_state(net, {k: v for k, v in state.items() if k != "norm.gain"})
    with pytest.raises(KeyError):
        twin.load_engine_state(net, {**state, "extra.weight": np.zeros(3)})
    with pytest.raises(ValueError):
        twin.load_engine_state(net, {**state, "norm.gain": np.zeros(7)})


def test_same_seed_gives_the_same_initial_weights():
    cfg = me.LMConfig(**CONFIGS["tied"])
    net = twin.TorchMitsuLM(cfg, rng=5)
    assert weights_differ_by(me.MitsuLM(cfg, rng=5), net) == 0.0
    me.manual_seed(9)
    a = twin.TorchMitsuLM(cfg)                    # rng=None: the engine's global generator
    me.manual_seed(9)
    assert weights_differ_by(me.MitsuLM(cfg), a) == 0.0


# ---- the central test: same weights, same results --------------------------------------

@pytest.mark.parametrize("name", list(CONFIGS))
def test_outputs_and_gradients_match_in_float64(name):
    ref, net = pair(np.float64, **CONFIGS[name])
    r = twin.compare(ref, net, batch=3, tokens=ref.config.context)
    assert r["logits"] < 1e-11 and r["loss"] < 1e-12 and r["cached"] < 1e-11, r
    assert r["gradient"] < 1e-10, r               # hand-written backward against autograd
    assert r["generated_equal"]


@pytest.mark.parametrize("name", ["tied", "untied"])
def test_outputs_and_gradients_match_in_float32(name):
    ref, net = pair(np.float32, **CONFIGS[name])
    r = twin.compare(ref, net, batch=3, tokens=ref.config.context)
    limit = twin.parity.TOLERANCE["float32"]
    assert all(r[k] < limit[k] for k in limit) and r["generated_equal"], r


def test_compare_detects_a_changed_weight():
    ref, net = pair(np.float64)
    with torch.no_grad():
        net.blocks[1].ffn.w2.weight[0, 0] += 1e-3
    r = twin.compare(ref, net)
    assert r["logits"] > 1e-6


def test_check_reports_pass_on_cpu():
    ok, rows = twin.check(amp=("bf16", "fp16"))
    assert ok, rows
    assert [r["precision"] for r in rows] == ["float64", "float32", "autocast bf16", "autocast fp16"]
    assert "pass" in twin.format_report(rows)


def test_forward_rejects_sequences_longer_than_the_context():
    _, net = pair(np.float32)
    with pytest.raises(ValueError):
        net(torch.zeros((1, net.config.context + 1), dtype=torch.long))


def test_dropout_is_active_in_training_and_off_in_evaluation():
    me.set_dtype(np.float64)
    ref = me.MitsuLM(me.LMConfig(**{**CONFIGS["tied"], "dropout": 0.3}), rng=0)
    net = twin.TorchMitsuLM.from_engine(ref)
    idx = np.random.default_rng(0).integers(0, 61, (2, 24))
    t = torch.from_numpy(idx)
    net.train()
    assert not torch.equal(net(t), net(t))
    net.eval(), ref.eval()
    np.testing.assert_allclose(net(t).detach().numpy(), ref(idx), atol=1e-11)


def test_dropout_has_the_same_distribution_in_both_engines():
    """The masks differ, so single losses differ; over many draws the mean and the spread agree."""
    ref = me.MitsuLM(me.LMConfig(**{**CONFIGS["tied"], "dropout": 0.3}), rng=0)
    net = twin.TorchMitsuLM.from_engine(ref)
    rng = np.random.default_rng(0)
    idx, tgt = rng.integers(0, 61, (4, 24)), rng.integers(0, 61, (4, 24))
    t_idx, t_tgt = torch.from_numpy(idx), torch.from_numpy(tgt)
    loss_fn, n = me.CrossEntropy(), 600
    torch.manual_seed(0)
    ref.train(), net.train()
    with me.no_grad(), torch.no_grad():
        a = np.array([loss_fn(ref(idx), tgt)[0] for _ in range(n)])
        b = np.array([net.loss(t_idx, t_tgt).item() for _ in range(n)])
    standard_error = np.sqrt(a.var() / n + b.var() / n)
    assert abs(a.mean() - b.mean()) < 4 * standard_error
    assert 0.85 < a.std() / b.std() < 1.15
    ref.eval()
    with me.no_grad():
        assert abs(loss_fn(ref(idx), tgt)[0] - a.mean()) > 10 * standard_error      # dropout really was on


# ---- generation ------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [dict(temperature=0.0), dict(temperature=0.9), dict(temperature=1.2, top_k=7),
                                    dict(temperature=0.8, top_p=0.9), dict(temperature=1.0, top_k=20, top_p=0.7)])
def test_generation_returns_the_same_tokens(kwargs):
    ref, net = pair(np.float64)
    prompt = np.random.default_rng(1).integers(0, 61, (2, 5))
    a = ref.generate(prompt, 60, rng=11, **kwargs)                 # 60 > context: the window restarts
    b = net.generate(prompt, 60, rng=11, **kwargs)
    assert a.dtype == b.dtype == np.int64 and np.array_equal(a, b)
    c = net.generate(prompt, 60, rng=11, use_cache=False, **kwargs)
    assert np.array_equal(b, c)


def test_generation_accepts_tensors_long_prompts_and_a_stop_token():
    ref, net = pair(np.float64)
    prompt = np.random.default_rng(2).integers(0, 61, 40)          # longer than the context of 24
    a, b = ref.generate(prompt, 10, temperature=0.0), net.generate(torch.from_numpy(prompt), 10, temperature=0.0)
    assert np.array_equal(a, b)
    stop = int(a[0, 41])
    assert net.generate(prompt, 10, temperature=0.0, stop_token=stop).shape[1] == 42
    assert net.training                                             # the mode is restored


# ---- checkpoints -----------------------------------------------------------------------

@pytest.mark.parametrize("name", ["tied", "untied"])
def test_checkpoints_are_interchangeable(tmp_path, name):
    ref, net = pair(np.float32, **CONFIGS[name])
    idx = np.random.default_rng(0).integers(0, 61, (2, 24))
    a, b = str(tmp_path / "from_torch.npz"), str(tmp_path / "from_numpy.npz")
    net.save(a, note="written by PyTorch")
    ref.save(b)
    for path in (a, b):                                             # either file, either engine
        loaded_ref, loaded_net = me.MitsuLM.load(path), twin.TorchMitsuLM.load(path)
        assert all(np.array_equal(p.data, q.data) for (_, p), (_, q) in zip(ref.named_parameters(), loaded_ref.named_parameters()))
        assert weights_differ_by(ref, loaded_net) == 0.0
        assert loaded_net.config == ref.config
    assert me.read_meta(a)["note"] == "written by PyTorch"
    with np.load(a) as za, np.load(b) as zb:
        assert sorted(za.files) == sorted(zb.files)                 # same keys, including no head.weight when tied
        assert all(np.array_equal(za[k], zb[k]) for k in za.files if k != "meta")
    np.testing.assert_allclose(twin.TorchMitsuLM.load(b)(torch.from_numpy(idx)).detach().numpy(), me.MitsuLM.load(a)(idx), atol=5e-5)


def test_load_can_choose_precision():
    ref, net = pair(np.float64)
    assert net.tok.weight.dtype == torch.float64                    # from_engine keeps the engine's precision
    assert twin.TorchMitsuLM.from_engine(ref, dtype=torch.float32).tok.weight.dtype == torch.float32


# ---- optimiser and training ------------------------------------------------------------

def test_clip_grad_norm_follows_the_engine_rule():
    ref, net = pair(np.float64)
    rng = np.random.default_rng(0)
    named = dict(net.named_parameters())
    for n, p in ref.named_parameters():
        p.grad[...] = rng.standard_normal(p.data.shape)
        named[n].grad = torch.from_numpy(twin.to_torch_array(n, p.grad))
    assert abs(twin.grad_norm(net.parameters()) - me.grad_norm(ref.parameters())) < 1e-10
    for limit in (1e9, 0.7):                                        # not clipped, then clipped
        a, b = me.clip_grad_norm(ref.parameters(), limit), twin.clip_grad_norm(net.parameters(), limit)
        assert abs(a - b) < 1e-10
        for n, p in ref.named_parameters():
            np.testing.assert_allclose(twin.to_engine_array(n, named[n].grad), p.grad, rtol=1e-13)
    assert abs(twin.grad_norm(net.parameters()) - 0.7) < 1e-9


def test_weight_decay_applies_to_weight_matrices_only():
    _, net = pair(np.float64)
    opt = twin.EngineAdamW(net, lr=0.1, weight_decay=0.5)
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    for p in net.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()                                                       # zero gradient: only decay moves anything
    for n, p in net.named_parameters():
        if twin.is_linear_weight(n):
            np.testing.assert_allclose(p.detach().numpy(), (before[n] * (1 - 0.1 * 0.5)).numpy(), rtol=1e-14)
        else:
            assert torch.equal(p, before[n]), n


def _data(n=4000, vocab=61, seed=0):
    return np.random.default_rng(seed).integers(0, vocab, n)


def _numpy_trainer(cfg, seed=3, steps=30, lr=2e-3, clip=0.5):
    me.manual_seed(seed)
    model = me.MitsuLM(cfg)
    batcher = me.TextBatcher(_data(vocab=cfg.vocab_size), cfg.context, 4)
    trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=lr, weight_decay=0.1), metric=None, clip_norm=clip,
                         scheduler=me.WarmupCosine(lr, 5, steps, min_lr=lr / 10), log=None)
    return model, trainer, batcher


def _torch_trainer(cfg, seed=3, steps=30, lr=2e-3, clip=0.5, amp="off", dtype=None):
    me.manual_seed(seed)
    torch.manual_seed(seed)
    model = twin.TorchMitsuLM(cfg)
    if dtype is not None:
        model.to(dtype)
    batcher = me.TextBatcher(_data(vocab=cfg.vocab_size), cfg.context, 4)
    trainer = twin.TorchTrainer(model, twin.EngineAdamW(model, lr=lr, weight_decay=0.1), clip_norm=clip, amp=amp,
                                scheduler=me.WarmupCosine(lr, 5, steps, min_lr=lr / 10), log=None)
    return model, trainer, batcher


@pytest.mark.parametrize("accumulate", [1, 2])
def test_training_follows_the_same_trajectory(accumulate):
    """Same seed, same batches, same optimiser rule: after 30 steps with warm-up, cosine
    decay, clipping and weight decay the two engines hold the same weights."""
    me.set_dtype(np.float64)
    cfg = me.LMConfig(**CONFIGS["tied"])
    ref, rt, rb = _numpy_trainer(cfg)
    ref_losses = [rt.train_step(next_batch=rb, accumulate=accumulate) for _ in range(30)]
    net, tt, tb = _torch_trainer(cfg, dtype=torch.float64)
    losses = [tt.train_step(next_batch=tb, accumulate=accumulate) for _ in range(30)]
    np.testing.assert_allclose([l[0] for l in losses], [l[0] for l in ref_losses], rtol=1e-10)
    np.testing.assert_allclose([l[2] for l in losses], [l[2] for l in ref_losses], rtol=1e-9)      # gradient norms
    assert max(l[2] for l in losses) > 0.5                          # clipping was active
    assert weights_differ_by(ref, net) < 1e-10
    assert tt.optimizer.lr == rt.optimizer.lr and tt.step == rt.step == 30
    state, ref_state = tt.optimizer.engine_state(), rt.optimizer.state_dict()
    assert list(state) == list(ref_state)
    assert max(float(np.abs(state[k] - ref_state[k]).max()) for k in state) < 1e-12


def test_training_trajectory_stays_close_in_float32():
    cfg = me.LMConfig(**CONFIGS["tied"])
    ref, rt, rb = _numpy_trainer(cfg)
    net, tt, tb = _torch_trainer(cfg)
    for _ in range(30):
        a, b = rt.train_step(next_batch=rb), tt.train_step(next_batch=tb)
        assert abs(a[0] - b[0]) < 1e-3
    assert weights_differ_by(ref, net) < 1e-3


@pytest.mark.parametrize("first", ["numpy", "torch"])
def test_a_run_can_change_engine_midway(tmp_path, first):
    """12 steps in one engine equal 6 steps in one, a state file, and 6 steps in the other."""
    me.set_dtype(np.float64)
    cfg = me.LMConfig(**CONFIGS["tied"])
    path = str(tmp_path / "state.npz")
    whole, wt, wb = _numpy_trainer(cfg, steps=12)
    wt.fit_steps(wb, 12)

    make = {"numpy": _numpy_trainer, "torch": lambda c, **k: _torch_trainer(c, dtype=torch.float64, **k)}
    second = "torch" if first == "numpy" else "numpy"
    _, t1, b1 = make[first](cfg, steps=12)
    t1.fit_steps(b1, 6, state_path=path)
    m2, t2, b2 = make[second](cfg, seed=99, steps=12)               # a different seed: everything must come from the file
    t2.load_state(path)
    assert t2.step == 6
    t2.fit_steps(b2, 12)
    diff = weights_differ_by(whole, m2) if second == "torch" else max(
        float(np.abs(p.data - q.data).max()) for (_, p), (_, q) in zip(whole.named_parameters(), m2.named_parameters()))
    assert diff < 1e-10


def test_torch_resume_is_exact_with_dropout(tmp_path):
    cfg = me.LMConfig(**{**CONFIGS["tied"], "dropout": 0.2})
    path = str(tmp_path / "state.npz")
    whole, wt, wb = _torch_trainer(cfg, steps=10)
    wt.fit_steps(wb, 10)
    _, t1, b1 = _torch_trainer(cfg, steps=10)
    t1.fit_steps(b1, 4, state_path=path)
    resumed, t2, b2 = _torch_trainer(cfg, seed=77, steps=10)
    t2.load_state(path)
    t2.fit_steps(b2, 10)
    assert all(torch.equal(p, q) for p, q in zip(whole.parameters(), resumed.parameters()))
    assert t2.step == 10
    assert me.read_meta(path)["trainer"]["backend"] == "torch"


def _memorise(amp, steps=150):
    """Train on one repeated 16-token sequence until the model can continue it."""
    cfg = me.LMConfig(vocab_size=32, context=16, d_model=32, n_layers=2, n_heads=4)
    seq = np.random.default_rng(5).integers(0, 32, 16)
    data = np.tile(seq, 60)
    me.manual_seed(0)
    model = twin.TorchMitsuLM(cfg)
    batcher = me.TextBatcher(data, 16, 8)
    trainer = twin.TorchTrainer(model, twin.EngineAdamW(model, lr=3e-3), clip_norm=1.0, amp=amp, log=None)
    first = trainer.train_step(next_batch=batcher)[0]
    for _ in range(steps - 1):
        last = trainer.train_step(next_batch=batcher)[0]
    return model, seq, first, last, trainer


@pytest.mark.parametrize("amp", ["off", "bf16", "fp16"])
def test_the_twin_learns_under_every_precision_mode(amp):
    model, seq, first, last, trainer = _memorise(amp)
    assert np.isfinite(last) and last < 0.25 * first
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert trainer.scaler.is_enabled() == (amp == "fp16")
    if amp == "off":
        out = model.generate(seq[:8], 8, temperature=0.0)[0]
        assert np.array_equal(out, seq)
        # and the numpy engine, given the trained weights, continues the sequence the same way
        assert np.array_equal(model.to_engine().generate(seq[:8], 8, temperature=0.0)[0], seq)


def test_evaluate_lm_matches_the_engine():
    ref, net = pair(np.float64)
    batcher = me.TextBatcher(_data(600), 24, 4)
    a = me.evaluate_lm(ref, batcher.windows(4))
    b = twin.evaluate_lm(net, batcher.windows(4))
    assert abs(a["loss"] - b["loss"]) < 1e-11 and abs(a["perplexity"] - b["perplexity"]) < 1e-8
    assert net.training


def test_trainer_rejects_unknown_modes():
    _, net = pair(np.float32)
    with pytest.raises(ValueError):
        twin.TorchTrainer(net, twin.EngineAdamW(net), amp="fp8")
    with pytest.raises(NotImplementedError):
        twin.TorchTrainer(net, twin.EngineAdamW(net), log=None).fit([])
    with pytest.raises(TypeError):
        twin.TorchTrainer(net, torch.optim.AdamW(net.parameters()))


# ---- the main repository's MitsuGLU drops in --------------------------------------------

def test_the_main_repository_mitsuglu_drops_in():
    mitsu_torch = pytest.importorskip("mitsu_torch", reason="the main MITSU repository's package is not installed")
    if not hasattr(mitsu_torch, "MitsuGLU"):
        pytest.skip("mitsu_torch has no MitsuGLU")
    ref, net = pair(np.float64)
    other = twin.TorchMitsuLM.from_engine(ref, glu=mitsu_torch.MitsuGLU)
    assert isinstance(other.blocks[0].ffn, mitsu_torch.MitsuGLU)
    r = twin.compare(ref, other)
    assert r["logits"] < 1e-8 and r["gradient"] < 1e-7 and r["generated_equal"], r


# ---- command line ----------------------------------------------------------------------

def _cli(*args, cwd):
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run([sys.executable, "-m", "mitsu_engine", *args], capture_output=True, text=True, cwd=cwd, env=env)


def test_command_line_parity_and_backends(tmp_path):
    out = _cli("parity", "--device", "cpu", cwd=tmp_path)
    assert out.returncode == 0 and "PASS" in out.stdout, out.stdout + out.stderr

    text = "the quick brown fox jumps over the lazy dog. " * 400
    (tmp_path / "t.txt").write_text(text)
    assert _cli("prepare", "--text", "t.txt", "--out", "data", "--vocab-size", "280", cwd=tmp_path).returncode == 0
    common = ["--data", "data", "--d-model", "32", "--layers", "2", "--heads", "4", "--context", "32", "--batch", "4",
              "--eval-every", "10", "--log-every", "5", "--eval-windows", "4", "--steps", "20"]
    a = _cli("train", *common, "--out", "np", cwd=tmp_path)
    b = _cli("train", *common, "--out", "pt", "--backend", "torch", "--device", "cpu", cwd=tmp_path)
    assert a.returncode == 0 and b.returncode == 0, a.stderr + b.stderr
    rows = lambda run: [json.loads(line) for line in open(tmp_path / run / "log.jsonl")]
    for x, y in zip(rows("np"), rows("pt")):                        # the two logs describe the same run
        assert x["step"] == y["step"] and abs(x["train_loss"] - y["train_loss"]) < 1e-3
        assert abs(x.get("val_loss", 0) - y.get("val_loss", 0)) < 1e-3

    # a run started in numpy, stopped, and finished in PyTorch ends where the numpy run ended
    c = _cli("train", *common, "--out", "mixed", "--stop-after", "10", cwd=tmp_path)
    d = _cli("train", "--data", "data", "--out", "mixed", "--resume", "--backend", "torch", "--device", "cpu", cwd=tmp_path)
    assert c.returncode == 0 and d.returncode == 0 and "resumed at step 10" in d.stdout, c.stderr + d.stderr
    with np.load(tmp_path / "np" / "model.npz") as za, np.load(tmp_path / "mixed" / "model.npz") as zb:
        assert max(float(np.abs(za[k] - zb[k]).max()) for k in za.files if k.startswith("param/")) < 1e-3

    g1 = _cli("generate", "--run", "pt", "--prompt", "the quick", "--tokens", "12", cwd=tmp_path)
    g2 = _cli("generate", "--run", "pt", "--prompt", "the quick", "--tokens", "12", "--backend", "torch", "--device", "cpu", cwd=tmp_path)
    assert g1.returncode == 0 and g2.returncode == 0 and g1.stdout == g2.stdout and g1.stdout.strip()
    assert _cli("info", "--run", "pt", cwd=tmp_path).returncode == 0
