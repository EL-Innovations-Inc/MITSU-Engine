"""Every engine model has a faithful PyTorch copy: twin.to_torch, the generic trainer, the
device check, and device discipline (nothing left on the wrong device).

Skipped as a whole when PyTorch is not installed.
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
from mitsu_engine import autograd as ag, twin
from mitsu_engine.twin import gpucheck

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZOO = [label for label, _, _ in twin.model_zoo()]


def zoo(label, dtype=np.float64):
    return next((m, x) for name, m, x in twin.model_zoo(dtype) if name == label)


def gap(model, net):
    state = twin.engine_state(net)
    return max(float(np.abs(state[n] - p.data).max()) for n, p in model.named_parameters())


# ---- same structure, same names, same numbers --------------------------------------------

@pytest.mark.parametrize("label", ZOO)
def test_mirror_has_the_engine_parameters(label):
    model, _ = zoo(label, np.float32)
    net = twin.to_torch(model)
    names = [n for n, _ in model.named_parameters()]
    assert [entry[0] for entry in net.engine_layout] == names
    assert sorted(n for n, _ in net.named_parameters()) == sorted(names)
    assert sum(p.numel() for p in net.parameters()) == model.num_parameters()
    assert [entry[3] for entry in net.engine_layout] == [p.decay for _, p in model.named_parameters()]
    assert gap(model, net) == 0.0                                  # copied bit for bit
    for p in model.parameters():
        p.data[...] = 0
    twin.to_engine(net, model)
    assert gap(model, net) == 0.0 and any(np.abs(p.data).max() > 0 for p in model.parameters())


@pytest.mark.parametrize("label", ZOO)
def test_outputs_and_gradients_match_in_float64(label):
    model, x = zoo(label)
    net = twin.to_torch(model)
    assert next(net.parameters()).dtype == torch.float64
    r = twin.compare_models(model, net, x)
    assert r["output"] < 1e-11 and r["gradient"] < 1e-10, r
    assert r.get("input_gradient", 0.0) < 1e-10, r


def test_every_network_passes_in_both_precisions():
    ok, rows = twin.check_models("cpu")
    assert ok and len(rows) == 2 * len(ZOO), [r for r in rows if not r["ok"]]


@pytest.mark.parametrize("loss", ["cross_entropy", "mse"])
def test_compare_models_with_targets(loss):
    model, x = zoo("MitsuConvNet")
    net = twin.to_torch(model)
    y = np.array([0, 3, 1]) if loss == "cross_entropy" else np.random.default_rng(0).standard_normal((3, 5))
    r = twin.compare_models(model, net, x, y, loss=loss)
    assert r["loss"] < 1e-12 and r["gradient"] < 1e-10 and r["input_gradient"] < 1e-10, r


def test_compare_models_detects_a_changed_weight():
    model, x = zoo("MLP")
    net = twin.to_torch(model)
    with torch.no_grad():
        net.layers[0].weight[0, 0] += 1e-3
    assert twin.compare_models(model, net, x)["output"] > 1e-6


def test_conv_layers_match_pytorch_functions():
    """The numpy kernels against torch.nn.functional directly, including strides and padding."""
    me.set_dtype(np.float64)
    r = np.random.default_rng(0)
    x = r.standard_normal((2, 3, 9, 10))
    t = torch.from_numpy(x)
    for kernel, stride, padding in [(3, 1, 1), (3, 2, 1), (5, 2, 0), (1, 1, 0), (2, 3, 2)]:
        conv = me.Conv2d(3, 4, kernel, stride, padding, rng=1)
        conv.bias.data[:] = r.standard_normal(4)
        want = torch.nn.functional.conv2d(t, torch.from_numpy(conv.weight.data), torch.from_numpy(conv.bias.data), stride, padding)
        np.testing.assert_allclose(conv(x), want.numpy(), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(me.MaxPool2d(3, 2)(x), torch.nn.functional.max_pool2d(t, 3, 2).numpy())
    np.testing.assert_allclose(me.AvgPool2d(2)(x), torch.nn.functional.avg_pool2d(t, 2).numpy(), rtol=1e-13)
    norm = me.GroupNorm(3, 3)
    np.testing.assert_allclose(norm(x), torch.nn.functional.group_norm(t, 3, eps=norm.eps).numpy(), rtol=1e-11, atol=1e-13)


def test_tied_table_is_shared_in_the_mirror():
    me.set_dtype(np.float64)
    tok = me.Embedding(11, 6, rng=0)
    model = me.Sequential(tok, me.LayerNorm(6), me.TiedHead(tok))
    net = twin.to_torch(model)
    assert net.layers[2].weight is net.layers[0].weight and [e[0] for e in net.engine_layout] == [n for n, _ in model.named_parameters()]
    assert twin.compare_models(model, net, np.array([[1, 4, 4, 9]]))["gradient"] < 1e-10


def test_language_model_goes_through_to_torch_too():
    model = me.MitsuLM(me.LMConfig(vocab_size=30, context=8, d_model=16, n_layers=1, n_heads=2), rng=0)
    net = twin.to_torch(model)
    assert isinstance(net, twin.TorchMitsuLM) and gap(model, net) == 0.0


# ---- layers of your own ------------------------------------------------------------------

class Scale(me.AutoModule):
    def __init__(self, dim):
        self.factor = me.Parameter(np.full(dim, 1.5), decay=False)

    def compute(self, x):
        return ag.mitsu(x) * self.factor


class TorchScale(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return twin.mitsu(x) * self.factor


def test_unknown_layers_need_registering():
    me.set_dtype(np.float64)
    model = me.Sequential(me.Linear(5, 6, rng=0), Scale(6), me.Linear(6, 2, rng=1))
    with pytest.raises(TypeError, match="twin.register"):
        twin.to_torch(model)
    twin.register(Scale, lambda m, convert: TorchScale(m.factor.data.size))
    try:
        net = twin.to_torch(model)
        r = twin.compare_models(model, net, np.random.default_rng(0).standard_normal((4, 5)))
        assert r["output"] < 1e-12 and r["gradient"] < 1e-10       # an autodiff layer against its PyTorch form
    finally:
        twin.mirror._REGISTRY.pop(Scale)


def test_layout_is_required():
    with pytest.raises(TypeError):
        twin.EngineAdamW(torch.nn.Linear(3, 3))


# ---- training ----------------------------------------------------------------------------

def _images(n, seed):
    return gpucheck._bars(n, seed)


@pytest.mark.parametrize("kind", ["conv", "mlp"])
def test_training_follows_the_numpy_run(kind):
    """Same seed, same shuffled batches, same optimiser rule: the same weights after two epochs."""
    me.set_dtype(np.float64)
    X, y = _images(96, 0)
    if kind == "mlp":
        X = X.reshape(96, -1)
    build = (lambda: me.MitsuConvNet(1, 2, channels=(8, 16), blocks=1)) if kind == "conv" else (lambda: me.MLP([144, 24, 24, 2], init=3.0))
    schedule = lambda: me.WarmupCosine(2e-3, 4, 24)

    me.manual_seed(1)
    ref = build()
    rt = me.Trainer(ref, me.AdamW(ref.parameters(), lr=2e-3, weight_decay=0.05), scheduler=schedule(), clip_norm=1.0, log=None)
    ref_history = rt.fit(me.DataLoader(X, y, 8), me.DataLoader(X, y, 48, shuffle=False), epochs=2)

    me.manual_seed(1)
    net = twin.to_torch(build())
    tt = twin.TorchTrainer(net, twin.EngineAdamW(net, lr=2e-3, weight_decay=0.05), metric="accuracy", scheduler=schedule(), clip_norm=1.0, log=None)
    history = tt.fit(me.DataLoader(X, y, 8), me.DataLoader(X, y, 48, shuffle=False), epochs=2)

    assert gap(ref, net) < 1e-10
    for a, b in zip(ref_history, history):                          # the same log, row for row
        for key in ("train_loss", "train_metric", "val_loss", "val_metric"):
            assert abs(a[key] - b[key]) < 1e-9, key


def test_fit_checkpoint_early_stopping_and_mse(tmp_path):
    X, y = _images(200, 0)
    Xva, yva = _images(100, 1)
    model = me.MitsuConvNet(1, 2, channels=(8, 16), blocks=1, rng=0)
    net = twin.to_torch(model)
    path = str(tmp_path / "best.npz")
    trainer = twin.TorchTrainer(net, twin.EngineAdamW(net, lr=3e-3), metric="accuracy", clip_norm=1.0, log=None)
    history = trainer.fit(me.DataLoader(X, y, 50, rng=0), me.DataLoader(Xva, yva, 100, shuffle=False), epochs=5, checkpoint=path, patience=2)
    assert 1 <= len(history) <= 5 and trainer.evaluate(me.DataLoader(Xva, yva, 100, shuffle=False))["metric"] > 0.9
    me.load_checkpoint(path, model)                                  # the best model, read by numpy
    assert gap(model, net) == 0.0 and "val_loss" in me.read_meta(path)

    reg = twin.to_torch(me.MLP([3, 16, 1], init=3.0, rng=0))
    rt = twin.TorchTrainer(reg, twin.EngineAdamW(reg, lr=1e-2), loss="mse", log=None)
    x = np.random.default_rng(0).standard_normal((64, 3)).astype(np.float32)
    target = x.sum(1, keepdims=True)
    first = rt.train_step(x, target)[0]
    for _ in range(150):
        last = rt.train_step(x, target)[0]
    assert last < 0.1 * first
    custom = twin.TorchTrainer(reg, twin.EngineAdamW(reg), loss=lambda out, t: (out - t).abs().mean(), metric=lambda out, t: 1.0, log=None)
    assert custom.train_step(x, target)[1] == 1.0


def test_state_and_checkpoints_cross_engines_for_any_model(tmp_path):
    me.set_dtype(np.float64)
    X, y = _images(64, 0)
    path, ckpt = str(tmp_path / "state.npz"), str(tmp_path / "model.npz")
    build = lambda: me.MitsuConvNet(1, 2, channels=(8, 16), blocks=1, rng=3)
    step = lambda trainer, i: trainer.train_step(X[8 * (i % 8):8 * (i % 8) + 8], y[8 * (i % 8):8 * (i % 8) + 8])

    whole = build()
    wt = me.Trainer(whole, me.AdamW(whole.parameters(), lr=2e-3), clip_norm=1.0, log=None)
    for i in range(10):
        step(wt, i)

    net = twin.to_torch(build())
    tt = twin.TorchTrainer(net, twin.EngineAdamW(net, lr=2e-3), clip_norm=1.0, log=None)
    for i in range(5):
        step(tt, i)
    tt.save_state(path)
    resumed = me.MitsuConvNet(1, 2, channels=(8, 16), blocks=1, rng=9)
    rt = me.Trainer(resumed, me.AdamW(resumed.parameters(), lr=2e-3), clip_norm=1.0, log=None)
    rt.load_state(path)                                              # PyTorch state, numpy trainer
    for i in range(5, 10):
        step(rt, i)
    assert max(float(np.abs(p.data - q.data).max()) for (_, p), (_, q) in zip(whole.named_parameters(), resumed.named_parameters())) < 1e-10

    twin.save_checkpoint(ckpt, net, tt.optimizer, meta={"note": "from PyTorch"})
    other = twin.to_torch(build())
    opt = twin.EngineAdamW(other, lr=2e-3)
    assert twin.load_checkpoint(ckpt, other, opt)["note"] == "from PyTorch" and opt.steps == 5
    assert all(torch.equal(p, q) for p, q in zip(net.parameters(), other.parameters()))


# ---- device discipline, without a GPU ------------------------------------------------------

@pytest.mark.parametrize("label", ZOO + ["MitsuLM"])
def test_nothing_is_left_on_the_wrong_device(label):
    """PyTorch's "meta" device holds shapes and no data, and refuses to mix with CPU tensors.
    Running the forward pass, the backward pass and an optimiser step there fails if any
    tensor (a mask, a position table, a constant) was created on the CPU by mistake:
    the same mistake that would fail on a GPU."""
    if label == "MitsuLM":
        model, x = me.MitsuLM(me.LMConfig(vocab_size=30, context=8, d_model=16, n_layers=2, n_heads=2), rng=0), np.zeros((2, 8), dtype=np.int64)
    else:
        model, x = zoo(label, np.float32)
    net = twin.to_torch(model, device="meta")
    assert all(p.device.type == "meta" for p in net.parameters())
    dtype = torch.float32 if np.issubdtype(x.dtype, np.floating) else torch.long
    out = net(torch.zeros(x.shape, dtype=dtype, device="meta"))
    assert out.device.type == "meta"
    out.sum().backward()
    optimizer = twin.EngineAdamW(net, lr=1e-3, weight_decay=0.1)
    optimizer.step()
    assert all(p.grad is not None and p.grad.device.type == "meta" for p in net.parameters())
    if label == "MitsuLM":                                           # the generation path keeps its cache on the device
        logits, caches = net.step(torch.zeros((2, 3), dtype=torch.long, device="meta"))
        logits, caches = net.step(torch.zeros((2, 1), dtype=torch.long, device="meta"), caches)
        assert logits.device.type == "meta" and caches[0][0].device.type == "meta" and caches[0][0].shape[2] == 4


def test_meta_device_would_catch_a_stray_cpu_tensor():
    with pytest.raises(RuntimeError):
        torch.zeros(3, device="meta") * torch.ones(3)


# ---- the device check ----------------------------------------------------------------------

def test_gpu_check_passes_on_the_cpu(tmp_path):
    lines = []
    out = str(tmp_path / "report.json")
    before = me.get_rng_state()
    ok, report = gpucheck.run("cpu", amp=("bf16",), speed=False, out=out, log=lines.append)
    assert ok, [c for c in report["checks"] if not c["ok"]]
    assert len(report["checks"]) == 9 and "PASS" in lines[-2]
    saved = json.load(open(out))
    assert saved["ok"] and saved["environment"]["torch"] == torch.__version__ and saved["environment"]["device"] == "cpu"
    assert me.get_rng_state() == before and me.get_dtype() == np.float32      # the engine's settings are left as found


def test_gpu_check_isolates_a_failing_check(monkeypatch):
    def broken(device):
        raise RuntimeError("simulated device failure")
    monkeypatch.setattr(gpucheck, "check_every_network", broken)
    monkeypatch.setattr(gpucheck, "check_learning", lambda device, amp: (True, "skipped in this test", {}))
    ok, report = gpucheck.run("cpu", amp=(), speed=False, log=None)
    failed = [c for c in report["checks"] if not c["ok"]]
    assert not ok and len(failed) == 1 and "simulated device failure" in failed[0]["detail"] and "Traceback" in failed[0]["traceback"]
    assert sum(c["ok"] for c in report["checks"]) == len(report["checks"]) - 1   # the others still ran


def test_command_line_gpu_check(tmp_path):
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-m", "mitsu_engine", "gpu-check", "--device", "cpu", "--amp", "bf16", "--no-speed", "--out", "r.json"],
                         capture_output=True, text=True, cwd=tmp_path, env=env)
    assert out.returncode == 0 and "PASS" in out.stdout, out.stdout + out.stderr
    assert json.load(open(tmp_path / "r.json"))["ok"]
