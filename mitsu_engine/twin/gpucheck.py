"""One command that tells you whether the PyTorch twin is correct on a given device.

    python -m mitsu_engine gpu-check                       # the best device available
    python -m mitsu_engine gpu-check --device cuda --out gpu_report.json

It runs the numpy engine on the CPU as the reference and the twin on the device, and
checks, in order: that every kind of network gives the same outputs and gradients; that
training follows the same trajectory; that training works under each mixed-precision mode;
that a training state moves between the engines; that an interrupted run resumes; that
generation returns the same text; and how fast a training step is. Each check is isolated:
one that raises an error is reported as failed, with the error, and the rest still run.

This is the same code path on every device, so running it on the CPU (which the test
suite does) verifies the checks themselves. Only running it on a GPU verifies the GPU.
"""
import json, os, platform, tempfile, time, traceback
import numpy as np
import torch
from .. import config as engine_config
from .. import (MitsuLM, LMConfig, MitsuConvNet, Trainer, AdamW, WarmupCosine, TextBatcher, DataLoader, __version__)
from . import parity
from .mirror import to_torch
from .model import TorchMitsuLM
from .optim import EngineAdamW
from .trainer import TorchTrainer, AMP_DTYPES
from .convert import engine_state

LM = dict(vocab_size=61, context=24, d_model=32, n_layers=2, n_heads=4)


def _gap(model, net):
    """Largest absolute weight difference between an engine model and a PyTorch model."""
    state = engine_state(net)
    return max(float(np.abs(state[n] - p.data).max()) for n, p in model.named_parameters())


def _tokens(vocab=61, n=4000, seed=0):
    return np.random.default_rng(seed).integers(0, vocab, n)


def _schedule(steps, lr=2e-3):
    return WarmupCosine(lr, 5, steps, min_lr=lr / 10)


def _numpy_lm(steps, seed=3, dropout=0.0, lr=2e-3):
    """(model, trainer, batcher) for a small language-model run in the numpy engine. The
    seed fixes the initial weights and the batches."""
    engine_config.manual_seed(seed)
    model = MitsuLM(LMConfig(**{**LM, "dropout": dropout}))
    trainer = Trainer(model, AdamW(model.parameters(), lr=lr, weight_decay=0.1), metric=None, clip_norm=0.5, scheduler=_schedule(steps), log=None)
    return model, trainer, TextBatcher(_tokens(), LM["context"], 4)


def _torch_lm(device, steps, seed=3, dropout=0.0, lr=2e-3):
    """The same run on the PyTorch twin: the same seed gives the same weights and batches."""
    engine_config.manual_seed(seed)
    torch.manual_seed(seed)
    model = TorchMitsuLM(LMConfig(**{**LM, "dropout": dropout})).to(device)
    trainer = TorchTrainer(model, EngineAdamW(model, lr=lr, weight_decay=0.1), clip_norm=0.5, scheduler=_schedule(steps), log=None)
    return model, trainer, TextBatcher(_tokens(), LM["context"], 4)


def _bars(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.3, (n, 1, 12, 12)).astype(engine_config.get_dtype())
    y = rng.integers(0, 2, n)
    for i in range(n):
        r, c = rng.integers(1, 11), rng.integers(0, 6)
        if y[i]:
            x[i, 0, r, c:c + 6] += 2.0
        else:
            x[i, 0, c:c + 6, r] += 2.0
    return x, y


class _float64:
    """Build engine models in float64 inside the block."""

    def __enter__(self):
        self.previous = engine_config.get_dtype()
        engine_config.set_dtype(np.float64)

    def __exit__(self, *exc):
        engine_config.set_dtype(self.previous)
        return False


# ---- the checks: each returns (passed, one-line detail, extra data) ---------------------

def check_language_model(device, amp):
    ok, rows = parity.check(device=device, amp=amp)
    worst = max((r for r in rows if "logits" in r), key=lambda r: r["logits"])
    return ok, f"largest logit difference {worst['logits']:.1e} ({worst['precision']})", rows


def check_every_network(device):
    ok, rows = parity.check_models(device)
    worst = max(rows, key=lambda r: r["gradient"])
    return ok, f"{len(rows)} model and precision pairs; largest gradient difference {worst['gradient']:.1e} ({worst['model']}, {worst['precision']})", rows


def check_same_training_run(device, steps=20):
    with _float64():
        ref, rt, rb = _numpy_lm(steps)
        rt.fit_steps(rb, steps)
        net, tt, tb = _torch_lm(device, steps)
        tt.fit_steps(tb, steps)
        gap = _gap(ref, net)
    return gap < 1e-9, f"language model, {steps} steps in float64: largest weight difference {gap:.1e}", {"gap": gap}


def check_conv_training_run(device, steps=10):
    with _float64():
        X, y = _bars(80, 0)
        ref = MitsuConvNet(1, 2, channels=(8, 16), blocks=1, rng=0)
        net = to_torch(ref, device=device)
        rt = Trainer(ref, AdamW(ref.parameters(), lr=2e-3, weight_decay=0.05), clip_norm=1.0, log=None)
        tt = TorchTrainer(net, EngineAdamW(net, lr=2e-3, weight_decay=0.05), metric="accuracy", clip_norm=1.0, log=None)
        for i in range(steps):
            b = slice(8 * (i % 10), 8 * (i % 10) + 8)
            rt.train_step(X[b], y[b]), tt.train_step(X[b], y[b])
        gap = _gap(ref, net)
    return gap < 1e-9, f"conv network, {steps} steps in float64: largest weight difference {gap:.1e}", {"gap": gap}


def check_learning(device, amp):
    """A language model memorises a sequence and a conv network learns a small task."""
    cfg = LMConfig(vocab_size=32, context=16, d_model=32, n_layers=2, n_heads=4)
    seq = np.random.default_rng(5).integers(0, 32, 16)
    engine_config.manual_seed(0)
    net = TorchMitsuLM(cfg).to(device)
    trainer = TorchTrainer(net, EngineAdamW(net, lr=3e-3), clip_norm=1.0, amp=amp, log=None)
    batcher = TextBatcher(np.tile(seq, 60), 16, 8)
    losses = [trainer.train_step(next_batch=batcher)[0] for _ in range(150)]
    finite = all(bool(torch.isfinite(p).all()) for p in net.parameters())
    lm_ok = finite and losses[-1] < 0.25 * losses[0]

    X, y = _bars(400, 0)
    conv = to_torch(MitsuConvNet(1, 2, channels=(8, 16), blocks=1, rng=0), device=device)
    ct = TorchTrainer(conv, EngineAdamW(conv, lr=3e-3), metric="accuracy", clip_norm=1.0, amp=amp, log=None)
    ct.fit(DataLoader(X, y, 50, rng=0), epochs=4)
    Xte, yte = _bars(200, 1)
    acc = ct.evaluate(DataLoader(Xte, yte, 200, shuffle=False))["metric"]
    return lm_ok and acc > 0.9, f"language-model loss {losses[0]:.2f} -> {losses[-1]:.3f}; conv accuracy {100 * acc:.1f}%", {"loss_start": losses[0], "loss_end": losses[-1], "conv_accuracy": acc}


def check_state_crosses_engines(device):
    """6 steps on the device, a state file, 6 steps in numpy: equal to 12 steps in numpy."""
    with _float64(), tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "state.npz")
        whole, wt, wb = _numpy_lm(12)
        wt.fit_steps(wb, 12)
        _, tt, tb = _torch_lm(device, 12)
        tt.fit_steps(tb, 6, state_path=path)
        resumed, rt, rb = _numpy_lm(12, seed=99)              # a different seed: everything must come from the file
        rt.load_state(path)                                   # restores the weights, the optimiser and the batch stream
        rt.fit_steps(rb, 12)
        gap = max(float(np.abs(p.data - q.data).max()) for (_, p), (_, q) in zip(whole.named_parameters(), resumed.named_parameters()))
    return gap < 1e-9, f"largest weight difference from an uninterrupted numpy run {gap:.1e}", {"gap": gap}


def check_resume(device):
    """8 steps against 4 steps, a state file, and 4 more, all on the device, with dropout."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "state.npz")
        whole, wt, wb = _torch_lm(device, 8, dropout=0.1)
        wt.fit_steps(wb, 8)
        _, t1, b1 = _torch_lm(device, 8, dropout=0.1)
        t1.fit_steps(b1, 4, state_path=path)
        resumed, t2, b2 = _torch_lm(device, 8, seed=77, dropout=0.1)
        t2.load_state(path)
        t2.fit_steps(b2, 8)
        gap = max((p.detach() - q.detach()).abs().max().item() for p, q in zip(whole.parameters(), resumed.parameters()))
    exact = gap == 0.0
    # bit-for-bit is required on the CPU; a GPU may reorder sums, so there a small gap is accepted
    ok = exact if torch.device(device).type == "cpu" else gap < 1e-3
    return ok, ("identical to an uninterrupted run, bit for bit" if exact else f"largest weight difference from an uninterrupted run {gap:.1e}"), {"gap": gap}


def check_generation(device):
    dtype = np.float32 if torch.device(device).type == "mps" else np.float64
    previous = engine_config.get_dtype()
    engine_config.set_dtype(dtype)
    try:
        ref = MitsuLM(LMConfig(**LM), rng=0)
    finally:
        engine_config.set_dtype(previous)
    net = TorchMitsuLM.from_engine(ref, device=device)
    prompt = np.random.default_rng(1).integers(0, 61, (2, 5))
    same = []
    for kwargs in (dict(temperature=0.0), dict(temperature=0.9), dict(temperature=1.1, top_k=7), dict(temperature=0.8, top_p=0.9)):
        a = ref.generate(prompt, 60, rng=11, **kwargs)
        b = net.generate(prompt, 60, rng=11, **kwargs)
        c = net.generate(prompt, 60, rng=11, use_cache=False, **kwargs)
        same.append(bool(np.array_equal(a, b) and np.array_equal(b, c)))
    return all(same), f"{sum(same)} of {len(same)} sampling settings give the same 60 tokens as numpy, with and without the cache", {"same": same}


def check_compile(device):
    net = TorchMitsuLM(LMConfig(**LM), rng=0).to(device)
    idx = torch.from_numpy(_tokens(n=48).reshape(2, 24)).to(device)
    plain = net(idx)
    compiled = torch.compile(net.forward)(idx)
    gap = (plain.detach() - compiled.detach()).abs().max().item()
    return gap < 1e-4, f"compiled and plain forward differ by {gap:.1e}", {"gap": gap}


def measure_speed(device, amp_modes):
    """Milliseconds per training step and generated tokens per second for one mid-sized model."""
    cfg = LMConfig(vocab_size=4096, context=256, d_model=256, n_layers=6, n_heads=8)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 4096, (8, 256)), rng.integers(0, 4096, (8, 256))
    out = {"model": "d_model 256, 6 layers, context 256, batch 8", "parameters": None, "step_ms": {}}
    sync = (lambda: torch.cuda.synchronize()) if torch.device(device).type == "cuda" else (lambda: None)
    for amp in amp_modes:
        net = TorchMitsuLM(cfg, rng=0).to(device)
        out["parameters"] = net.num_parameters()
        trainer = TorchTrainer(net, EngineAdamW(net, lr=1e-4), clip_norm=1.0, amp=amp, log=None)
        trainer.train_step(x, y)
        sync()
        times = []
        for _ in range(3):
            t = time.perf_counter()
            trainer.train_step(x, y)
            sync()
            times.append(time.perf_counter() - t)
        out["step_ms"][amp] = 1000 * min(times)
    t = time.perf_counter()
    net.generate(np.arange(8), 64, temperature=0)
    sync()
    out["generate_tokens_per_s"] = 64 / (time.perf_counter() - t)
    if torch.device(device).type == "cuda":
        out["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    detail = "  ".join(f"{amp} {ms:.0f} ms/step" for amp, ms in out["step_ms"].items()) + f"  generation {out['generate_tokens_per_s']:.0f} tokens/s"
    return True, detail, out


# ---- the runner ------------------------------------------------------------------------

def environment(device):
    info = {"mitsu_engine": __version__, "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "device": str(device)}
    if device.type == "cuda":
        info.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                    gpu_memory_gb=round(torch.cuda.get_device_properties(device).total_memory / 2 ** 30, 1))
    return info


def run(device="auto", amp=("bf16", "fp16"), compile=False, speed=True, out=None, log=print):
    """Run every check on `device`. Returns (all passed, report dict). `out` is a path for
    the report as JSON; `log` receives the lines as they are produced (None for silence)."""
    from .convert import pick_device
    device = pick_device(device)
    log = log or (lambda *a, **k: None)
    report = {"environment": environment(device), "checks": []}
    log(f"mitsu-engine {__version__} twin check on {device}" + (f" ({report['environment']['gpu']})" if "gpu" in report["environment"] else "")
        + f", PyTorch {torch.__version__}\n")
    amp = tuple(a for a in amp if a in AMP_DTYPES and a != "off")
    plan = [("language model: outputs, gradients, generation path", lambda: check_language_model(device, amp)),
            ("every network: outputs and gradients", lambda: check_every_network(device)),
            ("language model follows the numpy training run", lambda: check_same_training_run(device)),
            ("conv network follows the numpy training run", lambda: check_conv_training_run(device))]
    plan += [(f"training learns, mixed precision {mode}", (lambda m: lambda: check_learning(device, m))(mode)) for mode in ("off",) + amp]
    plan += [("a training state moves from the device to numpy", lambda: check_state_crosses_engines(device)),
             ("an interrupted run resumes on the device", lambda: check_resume(device)),
             ("generation returns the same tokens as numpy", lambda: check_generation(device))]
    if compile:
        plan.append(("torch.compile gives the same output", lambda: check_compile(device)))
    if speed:
        plan.append(("speed", lambda: measure_speed(device, ("off",) + amp)))
    if device.type == "mps":                                  # no float64 on this device
        plan = [(n, f) for n, f in plan if "follows the numpy" not in n and "moves from the device" not in n]
    saved_generator, saved_rng, saved_dtype = engine_config.get_rng(), engine_config.get_rng_state(), engine_config.get_dtype()
    for name, fn in plan:
        t = time.perf_counter()
        try:
            ok, detail, data = fn()
            entry = {"name": name, "ok": bool(ok), "detail": detail, "data": data}
        except Exception as exc:                              # one broken check must not hide the others
            entry = {"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        entry["seconds"] = time.perf_counter() - t
        report["checks"].append(entry)
        log(f"  {'pass' if entry['ok'] else 'FAIL'}  {name}\n        {entry['detail']}")
    engine_config.set_dtype(saved_dtype)                      # leave the engine's global settings as they were found
    engine_config._state["rng"] = saved_generator
    engine_config.set_rng_state(saved_rng)
    report["ok"] = all(c["ok"] for c in report["checks"])
    failed = [c["name"] for c in report["checks"] if not c["ok"]]
    log("\n" + ("PASS: the twin reproduces the numpy engine on this device." if report["ok"]
                else f"FAIL: {len(failed)} of {len(report['checks'])} checks failed: " + "; ".join(failed)))
    if out:
        with open(out, "w") as fh:
            json.dump(report, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        log(f"report written to {out}")
    return report["ok"], report
