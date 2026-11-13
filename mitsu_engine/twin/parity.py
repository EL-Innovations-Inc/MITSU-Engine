"""Measuring how closely the PyTorch twin agrees with the numpy engine.

    compare(engine_model, torch_model)     the largest differences, as a dict
    check(run=None, device="cpu")          the full comparison in float64 and float32 (and
                                           under autocast), with pass / fail per row

`python -m mitsu_engine parity --device cuda` runs `check` from the command line. It is
the first thing to run on a new machine or GPU: it shows whether that hardware and that
PyTorch build reproduce the reference.
"""
import numpy as np
import torch
from .. import config as engine_config
from ..losses import CrossEntropy, MSE
from ..models import MitsuLM, LMConfig
from .convert import layout, _to_engine
from .model import TorchMitsuLM

# Largest differences accepted by `check`. The float64 limits are close to rounding error;
# the float32 limits allow for single-precision arithmetic done in a different order.
TOLERANCE = {
    "float64": {"logits": 1e-10, "loss": 1e-11, "gradient": 1e-9, "cached": 1e-10},
    "float32": {"logits": 5e-4, "loss": 5e-5, "gradient": 5e-3, "cached": 5e-4},
}
AUTOCAST_LOSS_TOLERANCE = 5e-2        # bfloat16 / float16 against the float32 reference


def _f64(tensor):
    """A tensor as a float64 numpy array (converted on the CPU: not every device has float64)."""
    return tensor.detach().cpu().double().numpy()


def _random_tokens(cfg, batch, tokens, seed):
    rng = np.random.default_rng(seed)
    T = min(tokens or cfg.context, cfg.context)
    idx = rng.integers(0, cfg.vocab_size, (batch, T + 1))
    return idx[:, :-1], idx[:, 1:]


def compare(engine_model, torch_model, idx=None, targets=None, batch=2, tokens=32, seed=0, generate=16):
    """Run both models on the same tokens and return the largest differences.

    logits      largest absolute difference between the two logit arrays
    loss        absolute difference between the two cross-entropy losses
    gradient    largest difference between the numpy engine's hand-written gradient and
                PyTorch's autograd gradient, over all parameters, relative to the largest
                entry of that parameter's gradient; `gradient_worst` names the parameter
    cached      largest absolute logit difference along the incremental generation path
    generated_equal   True if greedy generation returns the same tokens from both

    Both models are put in evaluation mode for the comparison (dropout draws different
    random numbers in the two engines) and returned to their previous mode.
    """
    cfg = engine_model.config
    if idx is None:
        idx, targets = _random_tokens(cfg, batch, tokens, seed)
    idx, targets = np.asarray(idx, dtype=np.int64), np.asarray(targets, dtype=np.int64)
    device = next(torch_model.parameters()).device
    t_idx, t_tgt = torch.from_numpy(idx).to(device), torch.from_numpy(targets).to(device)
    modes = engine_model.training, torch_model.training
    engine_model.eval(), torch_model.eval()
    try:
        # numpy engine: forward, loss, hand-written backward
        engine_model.zero_grad()
        ref_logits = engine_model(idx)
        ref_loss, grad = CrossEntropy()(ref_logits, targets)
        engine_model.backward(grad)
        ref_grads = {n: p.grad.astype(np.float64) for n, p in engine_model.named_parameters()}
        engine_model.zero_grad()

        # PyTorch twin: forward, loss, autograd
        torch_model.zero_grad(set_to_none=True)
        logits = torch_model(t_idx)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), t_tgt.reshape(-1))
        loss.backward()
        out = {"logits": float(np.abs(_f64(logits) - ref_logits).max()),
               "loss": abs(loss.item() - ref_loss), "gradient": 0.0, "gradient_worst": None}
        for name, p, transposed, _ in layout(torch_model):
            ref = ref_grads[name]
            err = float(np.abs(_to_engine(p.grad, transposed).astype(np.float64) - ref).max() / max(np.abs(ref).max(), 1e-30))
            if err >= out["gradient"]:
                out["gradient"], out["gradient_worst"] = err, name
        torch_model.zero_grad(set_to_none=True)

        # the incremental generation path: a prompt, then one token at a time
        T = idx.shape[1]
        cut = max(T // 2, 1)
        with engine_config.no_grad():
            r, r_cache = engine_model.step(idx[:, :cut])
        t, t_cache = torch_model.step(t_idx[:, :cut])
        worst = float(np.abs(_f64(t) - r).max())
        for i in range(cut, T):
            r, r_cache = engine_model.step(idx[:, i:i + 1], r_cache)
            t, t_cache = torch_model.step(t_idx[:, i:i + 1], t_cache)
            worst = max(worst, float(np.abs(_f64(t) - r).max()))
        out["cached"] = worst

        prompt = idx[:, :min(4, T)]
        a = engine_model.generate(prompt, generate, temperature=0.0)
        b = torch_model.generate(prompt, generate, temperature=0.0)
        out["generated_equal"] = bool(np.array_equal(a, b))
    finally:
        engine_model.train(modes[0]), torch_model.train(modes[1])
    return out


def compare_models(engine_model, torch_model, x, y=None, loss="cross_entropy", seed=0):
    """Run any engine model and its twin.to_torch copy on the same input and return the
    largest differences.

    output           largest absolute difference between the two outputs
    loss             absolute difference between the two losses
    gradient         largest parameter-gradient difference (engine backward against PyTorch
                     autograd), relative to that parameter's largest entry; `gradient_worst`
                     names the parameter
    input_gradient   the same for the gradient with respect to a floating-point input

    With targets `y` the loss is "cross_entropy" or "mse". Without, the outputs are reduced
    to a number with a fixed random weighting, which exercises every output. Both models
    are compared in evaluation mode and returned to their previous mode."""
    x = np.asarray(x)
    first = next(torch_model.parameters())
    device, dtype = first.device, first.dtype
    is_float = np.issubdtype(x.dtype, np.floating)
    t_x = torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dtype if is_float else torch.long)
    t_x.requires_grad_(is_float)
    modes = engine_model.training, torch_model.training
    engine_model.eval(), torch_model.eval()
    try:
        engine_model.zero_grad()
        ref_out = engine_model(x.astype(first.detach().cpu().numpy().dtype) if is_float else x)
        out = torch_model(t_x)
        if y is None:
            weights = np.random.default_rng(seed).standard_normal(ref_out.shape).astype(ref_out.dtype)
            ref_loss, grad = float((ref_out * weights).sum()), weights
            t_loss = (out * torch.from_numpy(weights).to(device=device, dtype=out.dtype)).sum()
        else:
            y = np.asarray(y)
            ref_loss, grad = (CrossEntropy() if loss == "cross_entropy" else MSE())(ref_out, y)
            t_y = torch.from_numpy(np.ascontiguousarray(y)).to(device)
            t_loss = (torch.nn.functional.cross_entropy(out.reshape(-1, out.shape[-1]), t_y.reshape(-1).long()) if loss == "cross_entropy"
                      else torch.nn.functional.mse_loss(out, t_y.to(out.dtype)))
        ref_gx = engine_model.backward(grad)
        torch_model.zero_grad(set_to_none=True)
        t_loss.backward()
        result = {"output": float(np.abs(_f64(out) - ref_out).max()), "loss": abs(t_loss.item() - ref_loss), "gradient": 0.0, "gradient_worst": None}
        refs = dict(engine_model.named_parameters())
        for name, p, transposed, _ in layout(torch_model):
            ref = refs[name].grad.astype(np.float64)
            got = np.zeros_like(ref) if p.grad is None else _to_engine(p.grad, transposed).astype(np.float64)
            err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
            if err >= result["gradient"]:
                result["gradient"], result["gradient_worst"] = err, name
        if is_float and ref_gx is not None:
            result["input_gradient"] = float(np.abs(_f64(t_x.grad) - ref_gx).max() / max(np.abs(ref_gx).max(), 1e-30))
        engine_model.zero_grad()
        torch_model.zero_grad(set_to_none=True)
    finally:
        engine_model.train(modes[0]), torch_model.train(modes[1])
    return result


def model_zoo(dtype=np.float64, seed=0):
    """One small instance of every kind of network in the engine, with an input for each:
    [(label, engine model, input array)]. Used to check that each one has a faithful twin."""
    from .. import (MLP, DeepMitsuNet, MitsuConvNet, MitsuGPT, Sequential, Residual, Linear, LayerNorm, Flatten, Conv2d, MaxPool2d,
                    AvgPool2d, GroupNorm, MultiHeadSelfAttention, TransformerBlock, FeedForward, MitsuGLU, MITSU, GELU, SiLU, Tanh, ReLU)
    previous = engine_config.get_dtype()
    engine_config.set_dtype(dtype)
    try:
        r = np.random.default_rng(seed)
        f = lambda *shape: r.standard_normal(shape).astype(dtype)
        zoo = [
            ("MLP", MLP([12, 16, 16, 5], rng=1), f(4, 12)),
            ("DeepMitsuNet, mlp blocks", DeepMitsuNet(12, 5, width=16, depth=4, rng=2), f(4, 12)),
            ("DeepMitsuNet, MitsuGLU blocks", DeepMitsuNet(12, 5, width=18, depth=3, block="glu", rng=3), f(4, 12)),
            ("MitsuConvNet", MitsuConvNet(2, 5, channels=(8, 16), blocks=2, groups=4, rng=4), f(3, 2, 12, 12)),
            ("MitsuGPT", MitsuGPT(31, context=10, d_model=16, n_layers=2, n_heads=2, rng=5), r.integers(0, 31, (3, 10))),
            ("MitsuGPT, mlp feed-forward", MitsuGPT(31, context=10, d_model=16, n_layers=1, n_heads=4, ffn="mlp", rng=6), r.integers(0, 31, (3, 7))),
            ("every layer in one stack", Sequential(
                Conv2d(2, 4, 3, stride=2, padding=1, bias=False, rng=7), GroupNorm(1, 4), SiLU(), MaxPool2d(2), AvgPool2d(3, 1),
                Flatten(), Linear(16, 12, rng=7), LayerNorm(12), GELU(), Residual(FeedForward(12, 20, activation="tanh", rng=7)),
                MitsuGLU(12, 10, bias=True, c=1.4, gate_mult=2.0, rng=7), ReLU(), Linear(12, 6, bias=False, rng=7), MITSU(c=0.8, gain=3.0)), f(3, 2, 16, 16)),
            ("attention without mask, with bias", Sequential(
                MultiHeadSelfAttention(12, 3, causal=False, bias=True, rng=8), TransformerBlock(12, 2, ffn="mlp", activation="gelu", causal=False, rng=8),
                TransformerBlock(12, 2, norm="rms", rope=True, rng=8), Tanh()), f(2, 6, 12)),
        ]
        for _, model, _ in zoo:                                       # move gains and biases off their defaults
            for p in model.parameters():
                if not p.decay:
                    p.data += 0.2 * r.standard_normal(p.data.shape)
    finally:
        engine_config.set_dtype(previous)
    return zoo


MODEL_TOLERANCE = {"float64": {"output": 1e-10, "gradient": 1e-9, "input_gradient": 1e-9},
                   "float32": {"output": 5e-4, "gradient": 5e-3, "input_gradient": 5e-3}}


def check_models(device="cpu"):
    """Mirror every network in `model_zoo` on `device` and compare it with the engine, in
    float64 and float32. Returns (ok, rows)."""
    from .mirror import to_torch
    device = torch.device(device)
    rows = []
    for name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        if name == "float64" and device.type == "mps":
            continue
        for label, model, x in model_zoo(np_dtype):
            row = {"model": label, "precision": name, **compare_models(model, to_torch(model, device=device), x)}
            limit = MODEL_TOLERANCE[name]
            row["ok"] = all(row[k] <= limit[k] for k in limit if k in row)
            rows.append(row)
    return all(r["ok"] for r in rows), rows


def _build(cfg, params, dtype):
    """The numpy reference in `dtype`, with weights `params` (or fresh ones from seed 0)."""
    previous = engine_config.get_dtype()
    engine_config.set_dtype(dtype)
    try:
        model = MitsuLM(cfg, rng=0)
        if params is not None:
            model.load_state_dict(params)
    finally:
        engine_config.set_dtype(previous)
    return model


def check(run=None, device="cpu", batch=2, tokens=32, seed=0, cfg=None, amp=("bf16",)):
    """Compare the two engines in float64 and float32 on `device`, and the autocast loss
    against the float32 reference. Returns (ok, rows); each row is a dict with the
    precision, the measured differences and whether they are inside TOLERANCE.

    run   a checkpoint file (model.npz) to test real weights; None builds a small model.
    cfg   an LMConfig to use when `run` is None.
    """
    from ..io import read_checkpoint
    params = None
    if run is not None:
        params, _, meta = read_checkpoint(run)
        cfg = LMConfig.from_dict(meta["config"])
    cfg = cfg or LMConfig(vocab_size=257, context=48, d_model=64, n_layers=3, n_heads=4)
    device = torch.device(device)
    rows = []
    for name, np_dtype, t_dtype in (("float64", np.float64, torch.float64), ("float32", np.float32, torch.float32)):
        if name == "float64" and device.type == "mps":
            rows.append({"precision": name, "skipped": "this device has no float64"})
            continue
        ref = _build(cfg, params, np_dtype)
        net = TorchMitsuLM.from_engine(ref, device=device, dtype=t_dtype)
        row = {"precision": name, **compare(ref, net, batch=batch, tokens=tokens, seed=seed)}
        limit = TOLERANCE[name]
        row["ok"] = all(row[k] <= limit[k] for k in limit) and row["generated_equal"]
        rows.append(row)
        if name == "float32":
            idx, tgt = _random_tokens(cfg, batch, tokens, seed)
            ref.eval()
            with engine_config.no_grad():
                ref_loss, _ = CrossEntropy()(ref(idx), tgt)
            net.eval()
            for mode in amp:
                dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[mode]
                try:
                    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
                        loss = float(net.loss(torch.from_numpy(idx).to(device), torch.from_numpy(tgt).to(device)))
                    diff = abs(loss - ref_loss)
                    rows.append({"precision": f"autocast {mode}", "loss": diff, "ok": bool(np.isfinite(loss)) and diff <= AUTOCAST_LOSS_TOLERANCE})
                except Exception as exc:        # a device or build that cannot autocast to this type
                    rows.append({"precision": f"autocast {mode}", "skipped": f"{type(exc).__name__}: {exc}"})
    return all(r.get("ok", True) for r in rows), rows


def format_report(rows, device="cpu"):
    """`check`'s rows as a table."""
    lines = [f"numpy engine against PyTorch twin on {device}  (torch {torch.__version__})", "",
             f"{'precision':15s} {'logits':>10s} {'loss':>10s} {'gradient':>10s} {'cached':>10s}  {'same text':9s}  result",
             "-" * 78]
    for r in rows:
        if "skipped" in r:
            lines.append(f"{r['precision']:15s} skipped: {r['skipped']}")
            continue
        cell = lambda k: f"{r[k]:10.2e}" if k in r else f"{'':>10s}"
        same = {True: "yes", False: "NO"}.get(r.get("generated_equal"), "")
        lines.append(f"{r['precision']:15s} {cell('logits')} {cell('loss')} {cell('gradient')} {cell('cached')}  {same:9s}  {'pass' if r['ok'] else 'FAIL'}")
        if r.get("gradient_worst") and not r["ok"]:
            lines.append(f"{'':15s} largest gradient difference in {r['gradient_worst']}")
    lines += ["", "logits, loss, cached: largest absolute difference.  gradient: largest difference relative to",
              "the parameter's largest gradient entry (numpy hand-written backward against PyTorch autograd)."]
    return "\n".join(lines)
