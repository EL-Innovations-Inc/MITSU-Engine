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
from ..losses import CrossEntropy
from ..models import MitsuLM, LMConfig
from .convert import _named, to_engine_array
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
        for name, p in _named(torch_model).items():
            ref = ref_grads[name]
            err = float(np.abs(to_engine_array(name, p.grad).astype(np.float64) - ref).max() / max(np.abs(ref).max(), 1e-30))
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
