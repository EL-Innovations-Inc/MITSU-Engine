"""Training-step and generation speed: the numpy engine against its PyTorch twin.

    python3 benchmarks/twin_speed.py                    # CPU
    python3 benchmarks/twin_speed.py --device cuda      # numpy on the CPU against PyTorch on a GPU
    python3 benchmarks/twin_speed.py --device cuda --amp bf16

Both engines hold the same weights and train on the same batch.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mitsu_engine as me
from mitsu_engine import twin
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"])
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--skip-numpy-above", type=int, default=60_000_000, help="do not time the numpy engine on models larger than this")
args = ap.parse_args()
device = twin.pick_device(args.device)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def best(fn, n):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t)
    return min(times)


print(f"PyTorch {torch.__version__} on {device}, mixed precision {args.amp}, {torch.get_num_threads()} CPU threads\n")
print(f"{'model':34s} {'parameters':>11s} {'numpy step':>11s} {'torch step':>11s} {'ratio':>7s} {'numpy gen':>10s} {'torch gen':>10s}")
for label, kw, batch in [("d_model 128, 4 layers, ctx 128", dict(d_model=128, n_layers=4, n_heads=4, context=128), 16),
                         ("d_model 256, 6 layers, ctx 256", dict(d_model=256, n_layers=6, n_heads=8, context=256), 8),
                         ("d_model 512, 8 layers, ctx 256", dict(d_model=512, n_layers=8, n_heads=8, context=256), 4)]:
    cfg = me.LMConfig(vocab_size=4096, **kw)
    ref = me.MitsuLM(cfg, rng=0)
    net = twin.TorchMitsuLM.from_engine(ref, device=device)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 4096, (batch, cfg.context)), rng.integers(0, 4096, (batch, cfg.context))
    tt = twin.TorchTrainer(net, twin.EngineAdamW(net, lr=1e-4), clip_norm=1.0, amp=args.amp, log=None)
    tt.train_step(x, y)                                            # warm-up
    t_torch = best(lambda: tt.train_step(x, y), args.steps)
    n = min(64, cfg.context - 9)
    g_torch = n / best(lambda: net.generate(np.arange(8), n, temperature=0), 2)
    if ref.num_parameters() <= args.skip_numpy_above:
        rt = me.Trainer(ref, me.AdamW(ref.parameters(), lr=1e-4), metric=None, clip_norm=1.0, log=None)
        t_numpy = best(lambda: rt.train_step(x, y), max(args.steps // 2, 2))
        g_numpy = n / best(lambda: ref.generate(np.arange(8), n, temperature=0), 2)
        print(f"{label:34s} {ref.num_parameters():11,d} {t_numpy * 1e3:8.0f} ms {t_torch * 1e3:8.0f} ms {t_numpy / t_torch:6.1f}x {g_numpy:6.0f} t/s {g_torch:6.0f} t/s")
    else:
        print(f"{label:34s} {ref.num_parameters():11,d} {'':>11s} {t_torch * 1e3:8.0f} ms {'':>7s} {'':>10s} {g_torch:6.0f} t/s")
