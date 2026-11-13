"""How long do the two engines stay on the same training trajectory?

    python3 benchmarks/twin_trajectory.py                     # float64 and float32, on the CPU
    python3 benchmarks/twin_trajectory.py --device cuda       # numpy on the CPU against PyTorch on a GPU
    python3 benchmarks/twin_trajectory.py --data data/shake --d-model 128 --layers 4 --context 128 --batch 16 --only float64

Both engines start from the same weights, draw the same batches and use the same
optimiser rule, with no dropout. Any difference therefore comes from rounding: the two
do the same arithmetic in a different order. Training can amplify a rounding difference
from step to step, and how fast it does so depends on the task. In float64 the gap starts
near 1e-15 and stays far below anything that matters. In float32 it starts near 1e-7; on
this script's small default task it is still near 1e-6 after 400 steps, while on a larger
text model it can grow until the two runs are simply two equally valid runs (README,
"PyTorch twin").
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mitsu_engine as me
from mitsu_engine import twin

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--data", default=None, help="a folder written by `prepare`; default is a small synthetic task")
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--context", type=int, default=64)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--only", choices=["float64", "float32"], default=None)
args = ap.parse_args()
device = twin.pick_device(args.device)
if args.data:
    _, data, _, meta = me.load_corpus(args.data)
    vocab = meta["vocab_size"]
else:
    vocab = 256
    data = np.cumsum(np.random.default_rng(0).integers(0, 7, 60000)) % 256      # tokens with learnable structure
cfg = me.LMConfig(vocab_size=vocab, context=args.context, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
marks = [1, 10, 25, 50, 100, 200, 300, 400, 600, 800, 1000]

for name, dtype in (("float64", np.float64), ("float32", np.float32)):
    if (name == "float64" and device.type == "mps") or (args.only and args.only != name):
        continue
    me.set_dtype(dtype)
    runs = []
    for backend in ("numpy", "torch"):
        me.manual_seed(0)
        if backend == "numpy":
            model = me.MitsuLM(cfg)
            trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1), metric=None, clip_norm=1.0,
                                 scheduler=me.WarmupCosine(args.lr, 20, args.steps, min_lr=args.lr / 10), log=None)
        else:
            model = twin.TorchMitsuLM(cfg).to(device)
            trainer = twin.TorchTrainer(model, twin.EngineAdamW(model, lr=args.lr, weight_decay=0.1), clip_norm=1.0,
                                        scheduler=me.WarmupCosine(args.lr, 20, args.steps, min_lr=args.lr / 10), log=None)
        runs.append((model, trainer, me.TextBatcher(data, cfg.context, args.batch)))
    (m_np, t_np, b_np), (m_pt, t_pt, b_pt) = runs
    print(f"\n{name} on {device}\n{'step':>6s} {'numpy loss':>12s} {'torch loss':>12s} {'loss gap':>10s} {'weight gap':>11s}")
    for step in range(1, args.steps + 1):
        a, b = t_np.train_step(next_batch=b_np)[0], t_pt.train_step(next_batch=b_pt)[0]
        if step in marks or step == args.steps:
            state = twin.engine_state(m_pt)
            gap = max(float(np.abs(state[n] - p.data).max()) for n, p in m_np.named_parameters())
            print(f"{step:6d} {a:12.6f} {b:12.6f} {abs(a - b):10.1e} {gap:11.1e}")
me.set_dtype(np.float32)
