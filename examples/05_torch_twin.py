"""The PyTorch twin, start to finish: copy a numpy model into PyTorch, measure how closely
the two agree, train the same run in both engines, and hand a checkpoint back and forth.

    pip install torch
    python3 examples/05_torch_twin.py                     # a few seconds on a CPU
    python3 examples/05_torch_twin.py --device cuda       # the same on a GPU

No data is needed: the model learns a random token sequence.
"""
import argparse, os, tempfile
import numpy as np
import _path  # noqa: F401
import mitsu_engine as me
from mitsu_engine import twin

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
ap.add_argument("--steps", type=int, default=60)
args = ap.parse_args()
device = twin.pick_device(args.device)

# 1. the same weights in both engines -----------------------------------------------------
config = me.LMConfig(vocab_size=64, context=32, d_model=64, n_layers=2, n_heads=4)
reference = me.MitsuLM(config, rng=0)                              # numpy
net = twin.TorchMitsuLM.from_engine(reference, device=device)      # PyTorch, same weights
print(f"{reference.num_parameters():,} parameters in numpy, {net.num_parameters():,} in PyTorch on {device}\n")

r = twin.compare(reference, net)
print("float32, same weights:")
print(f"  largest logit difference      {r['logits']:.1e}")
print(f"  loss difference               {r['loss']:.1e}")
print(f"  largest gradient difference   {r['gradient']:.1e}   (hand-written backward against autograd)")
print(f"  same greedy continuation      {r['generated_equal']}\n")

# 2. the same training run in both engines ------------------------------------------------
data = np.tile(np.random.default_rng(1).integers(0, 64, 48), 40)   # a sequence to memorise
lr = 3e-3


def numpy_run():
    me.manual_seed(0)
    model = me.MitsuLM(config)
    trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=lr, weight_decay=0.1), metric=None, clip_norm=1.0,
                         scheduler=me.WarmupCosine(lr, 10, args.steps), log=None)
    return model, trainer, me.TextBatcher(data, config.context, 8)


def torch_run():
    me.manual_seed(0)                                              # same seed: same initial weights, same batches
    model = twin.TorchMitsuLM(config).to(device)
    trainer = twin.TorchTrainer(model, twin.EngineAdamW(model, lr=lr, weight_decay=0.1), clip_norm=1.0,
                                scheduler=me.WarmupCosine(lr, 10, args.steps), log=None)
    return model, trainer, me.TextBatcher(data, config.context, 8)


(m_np, t_np, b_np), (m_pt, t_pt, b_pt) = numpy_run(), torch_run()
print(f"{'step':>5s} {'numpy loss':>12s} {'PyTorch loss':>13s}")
for step in range(1, args.steps + 1):
    a, b = t_np.train_step(next_batch=b_np)[0], t_pt.train_step(next_batch=b_pt)[0]
    if step % max(args.steps // 6, 1) == 0:
        print(f"{step:5d} {a:12.6f} {b:13.6f}")
state = twin.engine_state(m_pt)
print(f"largest weight difference after {args.steps} steps: {max(float(np.abs(state[n] - p.data).max()) for n, p in m_np.named_parameters()):.1e}\n")

# 3. one checkpoint format ----------------------------------------------------------------
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "model.npz")
    m_pt.save(path)                                                # written by PyTorch ...
    loaded = me.MitsuLM.load(path)                                 # ... read by numpy, no PyTorch needed
    prompt = data[:8]
    same = np.array_equal(loaded.generate(prompt, 24, temperature=0.8, top_k=10, rng=5),
                          m_pt.generate(prompt, 24, temperature=0.8, top_k=10, rng=5))
    print(f"checkpoint written by PyTorch, loaded by numpy; both sample the same 24 tokens: {same}")
