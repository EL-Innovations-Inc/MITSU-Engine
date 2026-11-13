"""Does it train at depth? Residual MITSU networks from 4 to 256 blocks on the spirals task,
next to a plain (no skip, no normalisation) stack for contrast. Residual networks train
for 15 epochs at lr 1e-3; the plain stacks get 120 epochs at lr 1e-2.

    python3 benchmarks/depth_sweep.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mitsu_engine as me

X, y = me.make_spirals(600, classes=3, rng=0)
Xtr, ytr, Xva, yva = me.train_val_split(X, y, 0.2, rng=1)
EPOCHS, LR = 15, 1e-3


def run(model, lr=LR, epochs=EPOCHS):
    loader = me.DataLoader(Xtr, ytr, 96, rng=2)
    tr = me.Trainer(model, me.AdamW(model.parameters(), lr=lr, weight_decay=0.01), clip_norm=1.0,
                    scheduler=me.WarmupCosine(lr, 50, epochs * len(loader)), log=None)
    t = time.time()
    tr.fit(loader, epochs=epochs)
    return tr.evaluate(me.DataLoader(Xva, yva, 256, shuffle=False)), time.time() - t


print(f"{'network':34s} {'parameters':>11s} {'val acc':>8s} {'val loss':>9s} {'seconds':>8s}")
for depth in (4, 16, 64, 256):
    for block in ("mlp", "glu"):
        net = me.DeepMitsuNet(2, 3, width=64, depth=depth, block=block, rng=0)
        val, sec = run(net)
        print(f"{'residual ' + block + ', ' + str(depth) + ' blocks':34s} {net.num_parameters():11,d} {val['metric']:8.3f} {val['loss']:9.3f} {sec:8.1f}")
# The plain stack gets every advantage: ten times the learning rate and eight times the epochs.
for depth in (2, 4, 8, 16):
    net = me.MLP([2] + [64] * depth + [3], rng=0)
    val, sec = run(net, lr=1e-2, epochs=8 * EPOCHS)
    print(f"{'plain stack, ' + str(depth) + ' hidden layers':34s} {net.num_parameters():11,d} {val['metric']:8.3f} {val['loss']:9.3f} {sec:8.1f}")
