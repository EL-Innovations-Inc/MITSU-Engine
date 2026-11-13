"""How big and how fast: time one full training step (forward, backward, AdamW update)
for networks from 0.5 million to about 67 million parameters.

    python3 benchmarks/throughput.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mitsu_engine as me

rng = np.random.default_rng(0)
BATCH = 64
print(f"{'network':46s} {'parameters':>12s} {'ms / step':>10s} {'samples / s':>12s}")
for label, build, make_batch in [
    ("DeepMitsuNet width 128, 16 blocks", lambda: me.DeepMitsuNet(784, 10, 128, 16), None),
    ("DeepMitsuNet width 256, 32 blocks", lambda: me.DeepMitsuNet(784, 10, 256, 32), None),
    ("DeepMitsuNet width 512, 64 blocks", lambda: me.DeepMitsuNet(784, 10, 512, 64), None),
    ("DeepMitsuNet (MitsuGLU) width 512, 32 blocks", lambda: me.DeepMitsuNet(784, 10, 512, 32, block="glu"), None),
    ("MitsuGPT d_model 128, 4 layers, context 128", lambda: me.MitsuGPT(256, 128, 128, 4, 4), "gpt"),
]:
    me.manual_seed(0)
    model = build()
    opt = me.AdamW(model.parameters(), lr=1e-3)
    if make_batch == "gpt":
        x = rng.integers(0, 256, (16, 128)); y = rng.integers(0, 256, (16, 128)); n = 16
    else:
        x = rng.standard_normal((BATCH, 784)).astype(np.float32); y = rng.integers(0, 10, BATCH); n = BATCH
    loss_fn, times = me.CrossEntropy(), []
    for i in range(4):
        t = time.perf_counter()
        opt.zero_grad()
        loss, g = loss_fn(model(x), y)
        model.backward(g)
        opt.step()
        times.append(time.perf_counter() - t)
    ms = 1e3 * float(np.median(times[1:]))
    print(f"{label:46s} {model.num_parameters():12,d} {ms:10.1f} {n / (ms / 1e3):12.0f}")
