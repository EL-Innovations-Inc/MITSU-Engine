"""Generation speed with and without the key/value cache.

    python3 benchmarks/generation.py

Without the cache every new token recomputes the whole window; with it, each new token is
one incremental step. Both produce the same tokens.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mitsu_engine as me

print(f"{'model':44s} {'cached':>12s} {'uncached':>12s} {'speed-up':>9s}")
for label, kw in [("d_model 128, 4 layers, context 128", dict(d_model=128, n_layers=4, n_heads=4, context=128)),
                  ("d_model 256, 6 layers, context 256", dict(d_model=256, n_layers=6, n_heads=8, context=256))]:
    me.manual_seed(0)
    model = me.MitsuLM(1024, **kw)
    prompt, n = np.arange(8), kw["context"] - 9             # stay inside one context window
    out = {}
    for cached in (True, False):
        t = time.perf_counter()
        tokens = model.generate(prompt, n, temperature=0, use_cache=cached)
        out[cached] = (n / (time.perf_counter() - t), tokens)
    assert np.array_equal(out[True][1], out[False][1])
    print(f"{label:44s} {out[True][0]:8.0f} t/s {out[False][0]:8.0f} t/s {out[True][0] / out[False][0]:8.1f}x")
