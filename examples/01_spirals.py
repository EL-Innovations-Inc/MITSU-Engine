"""A 32-layer MITSU network on a toy problem. No data to download; runs in seconds.

    python3 examples/01_spirals.py
"""
import _path  # noqa: F401
import mitsu_engine as me

me.manual_seed(0)
X, y = me.make_spirals(600, classes=3)
Xtr, ytr, Xva, yva = me.train_val_split(X, y, val_fraction=0.2)

model = me.DeepMitsuNet(d_in=2, d_out=3, width=64, depth=32)
print(f"{model.num_parameters():,} parameters in {model.depth} residual blocks")

epochs = 15
loader = me.DataLoader(Xtr, ytr, batch_size=96)
trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01),
                     scheduler=me.WarmupCosine(2e-3, warmup=50, total=epochs * len(loader)), clip_norm=1.0)
trainer.fit(loader, me.DataLoader(Xva, yva, 256, shuffle=False), epochs=epochs)
