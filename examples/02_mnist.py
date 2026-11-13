"""MNIST or Fashion-MNIST with a MITSU network.

    python3 examples/02_mnist.py                          # 2-hidden-layer MLP, about 98%
    python3 examples/02_mnist.py --model deep --depth 16  # 16 residual blocks
    python3 examples/02_mnist.py --fashion

The dataset (about 11 MB, or 30 MB for Fashion-MNIST) is downloaded to --data on first use.
"""
import argparse
import _path  # noqa: F401
import mitsu_engine as me

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["mlp", "deep"], default="mlp")
ap.add_argument("--block", choices=["mlp", "glu"], default="mlp", help="branch type for --model deep")
ap.add_argument("--width", type=int, default=256)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--fashion", action="store_true")
ap.add_argument("--data", default="data")
ap.add_argument("--save", default=None, help="path for a checkpoint of the best model")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

me.manual_seed(args.seed)
Xtr, ytr, Xte, yte = me.load_mnist(args.data, "fashion" if args.fashion else "mnist")
Xtr, ytr, Xva, yva = me.train_val_split(Xtr, ytr, val_fraction=0.1)
scaler = me.Standardizer().fit(Xtr)
Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

if args.model == "mlp":
    model = me.MLP([784, args.width, args.width, 10])
else:
    model = me.DeepMitsuNet(784, 10, width=args.width, depth=args.depth, block=args.block)
print(f"{type(model).__name__}: {model.num_parameters():,} parameters")

loader = me.DataLoader(Xtr, ytr, args.batch)
trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01),
                     scheduler=me.WarmupCosine(args.lr, warmup=200, total=args.epochs * len(loader)), clip_norm=1.0)
trainer.fit(loader, me.DataLoader(Xva, yva, 512, shuffle=False), epochs=args.epochs, checkpoint=args.save)
test = trainer.evaluate(me.DataLoader(Xte, yte, 512, shuffle=False))
print(f"test accuracy {100 * test['metric']:.2f}%   test loss {test['loss']:.4f}")
