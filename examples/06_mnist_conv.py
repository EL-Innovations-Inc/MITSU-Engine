"""MNIST or Fashion-MNIST with a residual MITSU convolutional network.

    python3 examples/06_mnist_conv.py                                   # numpy, on the CPU
    python3 examples/06_mnist_conv.py --backend torch --device cuda     # the same network on a GPU
    python3 examples/06_mnist_conv.py --fashion --channels 16 32 64 --blocks 2

The numpy run takes about five minutes per epoch on a 2-core CPU with the default sizes.
The dataset is downloaded to --data on first use.
"""
import argparse
import _path  # noqa: F401
import mitsu_engine as me

ap = argparse.ArgumentParser()
ap.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64], help="width of each stage")
ap.add_argument("--blocks", type=int, default=1, help="residual blocks per stage")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--train-size", type=int, default=None, help="use only this many training images")
ap.add_argument("--fashion", action="store_true")
ap.add_argument("--data", default="data")
ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
ap.add_argument("--device", default="auto")
ap.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

me.manual_seed(args.seed)
Xtr, ytr, Xte, yte = me.load_mnist(args.data, "fashion" if args.fashion else "mnist", flatten=False)    # (N, 1, 28, 28)
mean, std = Xtr.mean(), Xtr.std()
Xtr, Xte = (Xtr - mean) / std, (Xte - mean) / std
Xtr, ytr, Xva, yva = me.train_val_split(Xtr, ytr, val_fraction=0.1)
if args.train_size:
    Xtr, ytr = Xtr[:args.train_size], ytr[:args.train_size]

model = me.MitsuConvNet(1, 10, channels=args.channels, blocks=args.blocks)
print(f"MitsuConvNet: {model.num_parameters():,} parameters, {len(Xtr):,} training images")

loader, val_loader = me.DataLoader(Xtr, ytr, args.batch), me.DataLoader(Xva, yva, 500, shuffle=False)
schedule = me.WarmupCosine(args.lr, warmup=100, total=args.epochs * len(loader), min_lr=args.lr / 20)
if args.backend == "torch":
    from mitsu_engine import twin
    net = twin.to_torch(model, device=twin.pick_device(args.device))            # the same weights, in PyTorch
    trainer = twin.TorchTrainer(net, twin.EngineAdamW(net, lr=args.lr, weight_decay=0.01), loss="cross_entropy", metric="accuracy",
                                scheduler=schedule, clip_norm=1.0, amp=args.amp)
else:
    trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01), scheduler=schedule, clip_norm=1.0)
trainer.fit(loader, val_loader, epochs=args.epochs)
test = trainer.evaluate(me.DataLoader(Xte, yte, 500, shuffle=False))
print(f"test accuracy {100 * test['metric']:.2f}%   test loss {test['loss']:.4f}")
