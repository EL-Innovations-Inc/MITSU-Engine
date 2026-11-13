"""A sub-word MITSU language model, start to finish, with the Python API:
train a byte-pair tokenizer, write token files to disk, train a MitsuLM, save it,
reload it, and generate text.

    python3 examples/04_language_model.py                  # about 6 minutes on a laptop CPU
    python3 examples/04_language_model.py --steps 3000 --d-model 128 --layers 4

The same thing from the command line is four lines; see the README.
"""
import argparse, os
import _path  # noqa: F401
import mitsu_engine as me

ap = argparse.ArgumentParser()
ap.add_argument("--text", default="data/shakespeare.txt", help="any UTF-8 text file; Tiny Shakespeare is downloaded if missing")
ap.add_argument("--work", default="runs/example_lm")
ap.add_argument("--vocab-size", type=int, default=1024)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--d-model", type=int, default=128)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--context", type=int, default=128)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

me.manual_seed(args.seed)
me.load_text(args.text)                                         # downloads Tiny Shakespeare on first use

# 1. tokenizer + token files on disk (the validation part is the last 10% of the file)
data_dir = os.path.join(args.work, "data")
me.prepare_corpus(args.text, data_dir, vocab_size=args.vocab_size, val_fraction=0.1)
tokenizer, train_tokens, val_tokens, meta = me.load_corpus(data_dir)

# 2. the model
config = me.LMConfig(vocab_size=meta["vocab_size"], context=args.context, d_model=args.d_model, n_layers=args.layers,
                     n_heads=args.heads, dropout=0.1)
model = me.MitsuLM(config)
print(f"MitsuLM: {model.num_parameters():,} parameters")

# 3. training
train, val = me.TextBatcher(train_tokens, args.context, args.batch), me.TextBatcher(val_tokens, args.context, args.batch)
trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1), metric=None, clip_norm=1.0,
                     scheduler=me.WarmupCosine(args.lr, warmup=100, total=args.steps, min_lr=args.lr / 10))
trainer.fit_steps(train, args.steps, eval_fn=lambda: me.evaluate_lm(model, val.windows(args.batch, 64)),
                  eval_every=max(args.steps // 4, 1), log_every=max(args.steps // 20, 1),
                  state_path=os.path.join(args.work, "state.npz"), state_every=max(args.steps // 4, 1))

# 4. save, reload, generate
model.save(os.path.join(args.work, "model.npz"))
model = me.MitsuLM.load(os.path.join(args.work, "model.npz"))
print("\n--- sample ---")
print(tokenizer.decode(model.generate(tokenizer.encode("ROMEO:"), 150, temperature=0.8, top_k=40)[0]))
