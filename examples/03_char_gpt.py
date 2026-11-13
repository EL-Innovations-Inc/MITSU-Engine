"""A character-level MITSU transformer on Tiny Shakespeare, then a sample of its writing.

    python3 examples/03_char_gpt.py                   # about 2 minutes on a laptop CPU
    python3 examples/03_char_gpt.py --steps 5000 --d-model 128 --layers 4

The text (1 MB) is downloaded to --data on first use.
"""
import argparse
import _path  # noqa: F401
import mitsu_engine as me

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--d-model", type=int, default=64)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--heads", type=int, default=2)
ap.add_argument("--context", type=int, default=64)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--ffn", choices=["glu", "mlp"], default="glu")
ap.add_argument("--data", default="data/shakespeare.txt")
ap.add_argument("--save", default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

me.manual_seed(args.seed)
text = me.load_text(args.data)
tok = me.CharTokenizer(text)
ids = tok.encode(text)
split = int(0.9 * len(ids))
train, val = me.TextBatcher(ids[:split], args.context, args.batch), me.TextBatcher(ids[split:], args.context, args.batch)

model = me.MitsuGPT(tok.vocab_size, args.context, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, ffn=args.ffn)
print(f"MitsuGPT: {model.num_parameters():,} parameters, vocabulary {tok.vocab_size}")

trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01), metric=None,
                     scheduler=me.WarmupCosine(args.lr, warmup=100, total=args.steps, min_lr=args.lr / 10), clip_norm=1.0)
trainer.fit_steps(train, args.steps, eval_fn=lambda: {"loss": trainer.evaluate(val.windows())["loss"]},
                  eval_every=max(args.steps // 4, 1), log_every=max(args.steps // 20, 1))
if args.save:
    me.save_checkpoint(args.save, model, meta={"chars": tok.chars, "context": args.context})

print("\n--- sample ---")
print(tok.decode(model.generate(tok.encode("ROMEO:"), 400, temperature=0.8, top_k=20)[0]))
