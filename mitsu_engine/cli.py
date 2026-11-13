"""Command line for language-model work.

    python -m mitsu_engine prepare  --text book.txt --out data/book --vocab-size 4096
    python -m mitsu_engine train    --data data/book --out runs/book --steps 5000
    python -m mitsu_engine train    --data data/book --out runs/book --steps 8000 --resume
    python -m mitsu_engine generate --run runs/book --prompt "Once upon a time"
    python -m mitsu_engine info     --run runs/book

With PyTorch installed, the same commands run on the PyTorch twin, and `parity` measures
how closely the twin reproduces the numpy engine on a given machine:

    python -m mitsu_engine gpu-check --device cuda --out gpu_report.json
    python -m mitsu_engine parity   --device cuda
    python -m mitsu_engine train    --data data/book --out runs/book --backend torch --device cuda --amp bf16
    python -m mitsu_engine generate --run runs/book --backend torch --device cuda

A run folder holds model.npz (weights and config), state.npz (for --resume),
tokenizer.json, log.jsonl (one line per logged step) and run.json (the settings). The
files are the same whichever backend wrote them, so a run started on one can be resumed,
sampled or inspected on the other.
"""
import argparse, json, math, os, shutil, sys, time
import numpy as np
from . import config
from .corpus import prepare_corpus, load_corpus
from .data import TextBatcher
from .models import MitsuLM, LMConfig
from .optim import AdamW, WarmupCosine
from .tokenizer import BPETokenizer
from .trainer import Trainer, evaluate_lm


def _twin(device):
    """Import PyTorch and the twin only when a command asks for them. Returns (twin, torch, device)."""
    try:
        from . import twin
    except ImportError as exc:
        sys.exit(str(exc))
    import torch
    return twin, torch, twin.pick_device(device)


def cmd_parity(a):
    twin, torch, device = _twin(a.device)
    cfg = None if a.run else LMConfig(vocab_size=a.vocab_size, context=a.context, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads)
    run = os.path.join(a.run, "model.npz") if a.run and os.path.isdir(a.run) else a.run
    ok, rows = twin.check(run=run, device=device, batch=a.batch, tokens=a.tokens, seed=a.seed, cfg=cfg, amp=tuple(a.amp))
    print(twin.format_report(rows, device))
    print("\nPASS: the PyTorch twin reproduces the numpy engine here." if ok else "\nFAIL: at least one difference is outside the tolerance.")
    sys.exit(0 if ok else 1)


def cmd_gpu_check(a):
    twin, torch, device = _twin(a.device)
    from .twin import gpucheck
    ok, _ = gpucheck.run(device, amp=tuple(a.amp), compile=a.compile, speed=not a.no_speed, out=a.out)
    sys.exit(0 if ok else 1)


def cmd_prepare(a):
    prepare_corpus(a.text, a.out, vocab_size=a.vocab_size, val_fraction=a.val_fraction, special_tokens=a.special or ())


def cmd_train(a):
    tokenizer, train, val, meta = load_corpus(a.data)
    os.makedirs(a.out, exist_ok=True)
    run_file, state_file = os.path.join(a.out, "run.json"), os.path.join(a.out, "state.npz")
    if a.resume:
        if not os.path.exists(state_file):
            sys.exit(f"nothing to resume: {state_file} does not exist")
        saved = json.load(open(run_file))
        for key in ("d_model", "layers", "heads", "context", "ffn_hidden", "untied", "dropout", "batch", "accumulate", "lr", "warmup",
                    "weight_decay", "clip", "seed", "eval_windows"):
            setattr(a, key, saved[key])                       # the run's own settings win; only --steps may change
        if a.steps is None:
            a.steps = saved["steps"]
    else:
        if os.path.exists(state_file):
            sys.exit(f"{a.out} already holds a run; add --resume to continue it or choose another --out")
        a.steps = a.steps or 2000
    config.manual_seed(a.seed)
    cfg = LMConfig(vocab_size=meta["vocab_size"], context=a.context, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads,
                   ffn_hidden=a.ffn_hidden, dropout=a.dropout, tie_embeddings=not a.untied)
    # the schedule is fixed by the run's first --steps, so extending a run does not rewrite its past
    schedule_steps = json.load(open(run_file))["schedule_steps"] if a.resume else a.steps
    scheduler = WarmupCosine(a.lr, a.warmup, schedule_steps, min_lr=a.lr / 10)
    if a.backend == "torch":
        twin, torch, device = _twin(a.device)
        torch.manual_seed(a.seed)                         # PyTorch's generator only decides dropout
        model = twin.TorchMitsuLM(cfg).to(device)         # initial weights drawn by the numpy engine
        if a.compile:
            model.forward = torch.compile(model.forward)
        trainer = twin.TorchTrainer(model, twin.EngineAdamW(model, lr=a.lr, weight_decay=a.weight_decay), scheduler=scheduler,
                                    clip_norm=a.clip, amp=a.amp)
        evaluate_model = lambda batches: twin.evaluate_lm(model, batches, a.amp)
        print(f"backend: PyTorch {torch.__version__} on {device}, mixed precision {a.amp}")
    else:
        model = MitsuLM(cfg)
        trainer = Trainer(model, AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay), metric=None, clip_norm=a.clip,
                          scheduler=scheduler)
        evaluate_model = lambda batches: evaluate_lm(model, batches)
    batcher = TextBatcher(train, a.context, a.batch)
    val_batcher = TextBatcher(val, a.context, a.batch) if val is not None else None
    if a.resume:
        trainer.load_state(state_file)
        print(f"resumed at step {trainer.step}")
    else:
        shutil.copy(os.path.join(a.data, "tokenizer.json"), os.path.join(a.out, "tokenizer.json"))
    settings = {**{k: v for k, v in vars(a).items() if k not in ("func", "resume", "stop_after")}, "schedule_steps": schedule_steps, "data_meta": meta}
    json.dump(settings, open(run_file, "w"), indent=1)
    print(f"MitsuLM: {model.num_parameters():,} parameters   vocabulary {cfg.vocab_size}   context {cfg.context}   "
          f"{meta['train_tokens']:,} training tokens")

    def evaluate():
        return evaluate_model(val_batcher.windows(a.batch, a.eval_windows)) if val_batcher else {}

    target = a.steps if not a.stop_after else min(a.steps, trainer.step + a.stop_after)
    trainer.fit_steps(batcher, target, eval_fn=evaluate if val_batcher else None, eval_every=a.eval_every, log_every=a.log_every,
                      accumulate=a.accumulate, state_path=state_file, state_every=a.eval_every, log_file=os.path.join(a.out, "log.jsonl"))
    final = evaluate()
    model.save(os.path.join(a.out, "model.npz"), step=trainer.step, **{"val_" + k: v for k, v in final.items()})
    print(f"saved {os.path.join(a.out, 'model.npz')} at step {trainer.step} of {a.steps}"
          + (f"   val loss {final['loss']:.4f}   perplexity {final['perplexity']:.2f}" if final else ""))


def cmd_generate(a):
    if a.backend == "torch":
        twin, torch, device = _twin(a.device)
        model = twin.TorchMitsuLM.load(os.path.join(a.run, "model.npz"), device=device)
    else:
        model = MitsuLM.load(os.path.join(a.run, "model.npz"))
    tokenizer = BPETokenizer.load(os.path.join(a.run, "tokenizer.json"))
    prompt = tokenizer.encode(a.prompt) if a.prompt else np.array([tokenizer.encode("\n")[0]])
    t = time.time()
    out = model.generate(prompt, a.tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, rng=a.seed)
    print(tokenizer.decode(out[0]))
    print(f"\n[{a.tokens} tokens in {time.time() - t:.1f}s]", file=sys.stderr)


def cmd_info(a):
    model = MitsuLM.load(os.path.join(a.run, "model.npz"))
    print(model.summary())
    print("\nconfig:", json.dumps(model.config.to_dict()))
    log = os.path.join(a.run, "log.jsonl")
    if os.path.exists(log):
        rows = [json.loads(line) for line in open(log)]
        evals = [r for r in rows if "val_loss" in r]
        if evals:
            print(f"last evaluation: step {evals[-1]['step']}   val loss {evals[-1]['val_loss']:.4f}   perplexity {evals[-1]['val_perplexity']:.2f}")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m mitsu_engine", description="Prepare data, train, and sample from a MITSU language model.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="train a tokenizer and write token files")
    s.add_argument("--text", required=True, help="a UTF-8 text file")
    s.add_argument("--out", required=True, help="folder for tokenizer.json, train.bin, val.bin, meta.json")
    s.add_argument("--vocab-size", type=int, default=1024)
    s.add_argument("--val-fraction", type=float, default=0.1)
    s.add_argument("--special", nargs="*", help="special tokens, e.g. '<|endoftext|>'")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="train a MitsuLM")
    s.add_argument("--data", required=True, help="folder written by prepare")
    s.add_argument("--out", required=True, help="run folder")
    s.add_argument("--steps", type=int, default=None, help="total optimisation steps (default 2000)")
    s.add_argument("--resume", action="store_true", help="continue the run in --out exactly where it stopped")
    s.add_argument("--stop-after", type=int, default=None, help="run at most this many steps now, then save and exit (continue with --resume)")
    s.add_argument("--d-model", type=int, default=128)
    s.add_argument("--layers", type=int, default=4)
    s.add_argument("--heads", type=int, default=4)
    s.add_argument("--context", type=int, default=128)
    s.add_argument("--ffn-hidden", type=int, default=None)
    s.add_argument("--untied", action="store_true", help="separate input and output token tables")
    s.add_argument("--dropout", type=float, default=0.0)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--accumulate", type=int, default=1, help="average gradients over this many batches per step")
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--warmup", type=int, default=100)
    s.add_argument("--weight-decay", type=float, default=0.1)
    s.add_argument("--clip", type=float, default=1.0)
    s.add_argument("--eval-every", type=int, default=250)
    s.add_argument("--eval-windows", type=int, default=64, help="validation windows per evaluation")
    s.add_argument("--log-every", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    _backend_arguments(s)
    s.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off", help="mixed precision for the torch backend")
    s.add_argument("--compile", action="store_true", help="torch backend: compile the model with torch.compile")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("generate", help="sample text from a trained run")
    s.add_argument("--run", required=True)
    s.add_argument("--prompt", default="")
    s.add_argument("--tokens", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--top-p", type=float, default=None)
    s.add_argument("--seed", type=int, default=0)
    _backend_arguments(s)
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("info", help="describe a trained run")
    s.add_argument("--run", required=True)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("parity", help="measure how closely the PyTorch twin reproduces the numpy engine (needs PyTorch)")
    s.add_argument("--run", default=None, help="a run folder or model.npz to test trained weights; default is a small random model")
    s.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, mps")
    s.add_argument("--amp", nargs="*", default=["bf16"], choices=["bf16", "fp16"], help="autocast types to test as well")
    s.add_argument("--batch", type=int, default=2)
    s.add_argument("--tokens", type=int, default=32)
    s.add_argument("--vocab-size", type=int, default=257)
    s.add_argument("--context", type=int, default=48)
    s.add_argument("--d-model", type=int, default=64)
    s.add_argument("--layers", type=int, default=3)
    s.add_argument("--heads", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_parity)

    s = sub.add_parser("gpu-check", help="run every twin check on a device and print PASS or FAIL (needs PyTorch)")
    s.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, mps")
    s.add_argument("--amp", nargs="*", default=["bf16", "fp16"], choices=["bf16", "fp16"], help="mixed-precision modes to test")
    s.add_argument("--compile", action="store_true", help="also test torch.compile")
    s.add_argument("--no-speed", action="store_true", help="skip the timing section")
    s.add_argument("--out", default=None, help="write the full report to this JSON file")
    s.set_defaults(func=cmd_gpu_check)
    return p


def _backend_arguments(s):
    s.add_argument("--backend", choices=["numpy", "torch"], default="numpy", help="numpy (the reference) or its PyTorch twin")
    s.add_argument("--device", default="auto", help="torch backend: auto, cpu, cuda, cuda:1, mps")


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
