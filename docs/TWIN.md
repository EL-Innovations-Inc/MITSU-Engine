# The PyTorch twin

`mitsu_engine.twin` is `MitsuLM` written a second time, with PyTorch modules. The numpy
engine is the reference: its backward passes are written by hand and checked against finite
differences, and it has no dependency that could change underneath it. The twin exists so the
same model can run on a GPU and under mixed precision, and it is only trusted because the
tests load identical weights into both and require the same results.

Contents: [what is mirrored](#what-is-mirrored) · [what the tests require](#what-the-tests-require) ·
[moving things across](#moving-things-across) · [where the engines differ](#where-the-engines-differ) ·
[running on a GPU](#running-on-a-gpu) · [mixed precision](#mixed-precision) ·
[what was run, and where](#what-was-run-and-where) · [limits](#limits)

## What is mirrored

| piece | numpy engine | PyTorch twin |
|---|---|---|
| model | `me.MitsuLM` | `twin.TorchMitsuLM` (same `LMConfig`) |
| activation | `functional.mitsu` | `twin.mitsu` |
| feed-forward | `me.MitsuGLU` | `twin.MitsuGLU`, or `mitsu_torch.MitsuGLU` from the main MITSU repository |
| normalisation | `me.RMSNorm` | `twin.RMSNorm` |
| attention | `me.MultiHeadSelfAttention(rope=True)` | `twin.Attention` (`scaled_dot_product_attention`) |
| generation | `MitsuLM.generate`, `step` | `TorchMitsuLM.generate`, `step`: the same window policy and the engine's own sampler |
| optimiser | `me.AdamW` | `twin.EngineAdamW`: the same update rule |
| clipping | `me.clip_grad_norm` | `twin.clip_grad_norm`: the same rule |
| schedule | `me.WarmupCosine` | the same object |
| batches | `me.TextBatcher` | the same object |
| training loop | `me.Trainer` | `twin.TorchTrainer`, a subclass: only the optimisation step and the state file code differ |
| checkpoints | `MitsuLM.save` / `load` | `TorchMitsuLM.save` / `load`: the same file |
| command line | `--backend numpy` | `--backend torch` |

The initial weights of a `TorchMitsuLM` are drawn by the numpy engine, so
`TorchMitsuLM(cfg, rng=5)` and `MitsuLM(cfg, rng=5)` start from the same numbers, bit for bit.

## What the tests require

`tests/test_twin.py` (50 tests). Each row loads the same weights into both engines.

| check | required | measured on CPU |
|---|---|---|
| logits, float64 | largest difference below 1e-11 | 3e-15 (small model), 1e-14 (trained 918k model) |
| loss, float64 | below 1e-12 | 0 |
| every gradient, float64: hand-written backward against autograd | below 1e-10, relative to the parameter's largest entry | 4e-15 |
| cached generation path, float64 | below 1e-11 | 2e-15 |
| logits / loss / gradients, float32 | below 5e-4 / 5e-5 / 5e-3 | 1e-6 / 5e-7 / 3e-6 (small); 1e-5 / 2e-6 / 1e-6 (trained) |
| generated tokens: greedy, temperature, top-k, top-p, with and without the cache, past the context window | identical | identical |
| initial weights from the same seed | identical bits | identical |
| weights numpy → PyTorch → numpy | identical bits | identical |
| 30 training steps with warm-up, cosine decay, clipping, weight decay, gradient accumulation, float64 | weights within 1e-10, optimiser moments within 1e-12 | 7e-15, 1e-15 |
| 12 steps in one engine against 6 + state file + 6 in the other, both directions, float64 | weights within 1e-10 | passes |
| PyTorch run stopped and resumed, with dropout | identical bits | identical |
| training loss under dropout 0.3, 600 draws per engine | means within 4 standard errors, spreads within 15% | 0.5 standard errors, 1% (1,500 draws) |
| checkpoint written by either engine, read by either | identical arrays and keys | identical |
| training under `bf16` and `fp16` autocast | finite, loss falls below a quarter of its start | passes |
| MITSU in float16 and bfloat16 from −60,000 to 60,000 | finite value and gradient | passes |
| `mitsu_torch.MitsuGLU` from the main repository used as the feed-forward | logits within 1e-8 | passes (skipped if that package is not installed) |
| `import mitsu_engine` | does not import PyTorch | passes |

Longer runs (`benchmarks/twin_trajectory.py`), no dropout, same seed:

| run | precision | steps | largest weight difference at the end |
|---|---|---|---|
| small synthetic task, 115k parameters | float64 | 400 | 2e-15 |
| small synthetic task, 115k parameters | float32 | 400 | 1e-6 |
| Shakespeare, 918k parameters | float64 | 300 | 7e-10 |
| Shakespeare, 918k parameters | float32 | 600 | 0.08 (largest weight 1.74) |

The last row is expected and is not a disagreement between the engines. The two do the same
arithmetic in a different order, so they round differently; training amplifies a rounding
difference from step to step. On the Shakespeare model that amplification is a factor of
about 300,000 over 300 steps (float64: 2e-15 at the start, 7e-10 at the end). Starting from
float32's 1e-7 it reaches a visible size: the two training losses agree to 6e-8 at step 100,
4e-5 at step 200 and 3e-3 at step 600, where validation loss is 3.77203 (numpy) and 3.77196
(PyTorch). In float32 the two engines produce two equally valid runs of the same
experiment, not the same digits. The same happens between two different machines running
the same float32 PyTorch code.

## Moving things across

A parameter has the same name in both engines.

| parameter | numpy shape | PyTorch shape |
|---|---|---|
| `tok.weight` | (vocab, d_model) | the same |
| `blocks.N.norm1.gain`, `norm2.gain`, `norm.gain` | (d_model,) | the same |
| `blocks.N.attn.qkv.weight` | (d_model, 3·d_model) | transposed |
| `blocks.N.attn.proj.weight` | (d_model, d_model) | transposed |
| `blocks.N.ffn.w1.weight`, `w3.weight` | (d_model, hidden) | transposed |
| `blocks.N.ffn.w2.weight` | (hidden, d_model) | transposed |
| `head.weight` (only when embeddings are not tied) | (d_model, vocab) | transposed |

`torch.nn.Linear` stores a weight as (out, in), the engine as (in, out). With tied embeddings
PyTorch also shows the token table under the name `head.weight`; it is one tensor, and that
second name is never written to a file.

```python
net = twin.TorchMitsuLM.from_engine(reference, device="cuda")   # numpy model -> PyTorch
reference = net.to_engine()                                     # PyTorch model -> numpy
net.save("model.npz")                                           # the engine's file format
net = twin.TorchMitsuLM.load("model.npz", device="cuda")        # written by either engine
state = twin.engine_state(net)                                  # {name: numpy array}, engine layout
twin.load_engine_state(net, state)
```

A state file (`Trainer.save_state`, `TorchTrainer.save_state`, `state.npz` in a run folder)
holds the weights, the optimiser's step counter and moment arrays in the engine's parameter
order, the trainer's step and history, and the state of the numpy random generators that
choose the batches. The PyTorch trainer also stores PyTorch's generator state (dropout) and
the gradient scaler; the numpy engine ignores those entries. Either trainer loads either file.

## Where the engines differ

- **Dropout.** The two draw different random numbers. With dropout above zero they agree in
  evaluation mode, and a training run is the same experiment with different masks, not the
  same numbers. In the numpy engine dropout also draws from the generator that picks the
  batches, so the batch order differs as well. Use dropout 0 when you need the two to follow
  the same trajectory. A test checks that the two dropouts give the same distribution of
  losses. On the 918k-parameter Shakespeare run with dropout 0.1, one numpy run ended at
  validation loss 3.351 and three PyTorch runs (seeds 0 to 2) at 3.381, 3.353 and 3.393.
- **Softplus.** `torch.nn.functional.softplus` returns its input unchanged above 20, an error
  of up to 2e-9. `twin.softplus` has no such shortcut and matches the engine to rounding.
  (`mitsu_torch.MitsuGLU` uses the built-in one; that is why its row above allows 1e-8.)
- **AdamW.** `torch.optim.AdamW` adds `eps` after the bias correction, the engine before it.
  `twin.EngineAdamW` follows the engine. `torch.optim.AdamW` would train the model equally
  well, but its runs could not be continued by the numpy engine or compared step for step.
- **Attention kernel.** The twin calls `scaled_dot_product_attention`, which uses fused
  kernels on a GPU. The mathematics is the same; the order of operations is not.
- **Half precision.** Under autocast the twin computes MITSU, RMSNorm and the position
  rotation in float32 and casts back. The numpy engine has no half-precision mode.

## Running on a GPU

1. **Check the hardware first.** This compares the numpy engine on the CPU against the twin on
   the GPU, in float64 and float32, and the autocast loss against the float32 reference:

   ```bash
   pip install torch
   python -m mitsu_engine parity --device cuda --amp bf16 fp16
   ```

   It prints a table and ends with `PASS` or `FAIL` (exit code 0 or 1). Add `--run runs/book`
   to test trained weights instead of a small random model.
   Run it with PyTorch's default settings. Faster, less exact float32 modes (for example
   `torch.set_float32_matmul_precision("high")`, which allows TF32 on recent NVIDIA GPUs) can
   be expected to widen the float32 differences beyond the limits in the table; the float64
   row does not depend on them.

2. **Run the tests on that machine:** `pytest -q` (they use the CPU; the parity command above
   is the GPU check).

3. **Train.** The commands and the run folder are the same as for the numpy engine:

   ```bash
   python -m mitsu_engine prepare --text corpus.txt --out data/corpus --vocab-size 8192
   python -m mitsu_engine train --data data/corpus --out runs/a --backend torch --device cuda --amp bf16 \
       --d-model 512 --layers 8 --heads 8 --context 512 --batch 32 --steps 20000 --lr 6e-4 --warmup 500
   python -m mitsu_engine train --data data/corpus --out runs/a --backend torch --device cuda --amp bf16 --resume
   python -m mitsu_engine generate --run runs/a --backend torch --device cuda --prompt "Once"
   python -m mitsu_engine generate --run runs/a --prompt "Once"         # the same model on numpy, no GPU
   ```

   `--backend`, `--device`, `--amp` and `--compile` are not part of a run's identity: a run
   can be resumed with different values, including on the other engine.

4. **Speed and stability.** `python benchmarks/twin_speed.py --device cuda --amp bf16` times
   a training step. `log.jsonl` records the largest gradient norm between log lines, which is
   where instability shows first.

In Python:

```python
import mitsu_engine as me
from mitsu_engine import twin

tokenizer, train_tokens, val_tokens, meta = me.load_corpus("data/corpus")
config = me.LMConfig(vocab_size=meta["vocab_size"], context=512, d_model=512, n_layers=8, n_heads=8)
me.manual_seed(0)
model = twin.TorchMitsuLM(config).to("cuda")
train, val = me.TextBatcher(train_tokens, 512, 32), me.TextBatcher(val_tokens, 512, 32)
trainer = twin.TorchTrainer(model, twin.EngineAdamW(model, lr=6e-4, weight_decay=0.1), clip_norm=1.0, amp="bf16",
                            scheduler=me.WarmupCosine(6e-4, warmup=500, total=20000, min_lr=6e-5))
trainer.fit_steps(train, 20000, eval_fn=lambda: twin.evaluate_lm(model, val.windows(32, 64), "bf16"), eval_every=500,
                  log_every=50, state_path="state.npz", state_every=500, log_file="log.jsonl")
model.save("model.npz")
```

## Mixed precision

| `--amp` | what runs in half precision | gradient scaler |
|---|---|---|
| `off` | nothing | no |
| `bf16` | the matrix products and attention, in bfloat16 | not needed |
| `fp16` | the same, in float16 | yes (`torch.amp.GradScaler`) |

The weights, the optimiser and the loss stay in float32 in every mode.

MITSU is safe in half precision by construction: it divides before it multiplies, and it
computes in float32 when given float16 or bfloat16. Without that, float16 overflows at
`u ≥ 257`, because `s·(s − 1)` exceeds 65,504. This is the fix made in MITSU v1.0.1, and
the twin's tests check values and gradients from −60,000 to 60,000 in both half types.

That makes the activation safe, not the network. Inside MitsuGLU the product
`gate_mult · MITSU(W1 x) · W3 x` is formed in half precision under autocast, and in float16
it overflows once it passes 65,504, for example when both factors are near 150. bfloat16
has float32's range and does not have this limit, so prefer `bf16` where the hardware
supports it. With
`fp16`, the gradient scaler skips a step whose gradients are not finite, and a loss that is
not finite stops the run with an error rather than continuing silently.

## What was run, and where

Everything in this package was written and tested on a 2-core CPU with PyTorch 2.15. No GPU
was available.

| | CPU | GPU |
|---|---|---|
| all 50 twin tests | pass | not run |
| `parity` in float64, float32, `bf16`, `fp16` | pass | not run |
| training with `--amp off`, `bf16`, `fp16` | runs and learns | not run |
| `--compile` | runs (about 40 s of compilation on this CPU) | not run |
| state saved and restored, CUDA random generators | not applicable | written, not run |
| `--device mps` | not applicable | not run |

The code paths are the same on every device, with one exception: the twin stores and
restores the CUDA random generators in a state file, and that branch has never executed.
Exact resume with dropout is tested on the CPU only. On a GPU a resumed run continues from
the same weights, optimiser state and batch order; whether it reproduces an uninterrupted
run bit for bit also depends on PyTorch's own determinism settings.

So step 1 above is not a formality. `parity --device cuda` is the test this package could
not run on its own behalf.

## Limits

- One device. There is no multi-GPU or distributed training.
- Only `MitsuLM` has a twin. `MLP`, `DeepMitsuNet` and `MitsuGPT` are numpy only.
- The tokenizer and the batcher are the numpy engine's and run on the CPU, in the main
  process. For a small model on a fast GPU, drawing batches can become the bottleneck.
- Written against the PyTorch 2.3+ API (`torch.amp.GradScaler`), tested with 2.15 only.
- The comparison in this document is between two implementations of one model. It says
  nothing about MITSU against other activations; the pre-registered experiments in the main
  MITSU repository are where that is done.
