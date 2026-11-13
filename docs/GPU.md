# Running on a GPU: the PyTorch twin

`mitsu_engine.twin` builds the engine's networks a second time, from PyTorch modules. The
numpy engine is the reference: its backward passes are written by hand and checked against
finite differences, and it has no dependency that could change underneath it. The twin
exists so the same networks can run on a GPU and under mixed precision, and it is only
trusted because the tests load identical weights into both and require the same results.

**Nothing here has been run on a GPU.** The package was written and tested on a 2-core CPU.
Start with [the device check](#the-device-check): it is one command, and it is the
verification this package could not do for itself.

Contents: [the device check](#the-device-check) · [any model: to_torch](#any-model-to_torch) ·
[what is mirrored](#what-is-mirrored) · [what the tests require](#what-the-tests-require) ·
[moving things across](#moving-things-across) · [where the engines differ](#where-the-engines-differ) ·
[running on a GPU](#running-on-a-gpu) · [mixed precision](#mixed-precision) ·
[what was run, and where](#what-was-run-and-where) · [limits](#limits)

## The device check

```bash
pip install torch
python -m mitsu_engine gpu-check --device cuda --compile --out gpu_report.json
```

It runs the numpy engine on the CPU as the reference and the twin on the device, and prints
one line per check, then `PASS` or `FAIL` (exit code 0 or 1):

| check | what must hold |
|---|---|
| language model: outputs, gradients, generation path | float64 and float32 within the limits in [what the tests require](#what-the-tests-require); autocast loss within 0.05 |
| every network: outputs and gradients | each of 8 networks, in float64 and float32, matches the engine (outputs, every parameter gradient, the input gradient) |
| language model follows the numpy training run | 20 steps in float64 with warm-up, clipping and weight decay: weights within 1e-9 |
| conv network follows the numpy training run | 10 steps in float64: weights within 1e-9 |
| training learns, mixed precision off / bf16 / fp16 | a language model memorises a sequence (loss below a quarter of its start) and a conv network reaches 90% on a small task; nothing becomes infinite |
| a training state moves from the device to numpy | 6 steps on the device, a state file, 6 steps in numpy equal 12 steps in numpy, to 1e-9 |
| an interrupted run resumes on the device | 4 steps, a state file, 4 more against 8 uninterrupted, with dropout: identical on a CPU, within 1e-3 on a GPU |
| generation returns the same tokens as numpy | greedy, temperature, top-k and top-p, with and without the cache, past the context window |
| torch.compile gives the same output | with `--compile` |
| speed | milliseconds per training step in each precision mode, tokens per second, peak memory |

A check that raises an error is reported as failed, with the error, and the others still
run. `--out` writes everything to a JSON file, including the GPU model and library
versions and the traceback of anything that failed; that file is the thing to keep.

On the CPU all twelve pass. The checks run the same code on every device, so this
confirms the checks themselves; only running them on a GPU confirms the GPU. Float64 is not
available on Apple's `mps` device, so the float64 checks are skipped there.

If something fails on your GPU, the float64 rows are the ones that matter most: a float64
failure is a real disagreement. A float32 row slightly outside its limit, with float64
passing, means the hardware rounds more loosely than the limits assume (see the note on
TF32 below), not that the model is wrong.

## Any model: to_torch

```python
import mitsu_engine as me
from mitsu_engine import twin

model = me.MitsuConvNet(1, 10, channels=(16, 32, 64))
net = twin.to_torch(model, device="cuda")              # same structure, same parameter names, same weights

trainer = twin.TorchTrainer(net, twin.EngineAdamW(net, lr=2e-3, weight_decay=0.01), loss="cross_entropy",
                            metric="accuracy", clip_norm=1.0, amp="bf16",
                            scheduler=me.WarmupCosine(2e-3, 100, 2000))
trainer.fit(me.DataLoader(Xtr, ytr, 64), me.DataLoader(Xva, yva, 500, shuffle=False), epochs=2, checkpoint="best.npz")

twin.to_engine(net, model)                             # the trained weights back in the numpy model
print(twin.compare_models(model, net, Xva[:8]))        # largest differences in outputs and gradients
```

`to_torch` walks the engine model and builds the same tree from PyTorch modules with the
same attribute names, so a parameter has the same name on both sides. It works for `MLP`,
`DeepMitsuNet`, `MitsuConvNet`, `MitsuGPT`, `MitsuLM` (which becomes a `TorchMitsuLM`) and
any `Sequential` or `Residual` stack of the engine's layers. The result carries
`engine_layout`: the engine's parameter order with, for each parameter, whether its layout
is transposed and whether weight decay applies. The optimiser, the checkpoint functions
and the parity check all work from that list.

`TorchTrainer` is the engine's `Trainer` with the optimisation step replaced: `fit`,
`fit_steps`, early stopping, schedules, logs and state files are the engine's own code.
`loss` is `"cross_entropy"`, `"mse"` or a function; `metric` is `None`, `"accuracy"` or a
function.

A layer of your own, including an `AutoModule`, has no PyTorch form until you name one:

```python
class TorchScale(torch.nn.Module):                     # the same parameter names as the engine layer
    def __init__(self, dim):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return twin.mitsu(x) * self.factor

twin.register(Scale, lambda layer, convert: TorchScale(layer.factor.data.size))
```

After that, `twin.compare_models` tells you whether the two forms agree.

## What is mirrored

| piece | numpy engine | PyTorch twin |
|---|---|---|
| any model | `me.MLP`, `me.DeepMitsuNet`, `me.MitsuConvNet`, `me.MitsuGPT`, stacks | `twin.to_torch(model)` |
| convolution, pooling, group norm | `me.Conv2d`, `me.MaxPool2d`, `me.AvgPool2d`, `me.GroupNorm` | `torch.nn.Conv2d`, `MaxPool2d`, `AvgPool2d`, `group_norm` |
| language model | `me.MitsuLM` | `twin.TorchMitsuLM` (same `LMConfig`) |
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

`tests/test_twin.py` (50 tests, the language model) and `tests/test_twin_models.py` (42
tests, every other network, training, device discipline and the device check). Each row
loads the same weights into both engines.

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
| every network in `twin.model_zoo`: outputs, parameter gradients, input gradient, float64 | below 1e-11, 1e-10, 1e-10 | 5e-15 or less |
| the same, float32 | below 5e-4, 5e-3, 5e-3 | 3e-6 or less |
| the numpy convolution, pooling and group-norm kernels against `torch.nn.functional` | 1e-12 | passes |
| a conv network and an MLP trained two epochs with the same seed, float64 | weights within 1e-10, the same log row for row | passes |
| a conv network's state saved by PyTorch and continued by numpy | weights within 1e-10 | passes |
| forward, backward, optimiser step and generation cache on the `meta` device | no tensor left on the CPU | passes |
| the device check on the CPU; a check that raises is isolated | all pass; the others still run | passes |

**Device discipline without a GPU.** The usual way GPU code fails is a tensor (a mask, a
position table, a constant) created on the CPU by mistake. PyTorch's `meta` device holds
shapes and no data and refuses to mix with CPU tensors, so running a network there fails
on exactly that mistake. The tests run every network's forward pass, backward pass and
optimiser step on it, and the language model's generation cache. It cannot check numbers,
only placement. It found one weakness in 0.3.0: tied embeddings relied on two modules
sharing one Parameter object, which a device move is allowed to break. The output layer
now reads the table from the embedding on every call.

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
| a convolution kernel (`Conv2d.weight`) | (out, in, kernel, kernel) | the same |

`torch.nn.Linear` stores a weight as (out, in), the engine as (in, out). With tied embeddings
there is one table, `tok.weight`, on both sides; the PyTorch output layer owns no parameter.

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

1. **Check the hardware first** with [the device check](#the-device-check). To test trained
   weights of a language model as well:

   ```bash
   python -m mitsu_engine parity --run runs/book --device cuda --amp bf16 fp16
   ```
   Run it with PyTorch's default settings. Faster, less exact float32 modes (for example
   `torch.set_float32_matmul_precision("high")`, which allows TF32 on recent NVIDIA GPUs) can
   be expected to widen the float32 differences beyond the limits in the table; the float64
   row does not depend on them.

2. **Run the tests on that machine:** `pytest -q` (they use the CPU; the device check is
   the GPU test).

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
| all 92 twin tests | pass | not run |
| `gpu-check`, all twelve checks | pass | not run |
| device placement of every tensor (`meta` device) | pass | this is the part of GPU behaviour that could be tested here |
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

So the device check is not a formality. It is the test this package could not run on its
own behalf.

## Limits

- One device. There is no multi-GPU or distributed training.
- Cached generation exists for `MitsuLM` only. `MitsuGPT`'s twin has the forward pass; sample
  from it with the numpy model after `twin.to_engine`.
- An `AutoModule` or other layer of your own must be registered before it can be mirrored.
- The tokenizer and the batcher are the numpy engine's and run on the CPU, in the main
  process. For a small model on a fast GPU, drawing batches can become the bottleneck.
- Written against the PyTorch 2.3+ API (`torch.amp.GradScaler`), tested with 2.15 only.
- The comparison in this document is between two implementations of one model. It says
  nothing about MITSU against other activations; the pre-registered experiments in the main
  MITSU repository are where that is done.
