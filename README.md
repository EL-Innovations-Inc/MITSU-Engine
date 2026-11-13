# mitsu-engine

A neural-network and language-model engine built around the MITSU activation, written from
scratch in Python with numpy as its only dependency, and a PyTorch twin of every model in
it, for GPUs, that is tested to reproduce it.

```
MITSU(u) = s (s − 1) / (s + 2),   s = softplus(u)
```

It is a small, complete framework: layers with hand-written backpropagation, MITSU and
MitsuGLU blocks, convolutions, attention with rotary positions, automatic differentiation,
a byte-pair tokenizer, disk-backed datasets, optimisers, a resumable training loop, cached
text generation, a command-line tool, and a gradient checker that verifies every backward
pass against finite differences. It trains residual MITSU networks hundreds of layers
deep, convolutional networks and sub-word language models, on a CPU.

The optional [PyTorch twin](#gpu-the-pytorch-twin) builds the same networks from PyTorch
modules, for GPUs and mixed precision. With identical weights the two agree to rounding
error on the outputs and every gradient, they follow the same training run, and they read
and write the same files. The twin has been verified on a CPU only; one command,
`python -m mitsu_engine gpu-check`, verifies it on your GPU.

## Quick start

```bash
pip install numpy
python3 examples/01_spirals.py        # a 32-layer network, no data needed, a few seconds
```

**A language model in four commands** (any UTF-8 text file works):

```bash
python -m mitsu_engine prepare  --text shakespeare.txt --out data/shake --vocab-size 1024
python -m mitsu_engine train    --data data/shake --out runs/shake --steps 3000 --lr 2e-3 --dropout 0.1
python -m mitsu_engine generate --run runs/shake --prompt "ROMEO:"
python -m mitsu_engine info     --run runs/shake
```

**A network in Python:**

```python
import mitsu_engine as me

X, y = me.make_spirals(600, classes=3)
Xtr, ytr, Xva, yva = me.train_val_split(X, y, val_fraction=0.2)

model = me.DeepMitsuNet(d_in=2, d_out=3, width=64, depth=32)      # 32 residual MITSU blocks
trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=1e-3), clip_norm=1.0)
trainer.fit(me.DataLoader(Xtr, ytr, batch_size=96),
            me.DataLoader(Xva, yva, batch_size=256, shuffle=False), epochs=15)

me.save_checkpoint("spirals.npz", model)
```

To install it as a package (this also adds a `mitsu-engine` command): `pip install .`
The PyTorch twin needs PyTorch as well: `pip install ".[torch]"`.

## What it does, measured

All numbers below were produced by the scripts in this repository on a 2-core CPU.

| task | network | parameters | result | time |
|---|---|---|---|---|
| MNIST | `MLP` 784-256-256-10 | 269,322 | 97.78% test accuracy | 5 epochs, 10 s |
| MNIST | `DeepMitsuNet`, 16 residual blocks | 4,418,826 | 98.20% test accuracy | 5 epochs, 3 min |
| MNIST | `DeepMitsuNet`, 16 MitsuGLU blocks | 8,604,938 | 98.03% test accuracy | 5 epochs, 5.5 min |
| Fashion-MNIST | `DeepMitsuNet`, 16 residual blocks | 4,418,826 | 89.89% test accuracy | 5 epochs, 3 min |
| MNIST | `MitsuConvNet`, 3 stages, 16-32-64 channels | 121,386 | 99.35% test accuracy | 2 epochs, 9.5 min |
| Fashion-MNIST | `MitsuConvNet`, 3 stages, 16-32-64 channels | 121,386 | 88.81% test accuracy | 2 epochs, 9.5 min |
| Tiny Shakespeare, characters | `MitsuGPT`, 2 layers, width 64 | 111,488 | validation loss 1.76 nats per character | 2,000 steps, 2 min |
| Tiny Shakespeare, 1,024 sub-word tokens | `MitsuLM`, 4 layers, width 128 | 918,144 | validation perplexity 28.6 (3.35 nats per token, 1.47 per byte) | 3,000 steps, 17 min |

**Depth.** Spirals task, width 64, same settings at every depth (`benchmarks/depth_sweep.py`):

| network | 4 | 16 | 64 | 256 blocks |
|---|---|---|---|---|
| residual MITSU blocks | 100% | 100% | 100% | 100% |
| residual MitsuGLU blocks | 100% | 100% | 100% | 98.6% |

A plain stack of Linear → MITSU layers, with no skip paths or normalisation and given ten
times the learning rate and eight times the epochs, trains at 2, 4 and 8 hidden layers and
fails at 16. That is why the deep models are built from normalised residual blocks.

**Size.** One full training step, batch of 64 (`benchmarks/throughput.py`):

| network | parameters | time per step |
|---|---|---|
| `DeepMitsuNet` width 128, 16 blocks | 1.2 million | 17 ms |
| `DeepMitsuNet` width 256, 32 blocks | 8.6 million | 122 ms |
| `DeepMitsuNet` width 512, 64 blocks | 67.7 million | 0.91 s |

**Generation.** Tokens per second with and without the cache (`benchmarks/generation.py`):

| model | cached | recomputing | speed-up |
|---|---|---|---|
| width 128, 4 layers, context 128 | 1,551 | 256 | 6.1× |
| width 256, 6 layers, context 256 | 452 | 35 | 12.9× |

The numpy engine is a CPU engine. It is the right size for research on networks up to tens
of millions of parameters. Beyond that, the [PyTorch twin](#gpu-the-pytorch-twin) runs the
same networks on a GPU.

## Language models

`MitsuLM` is a decoder-only transformer in the modern layout, with MITSU as the gate:

```
token embedding
n_layers ×  [ x + Attention(RMSNorm(x))        causal, rotary positions
              x + MitsuGLU(RMSNorm(x)) ]
RMSNorm → output layer, sharing the token table with the embedding
```

| piece | what it does |
|---|---|
| `BPETokenizer` | Byte-pair encoding trained from your text. The base vocabulary is the 256 byte values, so any text in any language encodes and decodes back exactly. Supports special tokens. Trains a 4,096-token vocabulary on 1 MB in about 6 seconds. |
| `prepare_corpus` | Trains the tokenizer on the training part only, then writes `train.bin` and `val.bin`. The text is read and encoded in pieces, so files larger than memory work, and the result is identical to encoding the whole text at once. |
| `open_tokens` + `TextBatcher` | Token files are memory-mapped: only the windows drawn for a batch are read from disk. |
| Rotary positions | Queries and keys are rotated by an angle that grows with position, so attention depends on distance. There is no position table. |
| Tied embeddings | The output layer reuses the token table. It is one parameter: stored, counted and updated once. |
| Cached generation | `generate` keeps each layer's keys and values, so a new token costs one incremental step. A test checks the cached logits against a full forward pass, and cached output against recomputed output. |
| Sampling | Temperature, top-k, top-p, and greedy decoding. |
| `LMConfig` | The full description of a model, stored inside every saved file: `MitsuLM.load(path)` rebuilds the network and loads the weights. |

In Python:

```python
import mitsu_engine as me

me.prepare_corpus("book.txt", "data/book", vocab_size=4096)
tokenizer, train_tokens, val_tokens, meta = me.load_corpus("data/book")

model = me.MitsuLM(me.LMConfig(vocab_size=meta["vocab_size"], context=256, d_model=256, n_layers=6, n_heads=8))
train = me.TextBatcher(train_tokens, context=256, batch_size=16)
val = me.TextBatcher(val_tokens, context=256, batch_size=16)

trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1), metric=None, clip_norm=1.0,
                     scheduler=me.WarmupCosine(1e-3, warmup=200, total=20000))
trainer.fit_steps(train, 20000, eval_fn=lambda: me.evaluate_lm(model, val.windows(16, 64)), eval_every=500,
                  accumulate=2, state_path="state.npz", state_every=500, log_file="log.jsonl")

model.save("model.npz")
print(tokenizer.decode(model.generate(tokenizer.encode("Once upon a time"), 200, temperature=0.8, top_k=40)[0]))
```

### Training that can be interrupted

- **Exact resume.** `save_state` stores the model, the optimiser, the step counter and the
  state of the random generators. A run that is stopped and resumed produces the same
  weights, bit for bit, as one that was never stopped. Tests check this for the Python API
  and for the command line.
- **Time-sliced runs.** `train --stop-after N` runs N steps, saves, and exits;
  `train --resume` continues. Useful on machines that are only available in slots.
- **Gradient accumulation.** `accumulate=k` averages the gradients of k batches per step:
  the effect of a batch k times as large in a k-th of the memory.
- **A log you can plot.** `log.jsonl` has one line per logged step with the training loss,
  the largest gradient norm since the last line, the learning rate, and validation loss
  and perplexity.

A sample from the 918,144-parameter model in the table above, prompted with `ROMEO:`

```
ROMEO:
Ay, like a case; and, by my creation.

ROMEO:
What, is they then?

Nurse:
Darry, then, my lord, and all my laps.

MERCUTIO:
A place of that,
That is your corse of mine mind depart
```

## Convolutions

```python
import mitsu_engine as me

Xtr, ytr, Xte, yte = me.load_mnist(flatten=False)                  # images as (N, 1, 28, 28)
model = me.MitsuConvNet(in_channels=1, n_classes=10, channels=(16, 32, 64), blocks=1)
trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01), clip_norm=1.0)
trainer.fit(me.DataLoader(Xtr, ytr, 64), epochs=2)                 # 99.35% on the test set
```

| piece | what it does |
|---|---|
| `Conv2d` | 2-D convolution with any kernel size, stride and padding. Images are (batch, channels, height, width); the weight is (out, in, kernel, kernel), the layout PyTorch uses. |
| `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d` | Pooling, including overlapping windows. |
| `GroupNorm` | Normalises groups of channels within each image. It keeps no batch statistics, so it behaves the same in training and evaluation and at any batch size. |
| `MitsuConvNet` | Residual convolutional network: stages of `x + Conv(MITSU(Conv(GroupNorm(x))))` blocks, halving the image between stages, then global pooling and a linear layer. Its branches are initialised by the same rule as the deep feed-forward networks, so depth does not change the signal size. |

A convolution gathers every input patch into one matrix once and is then a single matrix
product; the backward pass reuses the same matrix. Each backward pass is checked three
ways: against finite differences, against the engine's automatic differentiation, and
against `torch.nn.functional` (outputs agree to 1e-12 across kernel sizes, strides and
paddings).

On a 2-core CPU a 28×28 batch of 64 takes about 0.3 seconds, so MNIST is five minutes an
epoch. For larger images or networks, use the same model on a GPU through the twin.

## Automatic differentiation

`mitsu_engine.autograd` differentiates numpy code. The engine's own layers do not need it
(their backward passes are written by hand), but it removes that step for anything new.

```python
import numpy as np
import mitsu_engine as me
from mitsu_engine import autograd as ag

# 1. differentiate a formula: the mother function with all three seats live
x, y, z = (ag.Tensor(v, requires_grad=True) for v in (1.2, 0.7, 1.9))
f = (x ** (y + z) + y ** (x + z) + z ** (x + y)) / (x + y + z)
f.backward()
print(x.grad, y.grad, z.grad)

# 2. write a layer without deriving its gradient
class Affine(me.AutoModule):
    def __init__(self, d_in, d_out):
        self.weight = me.Parameter(me.init.linear_weight(d_in, d_out))
        self.bias = me.Parameter(np.zeros(d_out), decay=False)

    def compute(self, x):                       # Tensor in, Tensor out; Parameters used directly
        return ag.mitsu(x @ self.weight + self.bias)

net = me.Sequential(Affine(2, 32), Affine(32, 32), me.Linear(32, 3))     # mixes with hand-written layers
```

An `AutoModule` is an ordinary `Module`: it works in `Sequential` and `Residual`, with the
optimisers, the `Trainer`, checkpoints and `gradcheck`. Hand-written layers can also be
called inside `compute`.

| | |
|---|---|
| arithmetic | `+ − × ÷`, `**` with a number or a tensor exponent, `@`, broadcasting |
| functions | `exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `softplus`, `mitsu`, `relu`, `silu`, `gelu`, `abs`, `maximum`, `minimum`, `where` |
| reductions and shape | `sum`, `mean`, `max`, `reshape`, `transpose`, `swapaxes`, slicing, integer-array indexing, `concatenate`, `stack` |
| networks | `softmax`, `log_softmax`, `cross_entropy`, `mse`, `dropout`, `conv2d`, `max_pool2d`, `avg_pool2d` |
| helpers | `ag.grad(fn, *arrays)`, `ag.check(fn, *arrays)` (against finite differences) |

How it is verified: every operation is checked against finite differences; MITSU written
as the mother function `f(s, 1, 1) − 1` and differentiated automatically equals the
hand-derived derivative to 1e-16; and the whole of `MitsuLM`, rewritten with tensor
operations, gives the same gradient for every parameter as the hand-written backward pass,
to 1e-11. With PyTorch's autograd on the twin, that makes three independent derivations of
the same gradients.

It costs a little speed: measured on a 2-core CPU, a training step of the 918k-parameter
language model takes 26% longer with automatic differentiation than with the hand-written
backward pass, and an 8-layer MLP 8% longer. It computes first derivatives only, on the CPU.

## GPU: the PyTorch twin

`mitsu_engine.twin` builds any engine model a second time from PyTorch modules, so it can
run on a GPU and under mixed precision. The numpy engine stays the reference. The twin is
trusted only because the tests load the same weights into both and require the same
results. Nothing else in the package imports it: without PyTorch, everything above works
as before.

```python
import mitsu_engine as me
from mitsu_engine import twin

model = me.MitsuConvNet(1, 10)                              # any engine model: MLP, DeepMitsuNet,
net = twin.to_torch(model, device="cuda")                   # MitsuConvNet, MitsuGPT, MitsuLM, your own stack
print(twin.compare_models(model, net, images))              # the largest differences

trainer = twin.TorchTrainer(net, twin.EngineAdamW(net, lr=2e-3, weight_decay=0.01),
                            loss="cross_entropy", metric="accuracy", clip_norm=1.0, amp="bf16")
trainer.fit(me.DataLoader(Xtr, ytr, 64), epochs=2)          # the engine's loop, loaders, schedules and files
twin.to_engine(net, model)                                  # the trained weights back in numpy
```

**Check your GPU first.** Nothing in this package has been run on a GPU: it was written and
tested on a CPU, where every check below passes. This command runs the numpy engine on the
CPU as the reference and the twin on your device, and prints `PASS` or `FAIL` for each of
twelve checks (outputs and gradients of every kind of network, the same training run, each
mixed-precision mode, moving a training state between engines, resuming, generated text,
`torch.compile`, and speed):

```bash
pip install torch
python -m mitsu_engine gpu-check --device cuda --compile --out gpu_report.json
```

A check that raises an error is reported as failed with the error, and the rest still run.
`gpu_report.json` holds everything, including the device and library versions.

**Every network has a twin.** Largest difference between the engine and its PyTorch copy,
with the same weights, on a CPU (`twin.check_models`):

| network | outputs, float64 | gradients, float64 | outputs, float32 | gradients, float32 |
|---|---|---|---|---|
| `MLP` | 3e-17 | 2e-15 | 6e-8 | 5e-7 |
| `DeepMitsuNet`, both block types | 1e-15 | 2e-15 | 1e-6 | 1e-6 |
| `MitsuConvNet` | 2e-16 | 1e-15 | 3e-7 | 1e-6 |
| `MitsuGPT`, both feed-forward types | 5e-15 | 2e-15 | 3e-6 | 9e-7 |
| a stack using every other layer | 2e-16 | 5e-15 | 1e-7 | 2e-6 |
| attention without a mask, with bias | 2e-15 | 2e-15 | 1e-6 | 1e-6 |

"Gradients" compares the engine's hand-written backward pass with PyTorch's autograd, for
every parameter and for the input. A convolutional network and an MLP trained for two
epochs with the same seed end with the same weights to 1e-10 and the same log, row for row.

**Nothing left on the wrong device.** The usual way GPU code fails is a tensor (a mask, a
position table) created on the CPU by mistake. PyTorch's `meta` device refuses to mix with
CPU tensors, so the tests run every network's forward pass, backward pass, optimiser step
and generation cache there. This is a GPU-style check that runs without a GPU. It found
one weakness, since fixed: the tied output layer now reads its table from the embedding
on every call, so no device move can untie it.

**A layer of your own** (including an `AutoModule`) needs its PyTorch form named once:
`twin.register(MyLayer, lambda layer, convert: MyTorchLayer(...))`.

### The language model

`MitsuLM` has a dedicated twin, `TorchMitsuLM`, with cached generation:

```python
reference = me.MitsuLM(me.LMConfig(vocab_size=1024))        # numpy
net = twin.TorchMitsuLM.from_engine(reference)              # the same weights in PyTorch
print(twin.compare(reference, net))                         # the largest differences
```

**Same weights in, same results out.** Measured on a CPU by `python -m mitsu_engine parity`,
on a small random model and on the trained 918,144-parameter model from the table above
(each cell: small model · trained model):

| largest difference between the engines | float64 | float32 |
|---|---|---|
| logits | 3e-15 · 1e-14 | 1e-6 · 1e-5 |
| loss | 0 · 0 | 5e-7 · 2e-6 |
| any gradient (numpy's hand-written backward against PyTorch autograd) | 4e-15 · 2e-15 | 3e-6 · 1e-6 |
| logits along the cached generation path | 2e-15 · 1e-14 | 1e-6 · 7e-6 |
| generated tokens | identical | identical |

The tests also require:

- **The same text.** Greedy, temperature, top-k and top-p sampling return the same tokens
  from both engines, with and without the cache, including past the context window.
- **The same start.** `TorchMitsuLM(config, rng=5)` and `MitsuLM(config, rng=5)` hold the
  same initial weights, bit for bit: the numpy engine draws them for both.
- **The same training run.** `twin.EngineAdamW` and `twin.clip_grad_norm` follow the
  engine's rules, and the schedule and the batcher are the engine's own objects. After 30
  steps with warm-up, cosine decay, clipping, weight decay and gradient accumulation in
  float64, the two hold the same weights to 7e-15.
- **One file format.** A model or a training state written by either engine loads in the
  other. A run can start in numpy and finish in PyTorch, or the reverse, and ends with the
  weights an uninterrupted run would have (tested in float64, both directions).
- **Mixed precision.** The twin trains under `bf16` and `fp16` autocast. MITSU divides
  before it multiplies and computes in float32 when given a half-precision input, so the
  activation is finite from −60,000 to 60,000 in both half types. This is the fix from MITSU
  v1.0.1; it protects the activation, not the whole network (see `docs/GPU.md`).
- **The main repository's module drops in.** `twin.TorchMitsuLM(config, glu=mitsu_torch.MitsuGLU)`
  uses the `MitsuGLU` from the main MITSU repository unchanged.

**From the command line** the commands are the same, with a backend:

```bash
python -m mitsu_engine gpu-check --device cuda                         # run this first on a new machine
python -m mitsu_engine train    --data data/shake --out runs/gpu --backend torch --device cuda --amp bf16
python -m mitsu_engine train    --data data/shake --out runs/gpu --backend torch --device cuda --amp bf16 --resume
python -m mitsu_engine generate --run runs/gpu --backend torch --device cuda --prompt "ROMEO:"
python -m mitsu_engine generate --run runs/gpu --prompt "ROMEO:"       # the same model on numpy, no GPU
```

**What has and has not been run.** All of the above was run on a 2-core CPU. No GPU was
available, so nothing here has executed on one. `gpu-check --device cuda` is the check this
package could not make for itself. `docs/GPU.md` lists what was run where.

**Float32 runs drift apart, and that is expected.** With no dropout, the same seed gives
both engines the same weights and the same batches, and in float64 they stay together (a
918k-parameter model: weights within 7e-10 after 300 steps). In float32 the two round
differently and training amplifies the difference: on that model the training losses agree
to 6e-8 at step 100 and to 3e-3 at step 600, where validation loss is 3.77203 on numpy and
3.77196 on PyTorch. They are two valid runs of one experiment, not the same digits.

With dropout the engines also draw different masks, so runs differ the way runs with
different seeds do (a test checks that the two dropouts give the same loss distribution).
The stage-1 run above (918,144 parameters, dropout 0.1, 3,000 steps) ended at validation
loss 3.351 on numpy. The same command on the PyTorch backend ended at 3.381, 3.353 and
3.393 for seeds 0, 1 and 2 (perplexity 28.6 to 29.8, against 28.5). That is one numpy run
beside three PyTorch runs: consistent with seed-to-seed spread, and too few runs to say more.

**Speed on a CPU** (`benchmarks/twin_speed.py`, 2 threads, same weights and batch):

| model | parameters | numpy step | PyTorch step | | numpy generation | PyTorch generation |
|---|---|---|---|---|---|---|
| width 128, 4 layers, context 128 | 1,311,360 | 370 ms | 165 ms | 2.2× | 911 tokens/s | 844 tokens/s |
| width 256, 6 layers, context 256 | 5,772,032 | 1,344 ms | 467 ms | 2.9× | 459 tokens/s | 397 tokens/s |
| width 512, 8 layers, context 256 | 27,267,584 | 2,057 ms | 974 ms | 2.1× | 146 tokens/s | 148 tokens/s |

On a CPU the twin trains two to three times faster and generates at about the same speed.
Convolutions gain more: the MNIST run above takes 40 seconds an epoch through the twin on the
same CPU, against 260 in numpy, and reaches the same 99.35% (`examples/06_mnist_conv.py
--backend torch --device cpu`).
The reason to have it is the GPU.

## Design

Every layer is a `Module` with two methods:

```python
y = layer.forward(x)          # also remembers what backward will need
gx = layer.backward(gy)       # returns dL/dx and adds dL/dparameters into .grad
```

Each built-in backward pass is derived by hand, and `gradcheck` compares it with finite
differences. The tests run that check on every layer, block and model, including
convolutions, attention with rotary positions and the tied output layer, so a wrong
derivative cannot slip in unnoticed. A layer of your own can skip the derivation: an
`AutoModule` has its backward pass derived by `autograd`.

| file | contents |
|---|---|
| `functional.py` | `mitsu`, `mitsu_grad`, softplus, sigmoid, SiLU, GELU, softmax: stateless maths |
| `module.py` | `Parameter`, `Module`, `Sequential`, `Residual` |
| `layers.py` | `Linear`, `Embedding`, `TiedHead`, `LayerNorm`, `RMSNorm`, `Dropout`, `Flatten` |
| `activations.py` | `MITSU` (with `c` and `gain`), `ReLU`, `GELU`, `SiLU`, `Tanh` |
| `blocks.py` | `FeedForward`, `MitsuGLU`, `MultiHeadSelfAttention` (rotary, cached), `TransformerBlock`, `mitsu_block` |
| `conv.py` | `Conv2d`, `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`, `GroupNorm` |
| `autograd.py` | `Tensor`, the differentiable operations, `AutoModule`, `grad`, `check` |
| `models.py` | `MLP`, `DeepMitsuNet`, `MitsuConvNet`, `MitsuGPT`, `MitsuLM`, `LMConfig`, `sample_token` |
| `tokenizer.py` | `BPETokenizer` |
| `corpus.py` | `prepare_corpus`, `load_corpus`, token files, chunked reading |
| `init.py` | weight initialisation and the measured MITSU constants it relies on |
| `losses.py` | `CrossEntropy` (label smoothing, ignore index), `MSE`, `accuracy` |
| `optim.py` | `SGD`, `Adam`, `AdamW`, gradient clipping, `WarmupCosine`, `StepLR` |
| `data.py`, `datasets.py` | `DataLoader`, `Standardizer`, `CharTokenizer`, `TextBatcher`, MNIST and text loaders |
| `trainer.py` | `Trainer`: epochs or steps, accumulation, evaluation, checkpoints, exact resume; `evaluate_lm` |
| `io.py` | `save_checkpoint`, `load_checkpoint`, `read_meta` (model, optimiser state, metadata in one `.npz`) |
| `cli.py` | the `prepare`, `train`, `generate`, `info`, `parity` and `gpu-check` commands |
| `gradcheck.py` | the finite-difference checker |
| `twin/` | the PyTorch copies (needs PyTorch): `to_torch`, `TorchMitsuLM`, `EngineAdamW`, `TorchTrainer`, parity checks, `gpucheck` |

### How it handles MITSU

MITSU is a quiet activation: for a standard-normal input its output has a standard
deviation of 0.184, about a third of SiLU's, and a silent input gives −0.079, not 0. The
engine builds that into its defaults:

- **The layer after an activation is scaled to compensate.** `init.activation_gain` computes
  the exact factor for any activation by numerical integration (5.43 for MITSU), so each
  residual branch starts out handing on a signal of known size.
- **Residual branches shrink with depth.** Each of `depth` branches is initialised to add
  variance 1/`depth`, so the signal stays level through hundreds of blocks. A test checks
  this at 200 blocks.
- **`MitsuGLU` uses a gate gain of 3.04** by default, the measured ratio of SiLU's signal to
  MITSU's. Set `gate_mult=1.0` for the raw formula.
- **`MITSU(c=...)`** exposes the seat-weight family `s(s−1)/(s+c)`; `c = 2` is MITSU.
- **Numerically safe.** The formula divides before it multiplies and the derivative avoids
  squaring, so neither overflows, including in half precision.

## Building things

```python
# a plain network
net = me.MLP([784, 256, 256, 10])

# your own stack
net = me.Sequential(
    me.Linear(784, 512), me.MITSU(),
    me.Residual(me.Sequential(me.RMSNorm(512), me.MitsuGLU(512))),
    me.LayerNorm(512), me.Linear(512, 10),
)
print(net.summary())
```

### Writing a new layer

The short way is an `AutoModule`: write `compute` and the gradient is derived for you (see
[Automatic differentiation](#automatic-differentiation)). The fast way is to subclass
`Module`, hold trainable arrays in `Parameter`s, and write `forward` and `backward`
yourself. Then let `gradcheck` tell you whether the derivative is right.

```python
import numpy as np
import mitsu_engine as me

class Scale(me.Module):
    """y = x * exp(log_scale): one learned positive gain per feature."""

    def __init__(self, dim):
        self.log_scale = me.Parameter(np.zeros(dim), decay=False)
        self._c_x = None

    def forward(self, x):
        if me.grad_enabled():
            self._c_x = x
        return x * np.exp(self.log_scale.data)

    def backward(self, grad):
        s = np.exp(self.log_scale.data)
        self.log_scale.grad += (grad * self._c_x * s).reshape(-1, s.size).sum(0)
        return grad * s

me.set_dtype(np.float64)                      # gradient checks need double precision
error, where = me.gradcheck(me.Sequential(me.Linear(6, 6), Scale(6), me.MITSU()),
                            np.random.default_rng(0).standard_normal((4, 6)))
print(error)                                  # about 1e-9: the backward pass is correct
```

Parameters are discovered automatically from attributes, including lists of modules, so the
new layer works with the optimisers, `summary()`, and checkpoints with no registration.

## Examples

```bash
python3 examples/01_spirals.py                             # 32-layer network on a toy task
python3 examples/02_mnist.py                               # MLP on MNIST (downloads 11 MB)
python3 examples/02_mnist.py --model deep --depth 16       # 16 residual blocks
python3 examples/02_mnist.py --model deep --block glu      # MitsuGLU blocks
python3 examples/02_mnist.py --fashion                     # Fashion-MNIST
python3 examples/03_char_gpt.py                            # character-level transformer
python3 examples/04_language_model.py                      # tokenizer + MitsuLM, start to finish
python3 examples/05_torch_twin.py                          # numpy and PyTorch side by side (needs PyTorch)
python3 examples/06_mnist_conv.py                          # convolutional network on MNIST, 99.35%
python3 examples/06_mnist_conv.py --backend torch --device cuda    # the same network on a GPU
python3 examples/07_autodiff.py                            # differentiate the mother function; a three-seat layer
python3 benchmarks/depth_sweep.py                          # 4 to 256 blocks
python3 benchmarks/throughput.py                           # up to 67.7 million parameters
python3 benchmarks/generation.py                           # cached against recomputed generation
python3 benchmarks/twin_speed.py                           # numpy against PyTorch, step and generation speed
python3 benchmarks/twin_trajectory.py                      # how long the two engines stay together
```

## Tests

```bash
pip install pytest
pytest -q          # 239 tests in about 12 seconds; 330 in about 40 with PyTorch installed
```

They cover the MITSU formula against its definition; every backward pass against finite
differences; rotary positions, tied embeddings and the generation cache; the tokenizer's
lossless round trip; chunked encoding against whole-text encoding; the optimisers; exact
resume; the command line end to end; and learning itself, including 48-layer networks and
language models that memorise and regenerate a sequence. They also cover every automatic
differentiation operation, the convolution layers, and the whole language model
differentiated two ways. With PyTorch installed, 92 more tests check the twin against the
engine, including the device check itself and the device-discipline tests; without it they
are skipped. One uses the `mitsu_torch` package from the main MITSU repository and is
skipped if that is not installed.

## Limits

- **The GPU path has not been run on a GPU.** It is the PyTorch twin, tested on a CPU, with
  device-discipline tests that need no GPU. `python -m mitsu_engine gpu-check` is what
  confirms it on real hardware. It runs on one device; there is no multi-GPU training.
- The numpy engine is CPU only. Convolutions are practical at MNIST scale (five minutes an
  epoch), not at ImageNet scale.
- Automatic differentiation gives first derivatives only and runs on the CPU. An
  `AutoModule` needs its PyTorch form registered before `twin.to_torch` can mirror it.
- Convolutions are 2-D with square kernels; pooling has no padding; there is no batch
  normalisation (group normalisation takes its place).
- Very deep MitsuGLU stacks are sensitive to the learning rate: 128 blocks trained at
  1e-3 and failed at 2e-3. Keep it at or below 1e-3 with warm-up and gradient clipping.
- The tokenizer is pure Python. Encoding runs at about 4 MB of text per second and training
  is fast on megabytes; a multi-gigabyte corpus would want a compiled tokenizer.
- `GELU` uses the tanh approximation, because numpy has no error function.
- The figures above are single runs to show the engine works. They are not a comparison
  between MITSU and other activations; the pre-registered experiments in the main MITSU
  repository are where that is done.

## Licence

MIT. Evan David Little · EL Innovations Inc.
