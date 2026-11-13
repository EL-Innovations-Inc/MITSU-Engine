# mitsu-engine

A neural-network and language-model engine built around the MITSU activation, written from
scratch in Python with numpy as its only dependency.

```
MITSU(u) = s (s − 1) / (s + 2),   s = softplus(u)
```

It is a small, complete framework: layers with hand-written backpropagation, MITSU and
MitsuGLU blocks, attention with rotary positions, a byte-pair tokenizer, disk-backed
datasets, optimisers, a resumable training loop, cached text generation, a command-line
tool, and a gradient checker that verifies every backward pass against finite differences.
It trains residual MITSU networks hundreds of layers deep and sub-word language models, on
a CPU.

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

## What it does, measured

All numbers below were produced by the scripts in this repository on a 2-core CPU.

| task | network | parameters | result | time |
|---|---|---|---|---|
| MNIST | `MLP` 784-256-256-10 | 269,322 | 97.78% test accuracy | 5 epochs, 10 s |
| MNIST | `DeepMitsuNet`, 16 residual blocks | 4,418,826 | 98.20% test accuracy | 5 epochs, 3 min |
| MNIST | `DeepMitsuNet`, 16 MitsuGLU blocks | 8,604,938 | 98.03% test accuracy | 5 epochs, 5.5 min |
| Fashion-MNIST | `DeepMitsuNet`, 16 residual blocks | 4,418,826 | 89.89% test accuracy | 5 epochs, 3 min |
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

This is a CPU engine. It is the right size for research on networks up to tens of millions
of parameters; it is not a substitute for a GPU framework at large-language-model scale.

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

## Design

Every layer is a `Module` with two methods:

```python
y = layer.forward(x)          # also remembers what backward will need
gx = layer.backward(gy)       # returns dL/dx and adds dL/dparameters into .grad
```

There is no automatic differentiation. Each backward pass is derived by hand, and
`gradcheck` compares it with finite differences. The tests run that check on every layer,
block and model, including attention with rotary positions and the tied output layer, so a
wrong derivative cannot slip in unnoticed.

| file | contents |
|---|---|
| `functional.py` | `mitsu`, `mitsu_grad`, softplus, sigmoid, SiLU, GELU, softmax: stateless maths |
| `module.py` | `Parameter`, `Module`, `Sequential`, `Residual` |
| `layers.py` | `Linear`, `Embedding`, `TiedHead`, `LayerNorm`, `RMSNorm`, `Dropout`, `Flatten` |
| `activations.py` | `MITSU` (with `c` and `gain`), `ReLU`, `GELU`, `SiLU`, `Tanh` |
| `blocks.py` | `FeedForward`, `MitsuGLU`, `MultiHeadSelfAttention` (rotary, cached), `TransformerBlock`, `mitsu_block` |
| `models.py` | `MLP`, `DeepMitsuNet`, `MitsuGPT`, `MitsuLM`, `LMConfig`, `sample_token` |
| `tokenizer.py` | `BPETokenizer` |
| `corpus.py` | `prepare_corpus`, `load_corpus`, token files, chunked reading |
| `init.py` | weight initialisation and the measured MITSU constants it relies on |
| `losses.py` | `CrossEntropy` (label smoothing, ignore index), `MSE`, `accuracy` |
| `optim.py` | `SGD`, `Adam`, `AdamW`, gradient clipping, `WarmupCosine`, `StepLR` |
| `data.py`, `datasets.py` | `DataLoader`, `Standardizer`, `CharTokenizer`, `TextBatcher`, MNIST and text loaders |
| `trainer.py` | `Trainer`: epochs or steps, accumulation, evaluation, checkpoints, exact resume; `evaluate_lm` |
| `io.py` | `save_checkpoint`, `load_checkpoint`, `read_meta` (model, optimiser state, metadata in one `.npz`) |
| `cli.py` | the `prepare`, `train`, `generate` and `info` commands |
| `gradcheck.py` | the finite-difference checker |

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

Subclass `Module`, hold trainable arrays in `Parameter`s, and write `forward` and
`backward`. Then let `gradcheck` tell you whether the derivative is right.

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
python3 benchmarks/depth_sweep.py                          # 4 to 256 blocks
python3 benchmarks/throughput.py                           # up to 67.7 million parameters
python3 benchmarks/generation.py                           # cached against recomputed generation
```

## Tests

```bash
pip install pytest
pytest -q          # 140 tests, about 12 seconds
```

They cover the MITSU formula against its definition; every backward pass against finite
differences; rotary positions, tied embeddings and the generation cache; the tokenizer's
lossless round trip; chunked encoding against whole-text encoding; the optimisers; exact
resume; the command line end to end; and learning itself, including 48-layer networks and
language models that memorise and regenerate a sequence.

## Limits

- CPU and numpy only. No GPU, no convolutions, no automatic differentiation. Stage 2 of the
  language-model plan is a PyTorch twin of `MitsuLM` that loads the same weights and must
  produce the same outputs; this engine is the reference it will be checked against.
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
