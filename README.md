# MITSU-Engine-v0.1.0

A neural-network engine built around the MITSU activation, written from scratch in Python
with numpy as its only dependency.

```
MITSU(u) = s (s − 1) / (s + 2),   s = softplus(u)
```

It is a small, complete framework: layers with hand-written backpropagation, MITSU and
MitsuGLU blocks, self-attention, optimisers, a training loop, checkpoints, and a gradient
checker that verifies every backward pass against finite differences. It trains residual
MITSU networks hundreds of layers deep and a character-level transformer, on a CPU.

## Quick start

```bash
pip install numpy
python3 examples/01_spirals.py        # a 32-layer network, no data needed, a few seconds
```

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

To install it as a package: `pip install .` from this folder.

## What it does, measured

All numbers below were produced by the scripts in this repository on a 2-core CPU.

| task | network | parameters | result | time |
|---|---|---|---|---|
| MNIST | `MLP` 784-256-256-10 | 269,322 | 97.78% test accuracy | 5 epochs, 10 s |
| MNIST | `DeepMitsuNet`, 16 residual blocks | 4,418,826 | 98.20% test accuracy | 5 epochs, 3 min |
| MNIST | `DeepMitsuNet`, 16 MitsuGLU blocks | 8,604,938 | 98.03% test accuracy | 5 epochs, 5.5 min |
| Fashion-MNIST | `DeepMitsuNet`, 16 residual blocks | 4,418,826 | 89.89% test accuracy | 5 epochs, 3 min |
| Tiny Shakespeare | `MitsuGPT`, 2 layers, width 64 | 111,488 | validation loss 1.76 nats | 2,000 steps, 2 min |

**Depth.** Spirals task, width 64, same settings at every depth (`benchmarks/depth_sweep.py`):

| network | 4 | 16 | 64 | 256 blocks |
|---|---|---|---|---|
| residual MITSU blocks | 100% | 100% | 100% | 100% |
| residual MitsuGLU blocks | 100% | 100% | 100% | 98.6% |

A plain stack of Linear → MITSU layers, with no skip paths or normalisation and given ten
times the learning rate and eight times the epochs, trains at 2, 4 and 8 hidden layers and
fails at 16. That is why the deep model is built from normalised residual blocks.

**Size.** One full training step, batch of 64 (`benchmarks/throughput.py`):

| network | parameters | time per step |
|---|---|---|
| `DeepMitsuNet` width 128, 16 blocks | 1.2 million | 17 ms |
| `DeepMitsuNet` width 256, 32 blocks | 8.6 million | 122 ms |
| `DeepMitsuNet` width 512, 64 blocks | 67.7 million | 0.91 s |

This is a CPU engine. It is the right size for research on networks up to tens of millions
of parameters; it is not a substitute for a GPU framework at language-model scale.

## Design

Every layer is a `Module` with two methods:

```python
y = layer.forward(x)          # also remembers what backward will need
gx = layer.backward(gy)       # returns dL/dx and adds dL/dparameters into .grad
```

There is no automatic differentiation. Each backward pass is derived by hand, and
`gradcheck` compares it with finite differences. The tests run that check on every layer,
block and model, so a wrong derivative cannot slip in unnoticed.

| file | contents |
|---|---|
| `functional.py` | `mitsu`, `mitsu_grad`, softplus, sigmoid, SiLU, GELU, softmax: stateless maths |
| `module.py` | `Parameter`, `Module`, `Sequential`, `Residual` |
| `layers.py` | `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `Flatten` |
| `activations.py` | `MITSU` (with `c` and `gain`), `ReLU`, `GELU`, `SiLU`, `Tanh` |
| `blocks.py` | `FeedForward`, `MitsuGLU`, `MultiHeadSelfAttention`, `TransformerBlock`, `mitsu_block` |
| `models.py` | `MLP`, `DeepMitsuNet`, `MitsuGPT` |
| `init.py` | weight initialisation and the measured MITSU constants it relies on |
| `losses.py` | `CrossEntropy` (label smoothing, ignore index), `MSE`, `accuracy` |
| `optim.py` | `SGD`, `Adam`, `AdamW`, gradient clipping, `WarmupCosine`, `StepLR` |
| `data.py`, `datasets.py` | `DataLoader`, `Standardizer`, `CharTokenizer`, `TextBatcher`, MNIST and text loaders |
| `trainer.py` | `Trainer`: epochs or steps, evaluation, schedules, best-model checkpoint, early stopping |
| `io.py` | `save_checkpoint`, `load_checkpoint` (model, optimiser state, metadata in one `.npz`) |
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
    me.Residual(me.Sequential(me.LayerNorm(512), me.MitsuGLU(512))),
    me.LayerNorm(512), me.Linear(512, 10),
)

# a language model
gpt = me.MitsuGPT(vocab_size=65, context=128, d_model=128, n_layers=4, n_heads=4)
print(gpt.summary())
text_ids = gpt.generate(prompt_ids, max_new_tokens=200, temperature=0.8, top_k=20)
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
python3 examples/03_char_gpt.py                            # transformer on Tiny Shakespeare
python3 benchmarks/depth_sweep.py                          # 4 to 256 blocks
python3 benchmarks/throughput.py                           # up to 67.7 million parameters
```

## Tests

```bash
pip install pytest
pytest -q          # 83 tests, about 10 seconds
```

They cover the MITSU formula against its definition, every backward pass against finite
differences, the optimisers, schedules, data tools, checkpoints, determinism, and
end-to-end learning, including 48-layer networks and a transformer that memorises and
regenerates a sequence.

## Limits

- CPU and numpy only. No GPU, no convolutions, no automatic differentiation.
- Very deep MitsuGLU stacks are sensitive to the learning rate: 128 blocks trained at
  1e-3 and failed at 2e-3. Keep it at or below 1e-3 with warm-up and gradient clipping.
- `GELU` uses the tanh approximation, because numpy has no error function.
- The accuracy figures above are single runs to show the engine works. They are not a
  comparison between MITSU and other activations; the pre-registered experiments in the
  main MITSU repository are where that is done.

## Licence

MIT. Evan David Little · EL Innovations Inc.
