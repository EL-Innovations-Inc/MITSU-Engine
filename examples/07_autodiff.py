"""Automatic differentiation: differentiate the mother function directly, then build a
layer from it without deriving a gradient by hand.

    python3 examples/07_autodiff.py          # a few seconds, no data needed

MITSU is the mother function with one live seat: f(s, 1, 1) - 1. Part 3 keeps all three
seats live, so each output unit takes three inputs. Its gradient is derived by the engine.
"""
import numpy as np
import _path  # noqa: F401
import mitsu_engine as me
from mitsu_engine import autograd as ag


def mother(x, y, z):
    return (x ** (y + z) + y ** (x + z) + z ** (x + y)) / (x + y + z)


# 1. a gradient of the formula itself -------------------------------------------------------
x, y, z = (ag.Tensor(v, requires_grad=True) for v in (1.2, 0.7, 1.9))
f = mother(x, y, z)
f.backward()
print(f"f(1.2, 0.7, 1.9) = {f.item():.6f}")
print(f"df/dx = {x.grad:.6f}   df/dy = {y.grad:.6f}   df/dz = {z.grad:.6f}")

# 2. MITSU is f(s, 1, 1) - 1: the automatic derivative equals the hand-derived one --------------
u = np.linspace(-4, 4, 9)
t = ag.Tensor(u, requires_grad=True)
(mother(ag.softplus(t), 1.0, 1.0) - 1.0).sum().backward()
print(f"\nlargest difference from the hand-derived MITSU derivative: {np.abs(t.grad - me.functional.mitsu_grad(u)).max():.1e}")


# 3. a layer with three live seats, no hand-written backward ------------------------------------
class ThreeSeat(me.AutoModule):
    """Linear to 3 * units values, mapped to positive seats 2^tanh(.) in (1/2, 2), then the
    mother function on each triple: f(x, y, z) - 1. Three inputs in, one output out."""

    def __init__(self, d_in, units, rng=None):
        self.units = units
        self.weight = me.Parameter(me.init.linear_weight(d_in, 3 * units, rng=rng))
        self.bias = me.Parameter(np.zeros(3 * units), decay=False)

    def compute(self, x):
        seats = 2.0 ** ag.tanh(x @ self.weight + self.bias)
        seats = seats.reshape(*x.shape[:-1], self.units, 3)
        return mother(seats[..., 0], seats[..., 1], seats[..., 2]) - 1.0


me.set_dtype(np.float64)
layer = ThreeSeat(4, 5, rng=0)
print(f"gradient check of the three-seat layer against finite differences: {me.gradcheck(layer, np.random.default_rng(0).standard_normal((3, 4)))[0]:.1e}")
me.set_dtype(np.float32)

me.manual_seed(0)
X, y = me.make_spirals(300, classes=3)
Xtr, ytr, Xva, yva = me.train_val_split(X, y, val_fraction=0.2)
model = me.Sequential(ThreeSeat(2, 32), me.LayerNorm(32), ThreeSeat(32, 32), me.LayerNorm(32), me.Linear(32, 3))
print(f"\nthree-seat network: {model.num_parameters():,} parameters")
trainer = me.Trainer(model, me.AdamW(model.parameters(), lr=1e-2), clip_norm=1.0)
trainer.fit(me.DataLoader(Xtr, ytr, 64), me.DataLoader(Xva, yva, 256, shuffle=False), epochs=40)
