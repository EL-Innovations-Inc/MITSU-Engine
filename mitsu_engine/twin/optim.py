"""The engine's AdamW and gradient clipping, for PyTorch parameters.

torch.optim.AdamW would also train the model, but it is not the same rule: it adds eps
after the bias correction, the engine before it (the form in the original Adam paper).
EngineAdamW reproduces the engine's arithmetic, so the two engines follow the same
training trajectory and optimiser state can be saved by one and continued by the other.
"""
import math
import numpy as np
import torch
from .convert import layout, _to_engine, _to_torch


class EngineAdamW(torch.optim.Optimizer):
    """AdamW with the engine's update rule, for a TorchMitsuLM or any model from twin.to_torch.

        m <- b1 m + (1 - b1) g
        v <- b2 v + (1 - b2) g^2
        p <- p - lr * weight_decay * p                     weight matrices only
        p <- p - lr * sqrt(1 - b2^t) / (1 - b1^t) * m / (sqrt(v) + eps)

    Weight decay applies to the parameters the engine marks for it (weight matrices and
    convolution kernels, not embeddings, biases or norm gains). `lr` can be read and
    assigned, which is how schedules drive it.
    """

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self._layout = list(layout(model))                # engine order: (name, parameter, transposed, decay)
        self._decay = {id(p): decay for _, p, _, decay in self._layout}
        super().__init__([p for _, p, _, _ in self._layout], dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.steps = 0

    @property
    def lr(self):
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value):
        for group in self.param_groups:
            group["lr"] = value

    def _moments(self, p):
        state = self.state[p]
        if not state:
            state["m"], state["v"] = torch.zeros_like(p), torch.zeros_like(p)
        return state["m"], state["v"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.steps += 1
        t = self.steps
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            step_size = lr * math.sqrt(1.0 - b2 ** t) / (1.0 - b1 ** t)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                m, v = self._moments(p)
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                if wd and self._decay[id(p)]:
                    p.add_(p, alpha=-lr * wd)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
        return loss

    # ---- state in the engine's format -----------------------------------
    def engine_state(self):
        """The optimiser state as the engine's AdamW.state_dict() would hold it: the step
        counter and one first- and second-moment array per parameter, in the engine's order
        and layout."""
        state = {"steps": np.array(self.steps)}
        for i, (_, p, transposed, _) in enumerate(self._layout):
            m, v = self._moments(p)
            state[f"m.{i}"], state[f"v.{i}"] = _to_engine(m, transposed), _to_engine(v, transposed)
        return state

    @torch.no_grad()
    def load_engine_state(self, state):
        """Restore a state saved by this class or by the engine's AdamW."""
        self.steps = int(state["steps"])
        for i, (name, p, transposed, _) in enumerate(self._layout):
            for key, buf in zip((f"m.{i}", f"v.{i}"), self._moments(p)):
                value = _to_torch(state[key], transposed)
                if value.shape != tuple(buf.shape):
                    raise ValueError(f"optimiser state {key} ({name}) has shape {value.shape}, expected {tuple(buf.shape)}")
                buf.copy_(torch.from_numpy(value).to(device=buf.device, dtype=buf.dtype))


def _grads(params):
    return [p.grad for p in params if p.grad is not None]


def grad_norm(params):
    """Joint norm of all gradients, as a Python float."""
    grads = _grads(params)
    if not grads:
        return 0.0
    return math.sqrt(float(torch.stack([(g * g).sum() for g in grads]).sum()))


@torch.no_grad()
def clip_grad_norm(params, max_norm):
    """Scale all gradients down so their joint norm is at most max_norm, by the engine's
    rule (factor max_norm / (norm + 1e-12), applied only when norm > max_norm). Returns the
    norm before clipping."""
    params = list(params)
    norm = grad_norm(params)
    if norm > max_norm:
        k = max_norm / (norm + 1e-12)
        for g in _grads(params):
            g.mul_(k)
    return norm
