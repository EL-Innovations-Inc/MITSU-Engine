"""Optimisers, gradient clipping and learning-rate schedules."""
import math
import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        self.steps = 0

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0)

    def step(self):
        raise NotImplementedError

    def state_dict(self):
        return {"steps": np.array(self.steps)}

    def load_state_dict(self, state):
        self.steps = int(state["steps"])


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum, Nesterov momentum and weight decay."""

    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum, self.nesterov, self.weight_decay = momentum, nesterov, weight_decay
        self.buf = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.steps += 1
        for p, b in zip(self.params, self.buf):
            g = p.grad
            if self.weight_decay and p.decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                b *= self.momentum
                b += g
                g = g + self.momentum * b if self.nesterov else b
            p.data -= self.lr * g

    def state_dict(self):
        state = super().state_dict()
        state.update({f"buf.{i}": b for i, b in enumerate(self.buf)})
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for i, b in enumerate(self.buf):
            b[...] = state[f"buf.{i}"]


class Adam(Optimizer):
    """Adam. `weight_decay` here is classic L2 (added to the gradient)."""

    decoupled = False

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.steps += 1
        t = self.steps
        step_size = self.lr * math.sqrt(1.0 - self.b2 ** t) / (1.0 - self.b1 ** t)
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            decay = self.weight_decay and p.decay
            if decay and not self.decoupled:
                g = g + self.weight_decay * p.data
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            if decay and self.decoupled:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= step_size * m / (np.sqrt(v) + self.eps)

    def state_dict(self):
        state = super().state_dict()
        for i, (m, v) in enumerate(zip(self.m, self.v)):
            state[f"m.{i}"], state[f"v.{i}"] = m, v
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for i, (m, v) in enumerate(zip(self.m, self.v)):
            m[...] = state[f"m.{i}"]
            v[...] = state[f"v.{i}"]


class AdamW(Adam):
    """Adam with decoupled weight decay, applied only to parameters marked `decay`
    (weight matrices, not biases, norms or embeddings)."""

    decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay)


def grad_norm(params):
    return math.sqrt(sum(float(np.vdot(p.grad, p.grad)) for p in params))


def clip_grad_norm(params, max_norm):
    """Scale all gradients down so their joint norm is at most max_norm. Returns the norm before clipping."""
    params = list(params)
    norm = grad_norm(params)
    if norm > max_norm:
        k = max_norm / (norm + 1e-12)
        for p in params:
            p.grad *= k
    return norm


# ---- learning-rate schedules: callables step -> learning rate ----------------

class ConstantLR:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, step):
        return self.lr


class WarmupCosine:
    """Linear warm-up from 0 to `lr` over `warmup` steps, then a cosine fall to `min_lr` at `total`."""

    def __init__(self, lr, warmup, total, min_lr=0.0):
        self.lr, self.warmup, self.total, self.min_lr = lr, warmup, total, min_lr

    def __call__(self, step):
        if step < self.warmup:
            return self.lr * (step + 1) / max(self.warmup, 1)
        frac = min(1.0, (step - self.warmup) / max(self.total - self.warmup, 1))
        return self.min_lr + 0.5 * (self.lr - self.min_lr) * (1.0 + math.cos(math.pi * frac))


class StepLR:
    """Multiply the learning rate by `gamma` every `every` steps."""

    def __init__(self, lr, every, gamma=0.1):
        self.lr, self.every, self.gamma = lr, every, gamma

    def __call__(self, step):
        return self.lr * self.gamma ** (step // self.every)
