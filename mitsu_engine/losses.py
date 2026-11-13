"""Loss functions. Each returns (loss value, gradient with respect to the prediction)."""
import numpy as np
from . import functional as F


class CrossEntropy:
    """Softmax cross-entropy for integer class targets, averaged over all positions.

    logits: (..., classes); targets: (...) integers. Positions whose target equals
    `ignore_index` contribute nothing. `label_smoothing` spreads that much probability
    evenly over the classes."""

    def __init__(self, label_smoothing=0.0, ignore_index=None):
        self.label_smoothing, self.ignore_index = label_smoothing, ignore_index

    def __call__(self, logits, targets):
        C = logits.shape[-1]
        z = logits.reshape(-1, C)
        t = np.asarray(targets).reshape(-1)
        logp = F.log_softmax(z, -1)
        keep = np.ones(len(t), dtype=bool) if self.ignore_index is None else (t != self.ignore_index)
        n = max(int(keep.sum()), 1)
        tt = np.where(keep, t, 0)
        rows = np.arange(len(t))
        q = np.zeros_like(logp)
        q[rows, tt] = 1.0
        if self.label_smoothing:
            q = q * (1.0 - self.label_smoothing) + self.label_smoothing / C
        q *= keep[:, None]
        loss = float(-(q * logp).sum() / n)
        grad = (np.exp(logp) * keep[:, None] - q) / n
        return loss, grad.reshape(logits.shape).astype(logits.dtype, copy=False)


class MSE:
    """Mean squared error over every element."""

    def __call__(self, pred, target):
        diff = pred - target
        return float((diff * diff).mean()), (2.0 / diff.size) * diff


def accuracy(logits, targets):
    """Fraction of positions where the largest logit is the target class."""
    return float((logits.argmax(-1) == np.asarray(targets)).mean())
