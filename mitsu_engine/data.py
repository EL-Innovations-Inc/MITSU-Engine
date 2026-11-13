"""Data utilities: batching, splitting, standardising, text, and a synthetic dataset."""
import numpy as np
from . import config


class DataLoader:
    """Iterate over (X, y) in mini-batches. Reshuffles every epoch when shuffle=True."""

    def __init__(self, X, y, batch_size=128, shuffle=True, drop_last=False, rng=None):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X, self.y = X, y
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.rng = config.get_rng(rng)

    def __len__(self):
        n = len(self.X)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.X)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        stop = n - n % self.batch_size if self.drop_last else n
        for i in range(0, stop, self.batch_size):
            idx = order[i:i + self.batch_size]
            yield self.X[idx], self.y[idx]


def train_val_split(X, y, val_fraction=0.1, rng=None):
    order = config.get_rng(rng).permutation(len(X))
    n_val = int(round(len(X) * val_fraction))
    return X[order[n_val:]], y[order[n_val:]], X[order[:n_val]], y[order[:n_val]]


class Standardizer:
    """Zero-mean, unit-variance features, fitted on training data. Features that are
    constant in the training data are left centred and unscaled, so an unseen value in
    such a feature cannot turn into an enormous input."""

    def fit(self, X):
        self.mean = X.mean(0)
        std = X.std(0)
        self.std = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(config.get_dtype())

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class CharTokenizer:
    """Character-level tokenizer built from a text."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.chars[int(i)] for i in np.asarray(ids).reshape(-1))


class TextBatcher:
    """Random (input, next-token target) windows from a token array, for language modelling.
    `tokens` may be an in-memory array or a memory-mapped token file (corpus.open_tokens);
    only the windows that are drawn are read from disk."""

    def __init__(self, tokens, context, batch_size, rng=None):
        if len(tokens) < context + 2:
            raise ValueError(f"need at least context + 2 = {context + 2} tokens, got {len(tokens)}")
        self.tokens, self.context, self.batch_size = tokens, context, batch_size
        self.rng = config.get_rng(rng)

    def __call__(self):
        start = self.rng.integers(0, len(self.tokens) - self.context - 1, self.batch_size)
        idx = start[:, None] + np.arange(self.context)[None, :]
        return np.asarray(self.tokens[idx], dtype=np.int64), np.asarray(self.tokens[idx + 1], dtype=np.int64)

    def windows(self, batch_size=64, max_windows=None):
        """Non-overlapping windows in order, for evaluation: all of them, or `max_windows`
        spread evenly across the data."""
        n = (len(self.tokens) - 1) // self.context
        which = np.arange(n) if not max_windows or max_windows >= n else np.linspace(0, n - 1, max_windows).astype(np.int64)
        for i in range(0, len(which), batch_size):
            idx = (which[i:i + batch_size] * self.context)[:, None] + np.arange(self.context)[None, :]
            yield np.asarray(self.tokens[idx], dtype=np.int64), np.asarray(self.tokens[idx + 1], dtype=np.int64)


def make_spirals(n_per_class=500, classes=3, turns=1.75, noise=0.12, rng=None):
    """Interleaved 2-D spirals: a small problem that a linear model cannot solve."""
    rng = config.get_rng(rng)
    X, y = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1.0, n_per_class)
        t = np.linspace(0, turns * 2 * np.pi, n_per_class) + c * 2 * np.pi / classes + rng.normal(0, noise, n_per_class)
        X.append(np.stack([r * np.sin(t), r * np.cos(t)], 1))
        y.append(np.full(n_per_class, c))
    X, y = np.concatenate(X).astype(config.get_dtype()), np.concatenate(y).astype(np.int64)
    order = rng.permutation(len(X))
    return X[order], y[order]
