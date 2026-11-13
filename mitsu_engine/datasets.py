"""Loaders for the public datasets the examples use. Standard library only."""
import gzip, os, urllib.request
import numpy as np

_IDX = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
_URLS = {
    "mnist": "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _fetch(url, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        print(f"downloading {url}")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def _find(folder, kind, name, download):
    for candidate in (os.path.join(folder, f"{kind}-{name}"), os.path.join(folder, kind, name), os.path.join(folder, name)):
        if os.path.exists(candidate):
            return candidate
    if not download:
        raise FileNotFoundError(f"{name} for '{kind}' not found in {folder}; pass download=True to fetch it")
    return _fetch(_URLS[kind] + name, os.path.join(folder, f"{kind}-{name}"))


def load_mnist(folder="data", kind="mnist", download=True, flatten=True):
    """Return (X_train, y_train, X_test, y_test). Images are float32 in [0, 1], shape (N, 784),
    or (N, 1, 28, 28) with flatten=False (the layout the convolution layers take).
    kind is 'mnist' or 'fashion'."""
    if kind not in _URLS:
        raise ValueError("kind must be 'mnist' or 'fashion'")
    paths = [_find(folder, kind, name, download) for name in _IDX]

    def images(p):
        with gzip.open(p) as f:
            return (np.frombuffer(f.read(), np.uint8, offset=16).reshape((-1, 784) if flatten else (-1, 1, 28, 28)) / 255.0).astype(np.float32)

    def labels(p):
        with gzip.open(p) as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    return images(paths[0]), labels(paths[1]), images(paths[2]), labels(paths[3])


def load_text(path="data/shakespeare.txt", url=SHAKESPEARE_URL, download=True):
    """Return the text at `path`, downloading Tiny Shakespeare there first if it is missing."""
    if not os.path.exists(path):
        if not download:
            raise FileNotFoundError(path)
        _fetch(url, path)
    with open(path, encoding="utf-8") as f:
        return f.read()
