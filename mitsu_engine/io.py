"""Checkpoints: one .npz file holding the parameters, optional optimiser state and metadata.

The file layout is
    param/<name>     one array per parameter, under the name Module.named_parameters gives it
    optim/<name>     optimiser state (step counter and moment arrays), if saved
    meta             a JSON string

`write_checkpoint` and `read_checkpoint` work on plain dictionaries of arrays. They are the
only code that knows the layout, so everything that reads or writes these files (this
engine and its PyTorch twin) goes through them.
"""
import json, os
import numpy as np


def _npz(path):
    return path if path.endswith(".npz") else path + ".npz"


def write_checkpoint(path, params, optim=None, meta=None):
    """Write dictionaries of arrays to `path`. The file is written under a temporary name
    and then renamed, so an interrupted save never leaves a half-written checkpoint."""
    arrays = {"param/" + k: np.asarray(v) for k, v in params.items()}
    arrays.update({"optim/" + k: np.asarray(v) for k, v in (optim or {}).items()})
    arrays["meta"] = np.array(json.dumps(meta or {}))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, _npz(path))


def read_checkpoint(path):
    """Return (params, optim, meta): two dictionaries of arrays and the metadata dict."""
    with np.load(_npz(path), allow_pickle=False) as z:
        params = {k[6:]: z[k] for k in z.files if k.startswith("param/")}
        optim = {k[6:]: z[k] for k in z.files if k.startswith("optim/")}
        return params, optim, json.loads(str(z["meta"]))


def save_checkpoint(path, model, optimizer=None, meta=None):
    """Write the model (and optionally the optimiser and a JSON-serialisable `meta` dict)."""
    write_checkpoint(path, model.state_dict(), optimizer.state_dict() if optimizer is not None else None, meta)


def load_checkpoint(path, model, optimizer=None, strict=True):
    """Load parameters into `model` (and state into `optimizer`). Returns the saved meta dict."""
    params, optim, meta = read_checkpoint(path)
    model.load_state_dict(params, strict=strict)
    if optimizer is not None and optim:
        optimizer.load_state_dict(optim)
    return meta


def read_meta(path):
    """The metadata dict of a checkpoint, without loading any weights."""
    with np.load(_npz(path), allow_pickle=False) as z:
        return json.loads(str(z["meta"]))
