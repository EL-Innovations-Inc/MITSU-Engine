"""Checkpoints: one .npz file holding the parameters, optional optimiser state and metadata."""
import json, os
import numpy as np


def save_checkpoint(path, model, optimizer=None, meta=None):
    """Write the model (and optionally the optimiser and a JSON-serialisable `meta` dict)."""
    arrays = {"param/" + k: v for k, v in model.state_dict().items()}
    if optimizer is not None:
        arrays.update({"optim/" + k: np.asarray(v) for k, v in optimizer.state_dict().items()})
    arrays["meta"] = np.array(json.dumps(meta or {}))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path if path.endswith(".npz") else path + ".npz")


def load_checkpoint(path, model, optimizer=None, strict=True):
    """Load parameters into `model` (and state into `optimizer`). Returns the saved meta dict."""
    with np.load(path if path.endswith(".npz") else path + ".npz", allow_pickle=False) as z:
        model.load_state_dict({k[6:]: z[k] for k in z.files if k.startswith("param/")}, strict=strict)
        if optimizer is not None:
            state = {k[6:]: z[k] for k in z.files if k.startswith("optim/")}
            if state:
                optimizer.load_state_dict(state)
        return json.loads(str(z["meta"]))


def read_meta(path):
    """The metadata dict of a checkpoint, without loading any weights."""
    with np.load(path if path.endswith(".npz") else path + ".npz", allow_pickle=False) as z:
        return json.loads(str(z["meta"]))
