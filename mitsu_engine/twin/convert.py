"""Moving weights between the numpy engine and its PyTorch twin.

Everything here works from a model's `engine_layout`: the engine's parameter order with,
for each parameter, whether it is transposed and whether weight decay applies. A
TorchMitsuLM derives it from its config (the rules below); twin.to_torch records it for
any other model.

For MitsuLM, a parameter has the same name on both sides. Two things differ:

  * a Linear weight is stored (in, out) by the engine and (out, in) by torch.nn.Linear,
    so it is transposed on the way across. The token table and the norm gains are not;
  * with tied embeddings the shared table is listed once, as `tok.weight`, on both sides.
    The PyTorch output layer owns no parameter: it reads the table from the embedding.

`engine_param_names` is the engine's parameter order. Optimiser state is stored by
position in that order, so this function is what lets training state move between engines.
"""
import numpy as np
import torch


def engine_param_names(cfg):
    """Parameter names of a MitsuLM built from `cfg`, in the engine's order."""
    names = ["tok.weight"]
    for i in range(cfg.n_layers):
        b = f"blocks.{i}."
        names += [b + "norm1.gain", b + "attn.qkv.weight", b + "attn.proj.weight", b + "norm2.gain",
                  b + "ffn.w1.weight", b + "ffn.w3.weight", b + "ffn.w2.weight"]
    names.append("norm.gain")
    if not cfg.tie_embeddings:
        names.append("head.weight")
    return names


def is_linear_weight(name):
    """True for the parameters that are transposed between the engines (and that receive
    weight decay): every weight matrix except the token table."""
    return name.endswith(".weight") and name != "tok.weight"


def _to_torch(array, transposed):
    array = np.asarray(array)
    return np.ascontiguousarray(array.T) if transposed else np.array(array)


def _to_engine(tensor, transposed):
    t = tensor.detach()
    if t.dtype not in (torch.float32, torch.float64):
        t = t.float()
    array = t.cpu().numpy()
    return np.ascontiguousarray(array.T) if transposed else np.array(array)


def to_torch_array(name, array):
    """A MitsuLM array in PyTorch's layout, as a contiguous numpy array."""
    return _to_torch(array, is_linear_weight(name))


def to_engine_array(name, tensor):
    """A TorchMitsuLM tensor in the engine's layout, as a contiguous numpy array."""
    return _to_engine(tensor, is_linear_weight(name))


def lm_layout(model):
    """The engine's parameter list for a TorchMitsuLM: [(name, parameter, transposed, decay)]."""
    own = dict(model.named_parameters())             # a tied table appears once, as tok.weight
    names = engine_param_names(model.config)
    missing, unexpected = [n for n in names if n not in own], [n for n in own if n not in names]
    if missing or unexpected:
        raise KeyError(f"the PyTorch model does not have the engine's parameters. missing: {missing[:5]}  unexpected: {unexpected[:5]}")
    return [(n, own[n], is_linear_weight(n), is_linear_weight(n)) for n in names]


def layout(model):
    """[(name, torch parameter, transposed, decay)] in the engine's parameter order, for a
    TorchMitsuLM or anything built by twin.to_torch."""
    found = getattr(model, "engine_layout", None)
    if found is None:
        raise TypeError("this PyTorch module was not built by twin.to_torch or TorchMitsuLM, so the engine's parameter layout is unknown")
    return found


def engine_state(model):
    """The PyTorch model's weights as the engine's state dict: {name: numpy array}."""
    return {name: _to_engine(p, transposed) for name, p, transposed, _ in layout(model)}


@torch.no_grad()
def load_engine_state(model, state):
    """Copy an engine state dict into the PyTorch model. Names and shapes must match."""
    entries = layout(model)
    names = {entry[0] for entry in entries}
    missing, unexpected = [n for n in names if n not in state], [k for k in state if k not in names]
    if missing or unexpected:
        raise KeyError(f"state does not match the model. missing: {sorted(missing)[:5]}  unexpected: {unexpected[:5]}")
    for name, p, transposed, _ in entries:
        value = _to_torch(state[name], transposed)
        if value.shape != tuple(p.shape):
            raise ValueError(f"shape mismatch for {name}: model {tuple(p.shape)}, state {value.shape} (PyTorch layout)")
        p.copy_(torch.from_numpy(value).to(device=p.device, dtype=p.dtype))
    return model


def save_checkpoint(path, model, optimizer=None, meta=None):
    """Write a PyTorch model (and optionally its EngineAdamW) in the engine's checkpoint
    format. The numpy engine loads the file with me.load_checkpoint."""
    from ..io import write_checkpoint
    write_checkpoint(path, engine_state(model), optimizer.engine_state() if optimizer is not None else None, meta)


def load_checkpoint(path, model, optimizer=None):
    """Load a checkpoint written by either engine into a PyTorch model (and optimiser).
    Returns the saved metadata."""
    from ..io import read_checkpoint
    params, optim, meta = read_checkpoint(path)
    load_engine_state(model, params)
    if optimizer is not None and optim:
        optimizer.load_engine_state(optim)
    return meta


def pick_device(name="auto"):
    """'auto' -> cuda if available, then mps, then cpu. Anything else is passed to torch.device."""
    if name in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
