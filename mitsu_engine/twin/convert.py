"""Moving weights between the numpy engine and its PyTorch twin.

A parameter has the same name on both sides. Two things differ:

  * a Linear weight is stored (in, out) by the engine and (out, in) by torch.nn.Linear,
    so it is transposed on the way across. The token table and the norm gains are not;
  * with tied embeddings the engine lists the shared table once, as `tok.weight`. PyTorch
    also shows it as `head.weight`; that second name is never written to a checkpoint.

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


def to_torch_array(name, array):
    """An engine array in PyTorch's layout, as a contiguous numpy array."""
    array = np.asarray(array)
    return np.ascontiguousarray(array.T) if is_linear_weight(name) else np.array(array)


def to_engine_array(name, tensor):
    """A PyTorch tensor in the engine's layout, as a contiguous numpy array."""
    t = tensor.detach()
    if t.dtype not in (torch.float32, torch.float64):
        t = t.float()
    array = t.cpu().numpy()
    return np.ascontiguousarray(array.T) if is_linear_weight(name) else np.array(array)


def _named(model):
    """{engine name: torch Parameter}, in the engine's order."""
    own = dict(model.named_parameters())             # a tied table appears once, as tok.weight
    names = engine_param_names(model.config)
    missing, unexpected = [n for n in names if n not in own], [n for n in own if n not in names]
    if missing or unexpected:
        raise KeyError(f"the PyTorch model does not have the engine's parameters. missing: {missing[:5]}  unexpected: {unexpected[:5]}")
    return {n: own[n] for n in names}


def engine_state(model):
    """The PyTorch model's weights as the engine's state dict: {name: numpy array}."""
    return {name: to_engine_array(name, p) for name, p in _named(model).items()}


@torch.no_grad()
def load_engine_state(model, state):
    """Copy an engine state dict into the PyTorch model. Names and shapes must match."""
    own = _named(model)
    missing, unexpected = [k for k in own if k not in state], [k for k in state if k not in own]
    if missing or unexpected:
        raise KeyError(f"state does not match the model. missing: {missing[:5]}  unexpected: {unexpected[:5]}")
    for name, p in own.items():
        value = to_torch_array(name, state[name])
        if value.shape != tuple(p.shape):
            raise ValueError(f"shape mismatch for {name}: model {tuple(p.shape)}, state {value.shape} (PyTorch layout)")
        p.copy_(torch.from_numpy(value).to(device=p.device, dtype=p.dtype))
    return model


def pick_device(name="auto"):
    """'auto' -> cuda if available, then mps, then cpu. Anything else is passed to torch.device."""
    if name in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
