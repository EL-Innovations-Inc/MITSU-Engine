"""mitsu_engine.twin — the PyTorch copy of MitsuLM, checked against the numpy engine.

The numpy engine is the reference. This package holds the same architecture written with
PyTorch modules, so it runs on a GPU and under mixed precision, plus the code that moves
weights, checkpoints and training state between the two and measures how closely they agree.

    import mitsu_engine as me
    from mitsu_engine import twin

    ref = me.MitsuLM(me.LMConfig(vocab_size=1024))        # numpy
    net = twin.TorchMitsuLM.from_engine(ref)              # same weights, PyTorch
    print(twin.compare(ref, net))                         # largest differences

PyTorch is optional: `pip install mitsu-engine[torch]`. Nothing else in mitsu_engine
imports this package, so the core engine still needs numpy only.
"""
try:
    import torch as _torch
    _torch.nn.Linear
except Exception as exc:  # pragma: no cover - exercised only without PyTorch
    raise ImportError("mitsu_engine.twin needs PyTorch (pip install torch). The rest of mitsu_engine works without it.") from exc

from .functional import mitsu, softplus, rope_rotate
from .modules import MITSU, MitsuGLU, RMSNorm, Attention, Block
from .model import TorchMitsuLM
from .convert import (engine_param_names, is_linear_weight, to_torch_array, to_engine_array, engine_state, load_engine_state,
                      pick_device)
from .optim import EngineAdamW, clip_grad_norm, grad_norm
from .trainer import TorchTrainer, evaluate_lm
from .parity import compare, check, format_report

__all__ = ["mitsu", "softplus", "rope_rotate", "MITSU", "MitsuGLU", "RMSNorm", "Attention", "Block", "TorchMitsuLM",
           "engine_param_names", "is_linear_weight", "to_torch_array", "to_engine_array", "engine_state", "load_engine_state",
           "pick_device", "EngineAdamW", "clip_grad_norm", "grad_norm", "TorchTrainer", "evaluate_lm", "compare", "check",
           "format_report"]
