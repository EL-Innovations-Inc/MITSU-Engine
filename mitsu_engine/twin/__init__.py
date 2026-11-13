"""mitsu_engine.twin — PyTorch copies of the engine's models, checked against the numpy engine.

The numpy engine is the reference. This package builds the same networks from PyTorch
modules, so they run on a GPU and under mixed precision, plus the code that moves weights,
checkpoints and training state between the two and measures how closely they agree.

    import mitsu_engine as me
    from mitsu_engine import twin

    model = me.MitsuConvNet(1, 10)                        # any engine model, numpy
    net = twin.to_torch(model, device="cuda")             # same weights, PyTorch, on a GPU
    print(twin.compare_models(model, net, images))        # largest differences

    ref = me.MitsuLM(me.LMConfig(vocab_size=1024))        # the language model has a
    net = twin.TorchMitsuLM.from_engine(ref)              # dedicated twin with generation
    print(twin.compare(ref, net))

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
                      layout, pick_device, save_checkpoint, load_checkpoint)
from .optim import EngineAdamW, clip_grad_norm, grad_norm
from .trainer import TorchTrainer, evaluate_lm
from .parity import compare, check, format_report, compare_models, check_models, model_zoo
from .mirror import to_torch, to_engine, register
from . import gpucheck

__all__ = ["mitsu", "softplus", "rope_rotate", "MITSU", "MitsuGLU", "RMSNorm", "Attention", "Block", "TorchMitsuLM",
           "engine_param_names", "is_linear_weight", "to_torch_array", "to_engine_array", "engine_state", "load_engine_state",
           "pick_device", "EngineAdamW", "clip_grad_norm", "grad_norm", "TorchTrainer", "evaluate_lm", "compare", "check",
           "format_report", "compare_models", "check_models", "model_zoo", "to_torch", "to_engine", "register", "layout",
           "save_checkpoint", "load_checkpoint", "gpucheck"]
