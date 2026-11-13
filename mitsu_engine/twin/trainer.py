"""Training the PyTorch twin with the engine's loop.

TorchTrainer is the engine's Trainer with the optimisation step replaced. Schedules,
logging, evaluation points, early stopping and the state file are the engine's own code,
so a run looks the same whichever engine executes it, and a state file written by one can
be resumed by the other (at dropout 0 the two follow the same trajectory; see docs/GPU.md).
It trains any model built by twin.to_torch, and TorchMitsuLM.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F
from .. import config as engine_config
from ..io import read_checkpoint, write_checkpoint
from ..trainer import Trainer
from .convert import engine_state, load_engine_state
from .optim import EngineAdamW, clip_grad_norm, grad_norm

AMP_DTYPES = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}
LOW_PRECISION = (torch.float16, torch.bfloat16)


def _cross_entropy(out, y):
    return F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))


def _accuracy(out, y):
    return (out.argmax(-1) == y).sum().item() / max(y.numel(), 1)


LOSSES = {"cross_entropy": _cross_entropy, "mse": F.mse_loss}
METRICS = {None: None, "accuracy": _accuracy}


def _to_device(x, device, dtype):
    """A batch as a tensor on `device`: integers become int64, floats take the model's precision."""
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.ascontiguousarray(x))
    return x.to(device=device, dtype=dtype if x.is_floating_point() else torch.long)


def _tokens(x, device):
    return _to_device(x, device, torch.float32)


class TorchTrainer(Trainer):
    """TorchTrainer(model, optimizer, loss="cross_entropy", metric=None, scheduler=None, clip_norm=None, amp="off")

    model       a TorchMitsuLM, or any model from twin.to_torch (already on its device)
    optimizer   twin.EngineAdamW(model, ...)
    loss        "cross_entropy" (integer targets, any number of leading axes), "mse", or a
                callable (output tensor, target tensor) -> scalar tensor
    metric      None, "accuracy", or a callable (output tensor, target tensor) -> float
    scheduler   callable step -> learning rate; the engine's WarmupCosine works unchanged
    amp         "off"   full precision
                "bf16"  autocast to bfloat16 (no gradient scaler needed)
                "fp16"  autocast to float16 with a gradient scaler

    Batches may be numpy arrays (the engine's DataLoader and TextBatcher) or tensors. `fit`,
    `fit_steps`, `history` and the log format are inherited from the engine's Trainer, and
    the files it writes are the engine's.
    """

    def __init__(self, model, optimizer, loss="cross_entropy", metric=None, scheduler=None, clip_norm=None, amp="off", log=print):
        super().__init__(model, optimizer, scheduler=scheduler, clip_norm=clip_norm, metric=None, log=log)
        if amp not in AMP_DTYPES:
            raise ValueError(f"amp must be one of {sorted(AMP_DTYPES)}")
        if not isinstance(optimizer, EngineAdamW):
            raise TypeError("TorchTrainer needs twin.EngineAdamW: the schedule sets its learning rate and the state file stores "
                            "its moments in the engine's layout")
        if not callable(loss) and loss not in LOSSES:
            raise ValueError(f"loss must be one of {sorted(LOSSES)} or a callable")
        if not callable(metric) and metric not in METRICS:
            raise ValueError("metric must be None, 'accuracy' or a callable")
        self.loss_fn = loss if callable(loss) else LOSSES[loss]
        self.metric = metric if callable(metric) else METRICS[metric]
        self.amp = amp
        first = next(model.parameters())
        self.device, self.dtype = first.device, first.dtype
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(amp == "fp16"))

    def autocast(self):
        return torch.autocast(self.device.type, dtype=AMP_DTYPES[self.amp], enabled=self.amp != "off")

    def _loss(self, x, y):
        """Forward pass under autocast; the loss itself is computed in at least float32."""
        x, y = _to_device(x, self.device, self.dtype), _to_device(y, self.device, self.dtype)
        with self.autocast():
            out = self.model(x)
        if out.dtype in LOW_PRECISION:
            out = out.float()
        return self.loss_fn(out, y), out, y

    def train_step(self, x=None, y=None, next_batch=None, accumulate=1):
        """One optimisation step. Returns (loss, metric, gradient norm before clipping)."""
        self.model.train()
        if self.scheduler is not None:
            self.optimizer.lr = self.scheduler(self.step)
        self.optimizer.zero_grad(set_to_none=True)
        total = total_metric = 0.0
        for _ in range(accumulate):
            if next_batch is not None:
                x, y = next_batch()
            loss, out, target = self._loss(x, y)
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"loss became {value} at step {self.step}")
            self.scaler.scale(loss if accumulate == 1 else loss / accumulate).backward()
            total += value / accumulate
            total_metric += (self.metric(out.detach(), target) if self.metric else float("nan")) / accumulate
        self.scaler.unscale_(self.optimizer)
        params = list(self.model.parameters())
        gnorm = clip_grad_norm(params, self.clip_norm) if self.clip_norm else grad_norm(params)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        return total, total_metric, gnorm

    @torch.no_grad()
    def evaluate(self, batches):
        """Average loss and metric over an iterable of (x, y) batches, without gradients."""
        self.model.eval()
        tot_loss = tot_metric = n = 0
        for x, y in batches:
            loss, out, target = self._loss(x, y)
            k = target.numel()
            tot_loss += loss.item() * k
            tot_metric += (self.metric(out, target) if self.metric else 0.0) * k
            n += k
        self.model.train()
        return {"loss": tot_loss / max(n, 1), "metric": tot_metric / max(n, 1)}

    def save_model(self, path, meta=None):
        write_checkpoint(path, engine_state(self.model), meta=meta)

    def load_model(self, path):
        params, _, meta = read_checkpoint(path)
        load_engine_state(self.model, params)
        return meta

    # ---- state in the engine's format -----------------------------------
    def save_state(self, path, rngs=None, meta=None):
        """Write the full training state in the engine's format: weights, optimiser
        moments, step counter, history and the numpy random generators (which decide the
        batches). The PyTorch generators (which decide dropout) and the gradient scaler are
        stored too; the numpy engine ignores them."""
        state = {"step": self.step, "history": self.history, "global_rng": engine_config.get_rng_state(),
                 "rngs": {name: engine_config.get_rng_state(g) for name, g in (rngs or {}).items()},
                 "backend": "torch", "torch_rng": bytes(torch.get_rng_state().tolist()).hex()}
        if self.device.type == "cuda":
            state["cuda_rng"] = [bytes(s.tolist()).hex() for s in torch.cuda.get_rng_state_all()]
        if self.scaler.is_enabled():
            state["scaler"] = self.scaler.state_dict()
        write_checkpoint(path, engine_state(self.model), self.optimizer.engine_state(), meta={**(meta or {}), "trainer": state})

    def load_state(self, path, rngs=None):
        """Restore a state written by this class or by the engine's Trainer.save_state."""
        params, optim, meta = read_checkpoint(path)
        load_engine_state(self.model, params)
        if optim:
            self.optimizer.load_engine_state(optim)
        state = meta["trainer"]
        self.step, self.history = state["step"], state["history"]
        engine_config.set_rng_state(state["global_rng"])
        for name, g in (rngs or {}).items():
            engine_config.set_rng_state(state["rngs"][name], g)
        if "torch_rng" in state:
            torch.set_rng_state(torch.tensor(list(bytes.fromhex(state["torch_rng"])), dtype=torch.uint8))
        if "cuda_rng" in state and self.device.type == "cuda" and len(state["cuda_rng"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([torch.tensor(list(bytes.fromhex(s)), dtype=torch.uint8) for s in state["cuda_rng"]])
        if "scaler" in state and self.scaler.is_enabled():
            self.scaler.load_state_dict(state["scaler"])
        return meta


@torch.no_grad()
def evaluate_lm(model, batches, amp="off"):
    """Loss (nats per token) and perplexity over (x, y) batches, as the engine's evaluate_lm."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for x, y in batches:
        with torch.autocast(device.type, dtype=AMP_DTYPES[amp], enabled=amp != "off"):
            loss = model.loss(_tokens(x, device), _tokens(y, device))
        k = int(np.prod(y.shape))
        total += loss.item() * k
        n += k
    model.train(was_training)
    loss = total / max(n, 1)
    return {"loss": loss, "perplexity": math.exp(min(loss, 50.0))}
