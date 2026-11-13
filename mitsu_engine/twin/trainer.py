"""Training the PyTorch twin with the engine's loop.

TorchTrainer is the engine's Trainer with the optimisation step replaced. Schedules,
logging, evaluation points and the state file are the engine's own code, so a run looks
the same whichever engine executes it, and a state file written by one can be resumed by
the other (at dropout 0 the two follow the same trajectory; see docs/TWIN.md).
"""
import math
import numpy as np
import torch
from .. import config as engine_config
from ..io import read_checkpoint, write_checkpoint
from ..trainer import Trainer
from .convert import engine_state, load_engine_state
from .optim import EngineAdamW, clip_grad_norm, grad_norm

AMP_DTYPES = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _tokens(x, device):
    """A batch of token ids as an int64 tensor on `device`."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64)).to(device)


class TorchTrainer(Trainer):
    """TorchTrainer(model, optimizer, scheduler=None, clip_norm=None, amp="off")

    model       a TorchMitsuLM (already on its device)
    optimizer   twin.EngineAdamW(model, ...)
    scheduler   callable step -> learning rate; the engine's WarmupCosine works unchanged
    amp         "off"   full precision
                "bf16"  autocast to bfloat16 (no gradient scaler needed)
                "fp16"  autocast to float16 with a gradient scaler

    Batches may be numpy arrays (the engine's TextBatcher) or tensors. `fit_steps`,
    `history` and the log format are inherited from the engine's Trainer.
    """

    def __init__(self, model, optimizer, scheduler=None, clip_norm=None, amp="off", log=print):
        super().__init__(model, optimizer, scheduler=scheduler, clip_norm=clip_norm, metric=None, log=log)
        if amp not in AMP_DTYPES:
            raise ValueError(f"amp must be one of {sorted(AMP_DTYPES)}")
        if not isinstance(optimizer, EngineAdamW):
            raise TypeError("TorchTrainer needs twin.EngineAdamW: the schedule sets its learning rate and the state file stores "
                            "its moments in the engine's layout")
        self.amp = amp
        self.device = next(model.parameters()).device
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(amp == "fp16"))

    def autocast(self):
        return torch.autocast(self.device.type, dtype=AMP_DTYPES[self.amp], enabled=self.amp != "off")

    def train_step(self, x=None, y=None, next_batch=None, accumulate=1):
        """One optimisation step. Returns (loss, nan, gradient norm before clipping)."""
        self.model.train()
        if self.scheduler is not None:
            self.optimizer.lr = self.scheduler(self.step)
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulate):
            if next_batch is not None:
                x, y = next_batch()
            with self.autocast():
                loss = self.model.loss(_tokens(x, self.device), _tokens(y, self.device))
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"loss became {value} at step {self.step}")
            self.scaler.scale(loss if accumulate == 1 else loss / accumulate).backward()
            total += value / accumulate
        self.scaler.unscale_(self.optimizer)
        params = list(self.model.parameters())
        gnorm = clip_grad_norm(params, self.clip_norm) if self.clip_norm else grad_norm(params)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.step += 1
        return total, float("nan"), gnorm

    def fit(self, *args, **kwargs):
        raise NotImplementedError("TorchTrainer trains by steps: use fit_steps")

    def evaluate(self, batches):
        result = evaluate_lm(self.model, batches, self.amp)
        return {"loss": result["loss"], "metric": float("nan")}

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
