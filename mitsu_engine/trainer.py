"""Training loop with evaluation, schedules, clipping, checkpoints and early stopping."""
import math, time
import numpy as np
from . import config
from .losses import CrossEntropy, accuracy
from .optim import clip_grad_norm
from .io import save_checkpoint, load_checkpoint


class Trainer:
    """Trainer(model, optimizer, loss_fn=CrossEntropy(), scheduler=None, clip_norm=None, metric=accuracy)

    scheduler   callable step -> learning rate (see optim.WarmupCosine), applied before each step
    clip_norm   clip the joint gradient norm to this value
    metric      callable (predictions, targets) -> float, or None
    """

    def __init__(self, model, optimizer, loss_fn=None, scheduler=None, clip_norm=None, metric=accuracy, log=print):
        self.model, self.optimizer = model, optimizer
        self.loss_fn = loss_fn or CrossEntropy()
        self.scheduler, self.clip_norm, self.metric = scheduler, clip_norm, metric
        self.log = log or (lambda *a, **k: None)
        self.history = []
        self.step = 0

    def train_step(self, x, y):
        """One optimisation step. Returns (loss, metric, gradient norm before clipping)."""
        self.model.train()
        if self.scheduler is not None:
            self.optimizer.lr = self.scheduler(self.step)
        self.optimizer.zero_grad()
        out = self.model(x)
        loss, grad = self.loss_fn(out, y)
        if not math.isfinite(loss):
            raise FloatingPointError(f"loss became {loss} at step {self.step}")
        self.model.backward(grad)
        gnorm = clip_grad_norm(self.optimizer.params, self.clip_norm) if self.clip_norm else float("nan")
        self.optimizer.step()
        self.step += 1
        return loss, (self.metric(out, y) if self.metric else float("nan")), gnorm

    def evaluate(self, batches):
        """Average loss and metric over an iterable of (x, y) batches, without gradients."""
        self.model.eval()
        tot_loss = tot_metric = n = 0
        with config.no_grad():
            for x, y in batches:
                out = self.model(x)
                loss, _ = self.loss_fn(out, y)
                k = np.asarray(y).size
                tot_loss += loss * k
                tot_metric += (self.metric(out, y) if self.metric else 0.0) * k
                n += k
        self.model.train()
        return {"loss": tot_loss / max(n, 1), "metric": tot_metric / max(n, 1)}

    def fit(self, train_loader, val_loader=None, epochs=1, checkpoint=None, patience=None):
        """Train for `epochs` passes. With `checkpoint`, the best model by validation loss is
        saved there and restored at the end. With `patience`, stop after that many epochs
        without improvement. Returns the history (one dict per epoch)."""
        best, bad = float("inf"), 0
        for epoch in range(1, epochs + 1):
            t0, tl, tm, n = time.time(), 0.0, 0.0, 0
            for x, y in train_loader:
                loss, metric, _ = self.train_step(x, y)
                k = np.asarray(y).size
                tl += loss * k; tm += metric * k; n += k
            row = {"epoch": epoch, "step": self.step, "train_loss": tl / n, "train_metric": tm / n, "seconds": time.time() - t0,
                   "lr": self.optimizer.lr}
            msg = f"epoch {epoch:3d}  train loss {row['train_loss']:.4f}  metric {row['train_metric']:.4f}"
            if val_loader is not None:
                val = self.evaluate(val_loader)
                row.update(val_loss=val["loss"], val_metric=val["metric"])
                msg += f"  |  val loss {val['loss']:.4f}  metric {val['metric']:.4f}"
                if val["loss"] < best - 1e-9:
                    best, bad = val["loss"], 0
                    if checkpoint:
                        save_checkpoint(checkpoint, self.model, meta={"epoch": epoch, "val_loss": best})
                else:
                    bad += 1
            self.history.append(row)
            self.log(msg + f"  |  {row['seconds']:.1f}s")
            if patience is not None and val_loader is not None and bad >= patience:
                self.log(f"stopping early: no improvement for {patience} epochs")
                break
        if checkpoint and val_loader is not None and best < float("inf"):
            load_checkpoint(checkpoint, self.model)
        return self.history

    def fit_steps(self, next_batch, steps, eval_fn=None, eval_every=None, log_every=None):
        """Train for a fixed number of steps, drawing batches from the callable `next_batch`.
        `eval_fn()` (returning a dict) is called every `eval_every` steps and at the end."""
        t0, run_loss, run_n = time.time(), 0.0, 0
        for i in range(1, steps + 1):
            x, y = next_batch()
            loss, metric, gnorm = self.train_step(x, y)
            run_loss += loss; run_n += 1
            at_eval = eval_fn is not None and ((eval_every and i % eval_every == 0) or i == steps)
            if at_eval or (log_every and i % log_every == 0):
                row = {"step": self.step, "train_loss": run_loss / run_n, "lr": self.optimizer.lr, "seconds": time.time() - t0}
                msg = f"step {self.step:6d}  train loss {row['train_loss']:.4f}"
                if at_eval:
                    val = eval_fn()
                    row.update({"val_" + k: v for k, v in val.items()})
                    msg += "  |  " + "  ".join(f"val {k} {v:.4f}" for k, v in val.items())
                self.history.append(row)
                self.log(msg + f"  |  {row['seconds']:.0f}s")
                run_loss, run_n = 0.0, 0
        return self.history
