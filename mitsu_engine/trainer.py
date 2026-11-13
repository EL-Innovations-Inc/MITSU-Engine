"""Training loop: evaluation, schedules, clipping, gradient accumulation, checkpoints,
early stopping, a JSON-lines log, and exact resume."""
import json, math, os, time
import numpy as np
from . import config
from .losses import CrossEntropy, accuracy
from .optim import clip_grad_norm, grad_norm
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

    # ---- one optimisation step ---------------------------------------------
    def train_step(self, x=None, y=None, next_batch=None, accumulate=1):
        """One optimisation step. Returns (loss, metric, gradient norm before clipping).

        Either pass one batch (x, y), or pass `next_batch` (a callable returning (x, y))
        with accumulate=k: the gradients of k batches are averaged before the update,
        which is equivalent to one batch k times as large but uses a k-th of the memory."""
        self.model.train()
        if self.scheduler is not None:
            self.optimizer.lr = self.scheduler(self.step)
        self.optimizer.zero_grad()
        tot_loss = tot_metric = 0.0
        for _ in range(accumulate):
            if next_batch is not None:
                x, y = next_batch()
            out = self.model(x)
            loss, grad = self.loss_fn(out, y)
            if not math.isfinite(loss):
                raise FloatingPointError(f"loss became {loss} at step {self.step}")
            self.model.backward(grad if accumulate == 1 else grad / accumulate)
            tot_loss += loss / accumulate
            tot_metric += (self.metric(out, y) if self.metric else float("nan")) / accumulate
        params = self.optimizer.params
        gnorm = clip_grad_norm(params, self.clip_norm) if self.clip_norm else grad_norm(params)
        self.optimizer.step()
        self.step += 1
        return tot_loss, tot_metric, gnorm

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

    # ---- epoch-based training ---------------------------------------------
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
            row = {"epoch": epoch, "step": self.step, "train_loss": tl / max(n, 1), "train_metric": tm / max(n, 1), "seconds": time.time() - t0,
                   "lr": self.optimizer.lr}
            msg = f"epoch {epoch:3d}  train loss {row['train_loss']:.4f}  metric {row['train_metric']:.4f}"
            if val_loader is not None:
                val = self.evaluate(val_loader)
                row.update(val_loss=val["loss"], val_metric=val["metric"])
                msg += f"  |  val loss {val['loss']:.4f}  metric {val['metric']:.4f}"
                if val["loss"] < best - 1e-9:
                    best, bad = val["loss"], 0
                    if checkpoint:
                        self.save_model(checkpoint, {"epoch": epoch, "val_loss": best})
                else:
                    bad += 1
            self.history.append(row)
            self.log(msg + f"  |  {row['seconds']:.1f}s")
            if patience is not None and val_loader is not None and bad >= patience:
                self.log(f"stopping early: no improvement for {patience} epochs")
                break
        if checkpoint and val_loader is not None and best < float("inf"):
            self.load_model(checkpoint)
        return self.history

    def save_model(self, path, meta=None):
        """Write the model's weights (no optimiser state) to a checkpoint."""
        save_checkpoint(path, self.model, meta=meta)

    def load_model(self, path):
        """Load weights written by save_model. Returns the checkpoint's metadata."""
        return load_checkpoint(path, self.model)

    # ---- step-based training ----------------------------------------------
    def fit_steps(self, next_batch, steps, eval_fn=None, eval_every=None, log_every=None, accumulate=1,
                  state_path=None, state_every=None, log_file=None, rngs=None):
        """Train until the step counter reaches `steps`, drawing batches from `next_batch()`.

        eval_fn       callable returning a dict; called every `eval_every` steps and at the end
        accumulate    average the gradients of this many batches per step
        state_path    write the full training state there every `state_every` steps and at the
                      end, so an interrupted run continues exactly with load_state()
        log_file      append one JSON line per logged row
        rngs          extra named random generators to save with the state (see save_state)

        `steps` is the total: a trainer resumed at step 600 with steps=1000 runs 400 more."""
        t0, run_loss, run_n, gmax = time.time(), 0.0, 0, 0.0
        while self.step < steps:
            loss, _, gnorm = self.train_step(next_batch=next_batch, accumulate=accumulate)
            run_loss += loss; run_n += 1; gmax = max(gmax, gnorm)
            i = self.step
            at_eval = eval_fn is not None and ((eval_every and i % eval_every == 0) or i == steps)
            if at_eval or (log_every and i % log_every == 0):
                row = {"step": i, "train_loss": run_loss / run_n, "grad_norm_max": gmax, "lr": self.optimizer.lr, "seconds": time.time() - t0}
                msg = f"step {i:6d}  train loss {row['train_loss']:.4f}  grad norm max {gmax:.2f}"
                if at_eval:
                    val = eval_fn()
                    row.update({"val_" + k: v for k, v in val.items()})
                    msg += "  |  " + "  ".join(f"val {k} {v:.4f}" for k, v in val.items())
                self.history.append(row)
                self.log(msg + f"  |  {row['seconds']:.0f}s")
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(json.dumps(row) + "\n")
                run_loss, run_n, gmax = 0.0, 0, 0.0
            if state_path and ((state_every and i % state_every == 0) or i == steps):
                self.save_state(state_path, rngs)
        return self.history

    # ---- exact resume ------------------------------------------------------
    def save_state(self, path, rngs=None, meta=None):
        """Save everything needed to continue this run exactly: model, optimiser, step
        counter, history, and the state of the random generators (the global one, plus
        any passed in `rngs` as {name: Generator})."""
        state = {"step": self.step, "history": self.history, "global_rng": config.get_rng_state(),
                 "rngs": {name: config.get_rng_state(g) for name, g in (rngs or {}).items()}}
        save_checkpoint(path, self.model, self.optimizer, meta={**(meta or {}), "trainer": state})

    def load_state(self, path, rngs=None):
        """Restore a state written by save_state into this trainer's model and optimiser.
        Components must have been built the same way; random generators continue from
        where they were. Returns the checkpoint's metadata."""
        meta = load_checkpoint(path, self.model, self.optimizer)
        state = meta["trainer"]
        self.step, self.history = state["step"], state["history"]
        config.set_rng_state(state["global_rng"])
        for name, g in (rngs or {}).items():
            config.set_rng_state(state["rngs"][name], g)
        return meta


def evaluate_lm(model, batches):
    """Loss (nats per token) and perplexity of a language model over (x, y) batches."""
    loss_fn, tot, n = CrossEntropy(), 0.0, 0
    was_training = model.training
    model.eval()
    with config.no_grad():
        for x, y in batches:
            loss, _ = loss_fn(model(x), y)
            tot += loss * y.size
            n += y.size
    model.train(was_training)
    loss = tot / max(n, 1)
    return {"loss": loss, "perplexity": math.exp(min(loss, 50.0))}
