"""Explicit PyTorch training loop (no Lightning — every line here is meant to be explainable):
Adam, gradient clipping, early stopping with best-checkpoint restore, full determinism, and the
per-run benchmark instrumentation (wall-clock, epochs-to-best, peak RSS, inference latency).
"""

import copy
import random
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

from pdm.config import settings


def set_full_determinism(seed: int) -> list[str]:
    """Seed python/numpy/torch and request deterministic algorithms.

    **Call this before constructing your model**, not just before training it. ``train_model``
    also calls this internally, but only to make the *training loop's* randomness (data
    shuffling order, dropout masks) reproducible — by the time ``train_model`` runs, a model
    passed in has already been constructed, and its initial weights were drawn from whatever
    the ambient torch RNG state happened to be at that point. Relying on ``train_model``'s
    internal call alone to reproduce a full run end-to-end (including weight init) will silently
    fail to do so — this was caught by ``tests/test_torch_train.py``'s determinism test the first
    time this module was written, see ``docs/decisions/P04-pytorch.md``.

    Uses ``warn_only=True``: if some op PyTorch is asked to run has no deterministic CPU
    implementation, it warns and falls back rather than crashing training. Returns the text of
    any such warnings raised during this call (there normally are none — the LSTM/Linear/Adam
    ops used here all have deterministic CPU implementations; this exists to *notice* if that
    ever changes rather than assume it).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.use_deterministic_algorithms(True, warn_only=True)
        return [str(w.message) for w in caught]


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_score: float  # RMSE for the regressor, BCE loss for the classifier
    epoch_seconds: float
    lr: float

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_score": self.val_score,
            "epoch_seconds": self.epoch_seconds,
            "lr": self.lr,
        }


@dataclass
class TrainResult:
    history: list[EpochStats] = field(default_factory=list)
    best_epoch: int = 0
    best_val_score: float = float("inf")
    train_wall_clock_seconds: float = 0.0
    peak_rss_bytes: int = 0
    determinism_warnings: list[str] = field(default_factory=list)
    diverged: bool = False


def rmse_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def bce_with_logits_score(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
    return nn.functional.binary_cross_entropy_with_logits(y_pred_logits, y_true).item()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    target_col: str,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eval_score_fn: Callable[[torch.Tensor, torch.Tensor], float],
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    patience: int | None = None,
    grad_clip_norm: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> TrainResult:
    """Train ``model`` with Adam + gradient clipping, early-stopping on ``eval_score_fn`` (lower
    is better) computed on ``val_loader``, and restore the best epoch's weights before returning.
    """
    max_epochs = settings.training.max_epochs if max_epochs is None else max_epochs
    learning_rate = settings.training.learning_rate if learning_rate is None else learning_rate
    patience = settings.training.early_stopping_patience if patience is None else patience
    grad_clip_norm = settings.training.grad_clip_norm if grad_clip_norm is None else grad_clip_norm
    seed = settings.seed if seed is None else seed

    determinism_warnings = set_full_determinism(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    result = TrainResult(determinism_warnings=determinism_warnings)
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_improvement = 0

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    train_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["windows"])
            loss = loss_fn(pred, batch[target_col])
            if not torch.isfinite(loss):
                result.diverged = True
                model.load_state_dict(best_state)
                result.train_wall_clock_seconds = time.perf_counter() - train_start
                result.peak_rss_bytes = peak_rss
                return result
            loss.backward()
            if grad_clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["windows"])
                val_losses.append(loss_fn(pred, batch[target_col]).item())
                val_preds.append(pred)
                val_true.append(batch[target_col])
        val_pred_all = torch.cat(val_preds)
        val_true_all = torch.cat(val_true)
        val_score = eval_score_fn(val_true_all, val_pred_all)

        peak_rss = max(peak_rss, process.memory_info().rss)
        stats = EpochStats(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            val_loss=float(np.mean(val_losses)),
            val_score=val_score,
            epoch_seconds=time.perf_counter() - epoch_start,
            lr=optimizer.param_groups[0]["lr"],
        )
        result.history.append(stats)
        if verbose:
            print(stats.to_dict())

        if val_score < result.best_val_score - 1e-6:
            result.best_val_score = val_score
            result.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                break

    model.load_state_dict(best_state)
    result.train_wall_clock_seconds = time.perf_counter() - train_start
    result.peak_rss_bytes = peak_rss
    return result


def train_regressor(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, **kwargs
) -> TrainResult:
    return train_model(
        model,
        train_loader,
        val_loader,
        target_col="rul",
        loss_fn=nn.MSELoss(),
        eval_score_fn=rmse_score,
        **kwargs,
    )


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    pos_weight: torch.Tensor | None = None,
    **kwargs,
) -> TrainResult:
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return train_model(
        model,
        train_loader,
        val_loader,
        target_col="will_fail",
        loss_fn=loss_fn,
        eval_score_fn=bce_with_logits_score,
        **kwargs,
    )


def measure_inference_latency(
    model: nn.Module, sample_window: torch.Tensor, *, n_calls: int = 1000, n_warmup: int = 50
) -> dict:
    """Single-sample (batch size 1) inference latency in milliseconds, p50/p95/p99 over
    ``n_calls`` timed calls after ``n_warmup`` untimed warmup calls."""
    model.eval()
    single = sample_window.unsqueeze(0) if sample_window.dim() == 2 else sample_window
    with torch.no_grad():
        for _ in range(n_warmup):
            model(single)

        timings_ms = []
        for _ in range(n_calls):
            start = time.perf_counter()
            model(single)
            timings_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(timings_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "n_calls": n_calls,
    }
