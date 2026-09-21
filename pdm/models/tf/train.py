"""``model.fit``-based TensorFlow/Keras training mirroring
``pdm.models.torch.train``'s public contract (``TrainResult``, ``EpochStats``,
``measure_inference_latency``) so both frameworks' benchmark numbers are directly comparable.

Uses Keras callbacks for the pieces the spec calls out explicitly: ``EarlyStopping`` (restoring
best weights), ``ModelCheckpoint``, ``CSVLogger``, and a custom per-epoch wall-clock callback.
"""

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras

from pdm.config import settings


def set_full_determinism(seed: int) -> list[str]:
    """Seed python/numpy/keras/tf and request deterministic ops.

    **Call this before constructing your model**, not just before training it — the same
    ordering requirement ``pdm.models.torch.train.set_full_determinism`` documents: a model's
    initial weights are drawn at construction time, so seeding afterwards cannot retroactively
    fix them. ``keras.utils.set_random_seed`` seeds python's ``random``, ``numpy``, and TF's
    global RNG in one call. Returns any warning/error text raised while enabling deterministic
    ops (normally empty), rather than crashing — mirroring the PyTorch ``warn_only=True``
    contract, since not every op has a deterministic CPU kernel on every TF version.
    """
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    warnings_caught: list[str] = []
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # pragma: no cover - environment/version dependent
        warnings_caught.append(str(exc))
    return warnings_caught


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


class _WallClockCallback(keras.callbacks.Callback):
    """Records each epoch's wall-clock duration — deliverable #2's "custom callback"."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_seconds: list[float] = []
        self._epoch_start = 0.0

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.epoch_seconds.append(time.perf_counter() - self._epoch_start)


class _PeakRSSCallback(keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self._process = psutil.Process()
        self.peak_rss_bytes = self._process.memory_info().rss

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._process.memory_info().rss)


def _weighted_bce_loss(pos_weight: float | None):
    if pos_weight is None:
        return lambda y_true, y_logits: tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_logits)
        )
    return lambda y_true, y_logits: tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=y_logits, pos_weight=pos_weight
        )
    )


def _run_fit(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    loss,
    monitor: str,
    metrics: list | None,
    max_epochs: int,
    learning_rate: float,
    patience: int,
    grad_clip_norm: float | None,
    seed: int,
    verbose: bool,
    csv_log_path: Path | None,
) -> TrainResult:
    determinism_warnings = set_full_determinism(seed)
    model.clip_norm = grad_clip_norm
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics or [],
    )

    wallclock_cb = _WallClockCallback()
    rss_cb = _PeakRSSCallback()
    terminate_nan_cb = keras.callbacks.TerminateOnNaN()
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor=monitor, mode="min", patience=patience, min_delta=1e-6, restore_best_weights=True
    )
    callbacks = [wallclock_cb, rss_cb, terminate_nan_cb, early_stop_cb]
    if csv_log_path is not None:
        csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.CSVLogger(str(csv_log_path)))

    train_start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1 if verbose else 0,
    )
    train_wall_clock_seconds = time.perf_counter() - train_start

    h = history.history
    n_epochs = len(h["loss"])
    val_scores = h[monitor]  # ``monitor`` is always passed already ``val_``-prefixed below
    diverged = bool(np.any(~np.isfinite(h["loss"])))
    if diverged:
        best_idx = int(np.nanargmin([v if np.isfinite(v) else np.inf for v in val_scores]))
    else:
        best_idx = int(np.argmin(val_scores))

    result = TrainResult(
        history=[
            EpochStats(
                epoch=i + 1,
                train_loss=float(h["loss"][i]),
                val_loss=float(h["val_loss"][i]),
                val_score=float(val_scores[i]),
                epoch_seconds=wallclock_cb.epoch_seconds[i],
                lr=learning_rate,
            )
            for i in range(n_epochs)
        ],
        best_epoch=best_idx + 1,
        best_val_score=float(val_scores[best_idx]),
        train_wall_clock_seconds=train_wall_clock_seconds,
        peak_rss_bytes=rss_cb.peak_rss_bytes,
        determinism_warnings=determinism_warnings,
        diverged=diverged,
    )
    return result


def train_regressor(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    patience: int | None = None,
    grad_clip_norm: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
    csv_log_path: Path | None = None,
) -> TrainResult:
    max_epochs = settings.training.max_epochs if max_epochs is None else max_epochs
    learning_rate = settings.training.learning_rate if learning_rate is None else learning_rate
    patience = settings.training.early_stopping_patience if patience is None else patience
    grad_clip_norm = settings.training.grad_clip_norm if grad_clip_norm is None else grad_clip_norm
    seed = settings.seed if seed is None else seed

    return _run_fit(
        model,
        train_ds,
        val_ds,
        loss=keras.losses.MeanSquaredError(),
        monitor="val_rmse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        patience=patience,
        grad_clip_norm=grad_clip_norm,
        seed=seed,
        verbose=verbose,
        csv_log_path=csv_log_path,
    )


def train_classifier(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    pos_weight: float | None = None,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    patience: int | None = None,
    grad_clip_norm: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
    csv_log_path: Path | None = None,
) -> TrainResult:
    max_epochs = settings.training.max_epochs if max_epochs is None else max_epochs
    learning_rate = settings.training.learning_rate if learning_rate is None else learning_rate
    patience = settings.training.early_stopping_patience if patience is None else patience
    grad_clip_norm = settings.training.grad_clip_norm if grad_clip_norm is None else grad_clip_norm
    seed = settings.seed if seed is None else seed

    return _run_fit(
        model,
        train_ds,
        val_ds,
        loss=_weighted_bce_loss(pos_weight),
        monitor="val_loss",
        metrics=None,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        patience=patience,
        grad_clip_norm=grad_clip_norm,
        seed=seed,
        verbose=verbose,
        csv_log_path=csv_log_path,
    )


def measure_inference_latency(
    model: keras.Model, sample_window: np.ndarray, *, n_calls: int = 1000, n_warmup: int = 50
) -> dict:
    """Single-sample (batch size 1) eager-mode inference latency in milliseconds, p50/p95/p99
    over ``n_calls`` timed calls after ``n_warmup`` untimed warmup calls — directly comparable
    to ``pdm.models.torch.train.measure_inference_latency``'s eager PyTorch measurement.
    """
    single = sample_window[np.newaxis, ...] if sample_window.ndim == 2 else sample_window
    single = tf.convert_to_tensor(single, dtype=tf.float32)

    for _ in range(n_warmup):
        model(single, training=False)

    timings_ms = []
    for _ in range(n_calls):
        start = time.perf_counter()
        model(single, training=False)
        timings_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(timings_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "n_calls": n_calls,
    }
