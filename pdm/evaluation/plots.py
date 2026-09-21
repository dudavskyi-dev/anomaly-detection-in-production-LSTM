"""Diagnostic plots shared by every model: PR/ROC curves, RUL scatter/residuals, anomaly-score
distributions, and seed-stability bars. Saved under ``reports/`` — never displayed interactively,
since these run in scripts and CI.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_pr_curve(y_true, y_score, path: Path, label: str = "") -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=label or None)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    if label:
        ax.legend()
    _save(fig, path)


def plot_roc_curve(y_true, y_score, path: Path, label: str = "") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=label or None)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    if label:
        ax.legend()
    _save(fig, path)


def plot_rul_scatter(y_true, y_pred, path: Path) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=8, alpha=0.4)
    upper = float(max(y_true.max(), y_pred.max()))
    ax.plot([0, upper], [0, upper], linestyle="--", color="grey")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Predicted vs. true RUL")
    _save(fig, path)


def plot_residuals_vs_true_rul(y_true, y_pred, path: Path) -> None:
    y_true = np.asarray(y_true, dtype=float)
    residuals = np.asarray(y_pred, dtype=float) - y_true
    fig, ax = plt.subplots()
    ax.scatter(y_true, residuals, s=8, alpha=0.4)
    ax.axhline(0, linestyle="--", color="grey")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Residual (predicted - true)")
    ax.set_title("Residuals vs. true RUL")
    _save(fig, path)


def plot_score_distribution(scores_normal, scores_anomalous, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots()
    ax.hist(scores_normal, bins=40, alpha=0.6, label="normal")
    ax.hist(scores_anomalous, bins=40, alpha=0.6, label="anomalous")
    ax.axvline(threshold, color="red", linestyle="--", label="threshold")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by class")
    ax.legend()
    _save(fig, path)


def plot_seed_stability(metric_name: str, values: list[float], path: Path) -> None:
    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values)
    ax.axhline(float(np.mean(values)), color="red", linestyle="--", label="mean")
    ax.set_xlabel("Seed index")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} across seeds")
    ax.legend()
    _save(fig, path)


def plot_f1_vs_bottleneck(bottleneck_f1: dict[int, tuple[float, float]], path: Path) -> None:
    """P06 deliverable #7: F1 (mean ± std over seeds) against LSTM-autoencoder bottleneck size —
    the curve that should show the identity-function collapse at a wide-enough latent dimension.
    ``bottleneck_f1``: ``{latent_dim: (mean_f1, std_f1)}``."""
    dims = sorted(bottleneck_f1)
    means = [bottleneck_f1[d][0] for d in dims]
    stds = [bottleneck_f1[d][1] for d in dims]
    fig, ax = plt.subplots()
    ax.errorbar(dims, means, yerr=stds, marker="o", capsize=4)
    ax.set_xlabel("Bottleneck (latent) dimension")
    ax.set_ylabel("F1")
    ax.set_xscale("log", base=2)
    ax.set_title("Anomaly-detection F1 vs. autoencoder bottleneck size")
    _save(fig, path)


def plot_sensor_heatmap(per_sensor_error: np.ndarray, feature_names: list[str], path: Path) -> None:
    """P06 deliverable #7: per-sensor reconstruction-error heatmap for a set of windows (e.g. the
    highest-scoring detected anomalies) — the explainability view of *which* sensors drove the
    anomaly score. ``per_sensor_error``: ``(n_windows, n_features)``."""
    fig, ax = plt.subplots(
        figsize=(max(6, len(feature_names) * 0.4), max(3, per_sensor_error.shape[0] * 0.3))
    )
    im = ax.imshow(per_sensor_error, aspect="auto", cmap="inferno")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
    ax.set_ylabel("Window (ranked by total anomaly score)")
    ax.set_title("Per-sensor reconstruction error")
    fig.colorbar(im, ax=ax, label="MSE")
    _save(fig, path)


def plot_training_curves(history: list[dict], path: Path) -> None:
    """``history``: a list of per-epoch dicts with at least ``epoch``, ``train_loss``,
    ``val_loss`` (as produced by ``EpochStats.to_dict()``) — one line each, against epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots()
    ax.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves")
    ax.legend()
    _save(fig, path)
