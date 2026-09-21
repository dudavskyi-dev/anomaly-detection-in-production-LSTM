"""Threshold selection. Every function takes an explicit, required ``split`` argument naming
where the threshold was computed — a threshold picked on the test split can't happen by
accident, because there's no default to fall back on.

Both sweeps use ``sklearn.metrics.precision_recall_curve``'s single O(n log n) pass over sorted
scores rather than looping over every unique score and recomputing a full confusion matrix for
each candidate — the naive version is O(n * unique_scores) and becomes minutes-slow on anything
past a few thousand validation rows.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve

from pdm.evaluation.metrics import classification_metrics

TUNABLE_SPLITS = ("train", "validation")


def _check_split(split: str, allowed: tuple[str, ...]) -> None:
    if split not in allowed:
        raise ValueError(f"split must be one of {allowed}, got {split!r}")


def _precision_recall_thresholds(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """precision/recall/thresholds aligned 1:1 (sklearn appends an extra threshold-less point
    for "predict nothing positive", which we drop here so every index has a real threshold)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return precision[: len(thresholds)], recall[: len(thresholds)], thresholds


def max_f1_threshold(y_true, scores, *, split: str) -> dict:
    """The threshold (among all those precision_recall_curve considers) maximising F1, plus the
    full metrics achieved at it and which split it was computed on."""
    _check_split(split, TUNABLE_SPLITS)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    precision, recall, thresholds = _precision_recall_thresholds(y_true, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    best_idx = int(np.argmax(f1))
    threshold = float(thresholds[best_idx])

    y_pred = (scores >= threshold).astype(int)
    metrics = classification_metrics(y_true, y_pred)
    metrics["threshold"] = threshold
    metrics["split"] = split
    return metrics


def threshold_at_precision(
    y_true, scores, min_precision: float, *, split: str, include_brier: bool = True
) -> dict:
    """The lowest threshold (highest recall) that still achieves at least ``min_precision``.

    ``include_brier=False`` for callers whose ``scores`` aren't a probability (P06's anomaly
    detectors) — see :func:`pdm.evaluation.metrics.classification_metrics`.
    """
    _check_split(split, TUNABLE_SPLITS)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    precision, recall, thresholds = _precision_recall_thresholds(y_true, scores)
    meets_target = precision >= min_precision
    if not meets_target.any():
        raise ValueError(
            f"No threshold on this {split} split achieves precision >= {min_precision}"
        )

    candidates = np.where(meets_target)[0]
    best_idx = candidates[np.argmax(recall[candidates])]
    threshold = float(thresholds[best_idx])

    y_pred = (scores >= threshold).astype(int)
    metrics = classification_metrics(y_true, y_pred, scores, include_brier=include_brier)
    metrics["threshold"] = threshold
    metrics["split"] = split
    metrics["min_precision"] = min_precision
    return metrics


def percentile_threshold(train_scores, q: float) -> dict:
    """The ``q``-th percentile of the **training** score distribution.

    For unsupervised anomaly scoring, where there is no validation label to optimise a
    max-F1/precision-constrained threshold against — the only honest source of a threshold is
    the training distribution itself. Always computed on ``split="train"``; there is no
    parameter to point it at validation or test.
    """
    threshold = float(np.percentile(np.asarray(train_scores, dtype=float), q))
    return {"threshold": threshold, "percentile": q, "split": "train"}
