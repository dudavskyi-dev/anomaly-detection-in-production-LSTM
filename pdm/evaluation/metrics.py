"""Regression and classification metrics, plus the C-MAPSS NASA asymmetric scoring function.

Every classification metric that sklearn computes here also has a hand-written reference
derivation exercised in ``tests/test_evaluation_metrics.py`` against a small fixture, so
precision/recall/F1 can be derived from raw confusion-matrix counts without library help.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r_squared(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def nasa_score(y_true, y_pred) -> float:
    """NASA's asymmetric prognostics score (Saxena & Goebel, 2008).

    ``d = predicted - true``. Early predictions (``d < 0``: the model thinks less life remains
    than truly does — a conservative error that triggers an unnecessary inspection) are
    penalised gently; late predictions (``d >= 0``: the model thinks *more* life remains than
    truly does — an error that could let a real failure happen) are penalised much harder. A
    missed failure costs far more than a needless inspection, so the score reflects that
    asymmetry directly rather than treating both directions of error the same, the way RMSE does.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    early = d[d < 0]
    late = d[d >= 0]
    score = np.sum(np.exp(-early / 13.0) - 1.0) + np.sum(np.exp(late / 10.0) - 1.0)
    return float(score)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r_squared(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, y_score=None, *, include_brier: bool = True) -> dict:
    """precision, recall, F1, and confusion-matrix counts; adds PR-AUC/ROC-AUC when a continuous
    score (not just the hard 0/1 prediction) is available, plus Brier score if ``include_brier``
    (default ``True``, matching every existing caller's calibrated-probability scores).

    ``include_brier=False`` exists for callers whose "score" is not a probability — P06's
    anomaly detectors report a standardised/reconstruction-error score with no [0, 1] bound,
    and ``brier_score_loss`` raises on those rather than silently producing a meaningless
    number: pass ``include_brier=False`` there instead of trying to interpret a Brier score for
    a quantity that was never a probability.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        if include_brier:
            metrics["brier"] = float(brier_score_loss(y_true, y_score))
    return metrics


def precision_recall_f1_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Hand-derived precision/recall/F1 straight from confusion-matrix counts — no sklearn.

    This is the reference implementation ``tests/test_evaluation_metrics.py`` checks the
    sklearn-backed :func:`classification_metrics` against; it exists so precision/recall/F1 can
    be explained and derived from first principles, not just called from a library.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
