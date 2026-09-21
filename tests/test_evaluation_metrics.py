"""Regression/classification metrics and the NASA asymmetric score, checked against
hand-computed values — not just against sklearn calling itself."""

import numpy as np
import pytest

from pdm.evaluation.metrics import (
    classification_metrics,
    mae,
    nasa_score,
    precision_recall_f1_from_confusion,
    r_squared,
    regression_metrics,
    rmse,
)

pytestmark = pytest.mark.fast


def test_rmse_hand_computed():
    y_true = [0.0, 0.0, 0.0, 0.0]
    y_pred = [3.0, 4.0, 0.0, 0.0]
    # errors: 3,4,0,0 -> squared: 9,16,0,0 -> mean 6.25 -> sqrt 2.5
    assert rmse(y_true, y_pred) == pytest.approx(2.5)


def test_mae_hand_computed():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [12.0, 18.0, 33.0]
    # abs errors: 2, 2, 3 -> mean 7/3
    assert mae(y_true, y_pred) == pytest.approx(7 / 3)


def test_r_squared_perfect_fit_is_one():
    y_true = [1.0, 2.0, 3.0, 4.0]
    assert r_squared(y_true, y_true) == pytest.approx(1.0)


def test_r_squared_predicting_the_mean_is_zero():
    y_true = [1.0, 2.0, 3.0, 4.0]
    mean_pred = [np.mean(y_true)] * 4
    assert r_squared(y_true, mean_pred) == pytest.approx(0.0, abs=1e-10)


def test_nasa_score_hand_computed_single_values():
    # d = pred - true. Early (d<0): exp(-d/13)-1. Late (d>=0): exp(d/10)-1.
    # d = -13 -> exp(1)-1
    assert nasa_score([100.0], [87.0]) == pytest.approx(np.exp(1) - 1)
    # d = 10 -> exp(1)-1 (same magnitude of penalty coincides at these specific offsets)
    assert nasa_score([100.0], [110.0]) == pytest.approx(np.exp(1) - 1)
    # d = 0 -> exp(0)-1 = 0
    assert nasa_score([50.0], [50.0]) == pytest.approx(0.0)


def test_nasa_score_penalises_late_predictions_harder_than_early_of_equal_magnitude():
    late = nasa_score([100.0], [120.0])  # d = +20
    early = nasa_score([100.0], [80.0])  # d = -20
    assert late > early


def test_nasa_score_sums_over_multiple_predictions():
    y_true = [100.0, 100.0]
    y_pred = [87.0, 110.0]  # d=-13 and d=10, both score exp(1)-1
    expected = 2 * (np.exp(1) - 1)
    assert nasa_score(y_true, y_pred) == pytest.approx(expected)


def test_regression_metrics_bundle_has_all_four_keys():
    metrics = regression_metrics([1.0, 2.0], [1.5, 2.5])
    assert set(metrics) == {"rmse", "mae", "r2", "nasa_score"}


# --- classification: sklearn-backed implementation agrees with a hand-derived reference --------


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def test_classification_metrics_agrees_with_hand_derived_reference():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0])

    tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
    hand = precision_recall_f1_from_confusion(tp, fp, fn, tn)
    sklearn_backed = classification_metrics(y_true, y_pred)

    assert sklearn_backed["precision"] == pytest.approx(hand["precision"])
    assert sklearn_backed["recall"] == pytest.approx(hand["recall"])
    assert sklearn_backed["f1"] == pytest.approx(hand["f1"])
    assert (
        sklearn_backed["tp"],
        sklearn_backed["fp"],
        sklearn_backed["fn"],
        sklearn_backed["tn"],
    ) == (
        tp,
        fp,
        fn,
        tn,
    )


def test_precision_recall_f1_from_confusion_hand_example():
    # tp=3, fp=1, fn=2, tn=4 -> precision 3/4, recall 3/5, f1 = 2*0.75*0.6/(0.75+0.6)
    out = precision_recall_f1_from_confusion(tp=3, fp=1, fn=2, tn=4)
    assert out["precision"] == pytest.approx(0.75)
    assert out["recall"] == pytest.approx(0.6)
    assert out["f1"] == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_precision_recall_f1_from_confusion_handles_zero_denominators():
    out = precision_recall_f1_from_confusion(tp=0, fp=0, fn=0, tn=10)
    assert out == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_classification_metrics_includes_score_based_metrics_when_scores_given():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_score = [0.1, 0.6, 0.7, 0.9]
    metrics = classification_metrics(y_true, y_pred, y_score)
    assert {"pr_auc", "roc_auc", "brier"}.issubset(metrics)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
