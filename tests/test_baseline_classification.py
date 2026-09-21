"""Failure-classification baselines: the dummy floor never predicts the positive class, and
real models beat it on learnable synthetic data."""

import numpy as np
import pytest

from pdm.models.baseline.classification import (
    dummy_classifier_baseline,
    logistic_regression_baseline,
    random_forest_classifier_baseline,
)

pytestmark = pytest.mark.fast


def _make_synthetic_classification_windows(n=200, window_size=10, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    windows = rng.normal(size=(n, window_size, n_features))
    signal = windows.mean(axis=(1, 2))
    targets = (signal > np.median(signal)).astype(int)
    return windows, targets


def test_dummy_classifier_most_frequent_never_predicts_positive():
    train_targets = np.array([0, 0, 0, 1])
    eval_targets = np.array([0, 0, 1, 1])
    result = dummy_classifier_baseline(train_targets, eval_targets, seed=0)
    assert result["recall"] == pytest.approx(0.0)


def test_logistic_regression_beats_dummy_on_learnable_data():
    train_windows, train_targets = _make_synthetic_classification_windows(seed=0)
    eval_windows, eval_targets = _make_synthetic_classification_windows(seed=1)

    dummy = dummy_classifier_baseline(train_targets, eval_targets, seed=0)
    logreg = logistic_regression_baseline(
        train_windows, train_targets, eval_windows, eval_targets, seed=0
    )

    assert logreg["pr_auc"] > dummy["pr_auc"]


def test_random_forest_classifier_beats_dummy_on_learnable_data():
    train_windows, train_targets = _make_synthetic_classification_windows(seed=0)
    eval_windows, eval_targets = _make_synthetic_classification_windows(seed=1)

    dummy = dummy_classifier_baseline(train_targets, eval_targets, seed=0)
    rf = random_forest_classifier_baseline(
        train_windows, train_targets, eval_windows, eval_targets, seed=0
    )

    assert rf["pr_auc"] > dummy["pr_auc"]
