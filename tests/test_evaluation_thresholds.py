"""Threshold selection: max-F1, precision-constrained, and training-percentile thresholds, plus
the explicit-split guard that makes picking a threshold on the wrong split a hard error.
"""

import numpy as np
import pytest

from pdm.evaluation.thresholds import max_f1_threshold, percentile_threshold, threshold_at_precision

pytestmark = pytest.mark.fast


def test_max_f1_threshold_requires_a_valid_split():
    with pytest.raises(ValueError, match="split must be one of"):
        max_f1_threshold([0, 1], [0.1, 0.9], split="test")


def test_threshold_at_precision_requires_a_valid_split():
    with pytest.raises(ValueError, match="split must be one of"):
        threshold_at_precision([0, 1], [0.1, 0.9], 0.9, split="test")


def test_max_f1_threshold_finds_the_perfect_separator():
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    result = max_f1_threshold(y_true, scores, split="validation")
    assert result["f1"] == pytest.approx(1.0)
    assert 0.3 < result["threshold"] <= 0.7
    assert result["split"] == "validation"


def test_max_f1_threshold_on_noisy_data_beats_a_naive_midpoint():
    rng = np.random.default_rng(0)
    n = 200
    y_true = np.array([0] * (n // 2) + [1] * (n // 2))
    scores = np.concatenate([rng.normal(0.3, 0.15, n // 2), rng.normal(0.7, 0.15, n // 2)])
    result = max_f1_threshold(y_true, scores, split="validation")
    assert result["f1"] > 0.8


def test_threshold_at_precision_achieves_at_least_the_target():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.6, 0.55, 0.7, 0.8, 0.9]
    result = threshold_at_precision(y_true, scores, min_precision=0.9, split="validation")
    assert result["precision"] >= 0.9
    assert result["min_precision"] == 0.9
    assert result["split"] == "validation"


def test_threshold_at_precision_raises_if_unreachable():
    y_true = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]  # indistinguishable -> can't hit high precision
    with pytest.raises(ValueError, match="No threshold"):
        threshold_at_precision(y_true, scores, min_precision=0.99, split="validation")


def test_percentile_threshold_matches_numpy():
    train_scores = np.arange(100, dtype=float)
    result = percentile_threshold(train_scores, q=99)
    assert result["threshold"] == pytest.approx(np.percentile(train_scores, 99))
    assert result["split"] == "train"
    assert result["percentile"] == 99
