"""RUL regression baselines: dummy floors match hand-computed RMSE, and real models beat them
on learnable synthetic data."""

import numpy as np
import pytest

from pdm.models.baseline.regression import (
    dummy_cap_baseline,
    dummy_mean_baseline,
    random_forest_regressor_baseline,
    ridge_baseline,
)

pytestmark = pytest.mark.fast


def _make_synthetic_rul_windows(n=200, window_size=10, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    windows = rng.normal(size=(n, window_size, n_features))
    targets = windows.mean(axis=(1, 2)) * 50 + 100 + rng.normal(0, 1, n)
    return windows, targets


def test_dummy_mean_baseline_matches_hand_computed_rmse():
    train_targets = np.array([10.0, 20.0, 30.0])
    eval_targets = np.array([10.0, 20.0, 30.0])
    result = dummy_mean_baseline(train_targets, eval_targets)
    assert result["rmse"] == pytest.approx(np.sqrt((100 + 0 + 100) / 3))


def test_dummy_cap_baseline_zero_error_when_targets_equal_cap():
    eval_targets = np.array([125.0, 125.0])
    result = dummy_cap_baseline(eval_targets, cap=125)
    assert result["rmse"] == pytest.approx(0.0)


def test_ridge_and_random_forest_beat_dummy_mean_on_learnable_data():
    train_windows, train_targets = _make_synthetic_rul_windows(seed=0)
    eval_windows, eval_targets = _make_synthetic_rul_windows(seed=1)

    dummy = dummy_mean_baseline(train_targets, eval_targets)
    ridge = ridge_baseline(train_windows, train_targets, eval_windows, eval_targets, seed=0)
    rf = random_forest_regressor_baseline(
        train_windows, train_targets, eval_windows, eval_targets, seed=0
    )

    assert ridge["rmse"] < dummy["rmse"]
    assert rf["rmse"] < dummy["rmse"]
