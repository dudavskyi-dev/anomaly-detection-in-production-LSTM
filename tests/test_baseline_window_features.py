"""Hand-crafted window feature engineering: flattening and per-sensor summary statistics."""

import numpy as np
import pytest

from pdm.models.baseline.window_features import (
    flatten_windows,
    make_baseline_features,
    window_summary_features,
)

pytestmark = pytest.mark.fast


def test_flatten_windows_shape():
    windows = np.zeros((5, 10, 3))
    assert flatten_windows(windows).shape == (5, 30)


def test_window_summary_features_hand_computed():
    windows = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(1, 5, 1)
    mean, std, mn, mx, slope = window_summary_features(windows)[0]
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(np.std([1, 2, 3, 4, 5]))
    assert mn == pytest.approx(1.0)
    assert mx == pytest.approx(5.0)
    assert slope == pytest.approx(1.0)  # perfectly linear, +1 per step


def test_window_summary_features_constant_window():
    windows = np.full((1, 5, 1), 7.0)
    mean, std, mn, mx, slope = window_summary_features(windows)[0]
    assert (mean, std, mn, mx, slope) == pytest.approx((7.0, 0.0, 7.0, 7.0, 0.0))


def test_make_baseline_features_concatenates_flatten_and_summary():
    windows = np.random.default_rng(0).normal(size=(4, 6, 2))
    features = make_baseline_features(windows)
    assert features.shape == (4, 6 * 2 + 2 * 5)
