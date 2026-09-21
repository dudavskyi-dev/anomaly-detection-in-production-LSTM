"""Isolation Forest anomaly baseline: fit on healthy-only data, and the higher-is-more-anomalous
score convention it shares with the LSTM autoencoder for fusion."""

import numpy as np
import pytest

from pdm.models.baseline.isolation_forest import fit_isolation_forest, isolation_forest_scores

pytestmark = pytest.mark.fast


def test_anomalous_windows_score_higher_than_healthy_ones():
    rng = np.random.default_rng(0)
    healthy = rng.normal(0, 1, size=(200, 5, 3)).astype(np.float32)
    anomalous = rng.normal(8, 1, size=(20, 5, 3)).astype(np.float32)

    model = fit_isolation_forest(healthy, seed=0)
    healthy_scores = isolation_forest_scores(model, healthy)
    anomalous_scores = isolation_forest_scores(model, anomalous)

    assert anomalous_scores.mean() > healthy_scores.mean()


def test_contamination_changes_the_fitted_model():
    rng = np.random.default_rng(0)
    healthy = rng.normal(0, 1, size=(100, 5, 2)).astype(np.float32)
    low = fit_isolation_forest(healthy, seed=0, contamination=0.05)
    high = fit_isolation_forest(healthy, seed=0, contamination=0.3)
    assert low.offset_ != high.offset_
