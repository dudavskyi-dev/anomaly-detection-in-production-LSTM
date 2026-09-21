"""The drift-check batch job's Prometheus textfile output (P09 deliverable #4)."""

import pytest

from pdm.models.bundle import compute_reference_stats
from pdm.monitoring.drift import compute_input_drift
from pdm.monitoring.metrics import write_drift_metrics

pytestmark = pytest.mark.fast


def test_write_drift_metrics_writes_a_valid_textfile(tmp_path):
    import numpy as np

    rng = np.random.default_rng(0)
    train_windows = rng.normal(size=(50, 5, 2)).astype(np.float32)
    reference_stats = compute_reference_stats(train_windows, ["a", "b"])
    production = rng.normal(size=(20, 5, 2)).astype(np.float32)
    input_drift = compute_input_drift(reference_stats, production, ["a", "b"])

    path = tmp_path / "drift_metrics.prom"
    write_drift_metrics(
        target="bundle::FD002:test",
        input_drift=input_drift,
        triggered=True,
        cooldown_active=False,
        path=path,
    )

    text = path.read_text(encoding="utf-8")
    assert "pdm_drift_psi_mean" in text
    assert 'target="bundle::FD002:test"' in text
    assert "pdm_drift_retrain_triggered" in text
    # The exact names P10's spec asks for, including a per-feature label.
    assert "pdm_drift_score_aggregate" in text
    assert 'pdm_drift_score{feature="a"' in text
    assert 'pdm_drift_score{feature="b"' in text
