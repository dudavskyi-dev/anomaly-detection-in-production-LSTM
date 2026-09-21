"""Sweep orchestration for the bottleneck ablation and Isolation Forest contamination sweep:
tested by monkeypatching the per-config trainer, the same pattern
``tests/test_torch_experiments.py`` uses for P04's sweeps — real training is exercised for real
in the actual experiment run (see ``docs/decisions/P06-anomaly.md``), not re-run here."""

import numpy as np
import pytest

import pdm.models.torch.anomaly_experiments as anomaly_experiments

pytestmark = pytest.mark.fast


def test_healthy_mask_uses_the_configured_threshold():
    targets = {"rul": np.array([50.0, 125.0, 10.0, 125.0])}
    mask = anomaly_experiments._healthy_mask(targets)
    np.testing.assert_array_equal(mask, [False, True, False, True])


def test_is_anomaly_labels_is_the_complement_of_healthy():
    targets = {"rul": np.array([50.0, 125.0, 10.0, 125.0])}
    labels = anomaly_experiments._is_anomaly_labels(targets)
    np.testing.assert_array_equal(labels, [1, 0, 1, 0])


def test_numeric_only_drops_non_numeric_fields():
    d = {"threshold": 0.5, "split": "validation", "f1": 0.8, "reachable": True}
    assert anomaly_experiments._numeric_only(d) == {"threshold": 0.5, "f1": 0.8}


def _fake_bottleneck_run(latent_dim, seed, subset, processed_dir):
    # encodes latent_dim and seed into the "f1" so the test can check both were threaded through
    return {"f1": float(latent_dim) + float(seed) / 10, "precision": 1.0, "recall": 1.0}


def test_bottleneck_ablation_covers_every_configured_dim_and_aggregates_by_it(monkeypatch):
    monkeypatch.setattr(anomaly_experiments, "_run_one_bottleneck_config", _fake_bottleneck_run)
    results = anomaly_experiments.bottleneck_ablation(seeds=(0, 1))
    assert set(results) == set(anomaly_experiments.BOTTLENECK_DIMS)
    smallest, largest = min(anomaly_experiments.BOTTLENECK_DIMS), max(
        anomaly_experiments.BOTTLENECK_DIMS
    )
    assert results[largest]["f1"]["mean"] > results[smallest]["f1"]["mean"]
    assert results[smallest]["f1"]["seeds"] == [0, 1]


def _fake_contamination_run(contamination, seed, subset, processed_dir):
    return {"f1": contamination * 10 + seed, "precision": 1.0, "recall": 1.0}


def test_contamination_sweep_covers_every_grid_point(monkeypatch):
    monkeypatch.setattr(
        anomaly_experiments, "_run_one_contamination_config", _fake_contamination_run
    )
    results = anomaly_experiments.isolation_forest_contamination_sweep(seeds=(0,))
    assert set(results) == {str(c) for c in anomaly_experiments.CONTAMINATION_GRID}
