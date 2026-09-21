"""Seed-stability aggregation: mean/std/values are computed correctly and every metric key is
carried through, so no headline number can quietly become a single-run number."""

import pytest

from pdm.evaluation.stability import aggregate_over_seeds, run_seeds

pytestmark = pytest.mark.fast


def test_run_seeds_calls_once_per_seed_and_aggregates():
    calls = []

    def train_fn(seed: int) -> dict:
        calls.append(seed)
        return {"rmse": float(seed), "mae": float(seed) * 2}

    summary = run_seeds(train_fn, seeds=(0, 1, 2, 3, 4))

    assert calls == [0, 1, 2, 3, 4]
    assert set(summary) == {"rmse", "mae"}
    assert summary["rmse"]["mean"] == pytest.approx(2.0)  # mean of 0,1,2,3,4
    assert summary["rmse"]["values"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert summary["mae"]["mean"] == pytest.approx(4.0)


def test_run_seeds_uses_config_default_seeds_when_none_given():
    from pdm.config import settings

    seen = []

    def train_fn(seed: int) -> dict:
        seen.append(seed)
        return {"x": 1.0}

    run_seeds(train_fn)
    assert seen == list(settings.training.seeds)


def test_aggregate_over_seeds_matches_manual_computation():
    results = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}]
    summary = aggregate_over_seeds(results, seeds=(10, 20, 30))
    assert summary["a"]["mean"] == pytest.approx(2.0)
    assert summary["a"]["std"] == pytest.approx(0.816496580927726)
    assert summary["a"]["seeds"] == [10, 20, 30]
