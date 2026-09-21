"""Sweep orchestration logic (window size / architecture / RUL cap), tested by monkeypatching
the underlying per-run trainer — real training is exercised for real in the actual experiment
run (see docs/decisions/P04-pytorch.md), not re-run here where it would make the suite minutes
slower for no additional coverage of the orchestration logic itself.
"""

import pytest

import pdm.models.torch.experiments as experiments_module

pytestmark = pytest.mark.fast


def _fake_train_one_rul_run(
    subset,
    processed_dir,
    *,
    seed,
    window_size,
    hidden_sizes,
    dropout,
    rul_cap,
    max_epochs,
    patience,
):
    # a fast, deterministic stand-in: encodes every input that should vary the "result" so the
    # test can check the right values were threaded through to the right call.
    return {
        "val_rmse": float(window_size) + float(seed) + len(hidden_sizes) + dropout + rul_cap / 1000,
        "epochs_to_best": 1.0,
        "train_wall_clock_seconds": 0.01,
        "peak_rss_mb": 1.0,
        "parameter_count": 100.0,
        "latency_p50_ms": 0.1,
        "latency_p95_ms": 0.2,
        "latency_p99_ms": 0.3,
        "diverged": 0.0,
    }


def test_window_size_sweep_passes_window_size_through_and_aggregates_by_it(monkeypatch):
    monkeypatch.setattr(experiments_module, "_train_one_rul_run", _fake_train_one_rul_run)
    results = experiments_module.window_size_sweep(
        "FD001", "unused", window_sizes=(10, 20), seeds=(0, 1)
    )
    assert set(results) == {10, 20}
    # val_rmse for window_size=10 averaged over seed 0 and 1 differs by exactly the window size
    assert results[20]["val_rmse"]["mean"] - results[10]["val_rmse"]["mean"] == pytest.approx(10.0)
    assert results[10]["val_rmse"]["seeds"] == [0, 1]


def test_architecture_ablation_covers_all_four_named_configs(monkeypatch):
    monkeypatch.setattr(experiments_module, "_train_one_rul_run", _fake_train_one_rul_run)
    results = experiments_module.architecture_ablation("FD001", "unused", seeds=(0,))
    assert set(results) == {
        "two_layer_with_dropout",
        "two_layer_no_dropout",
        "one_layer_with_dropout",
        "one_layer_no_dropout",
    }
    # two-layer configs show a higher "val_rmse" than one-layer here (the len(hidden_sizes) term)
    assert (
        results["two_layer_with_dropout"]["val_rmse"]["mean"]
        > results["one_layer_with_dropout"]["val_rmse"]["mean"]
    )


def test_rul_cap_ablation_covers_capped_and_uncapped(monkeypatch):
    monkeypatch.setattr(experiments_module, "_train_one_rul_run", _fake_train_one_rul_run)
    results = experiments_module.rul_cap_ablation("FD001", "unused", seeds=(0,))
    assert set(results) == {"capped", "uncapped"}
    # uncapped uses a much larger rul_cap (100_000 vs 125), which the fake linearly reflects
    assert results["uncapped"]["val_rmse"]["mean"] > results["capped"]["val_rmse"]["mean"]
