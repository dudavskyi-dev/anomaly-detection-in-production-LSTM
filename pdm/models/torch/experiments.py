"""The P04 required experiments: window-size sweep, architecture ablation, RUL-cap ablation,
and final 5-seed stability — each evaluated on **validation only** except the final
configuration, whose headline number is scored on test through the single-use harness.
"""

import io
from pathlib import Path

import torch

from pdm.config import settings
from pdm.evaluation.harness import Split, evaluate
from pdm.evaluation.metrics import classification_metrics, regression_metrics
from pdm.evaluation.stability import aggregate_over_seeds
from pdm.models.torch.architecture import LSTMClassifier, LSTMRegressor, count_parameters
from pdm.models.torch.dataset import make_dataloader
from pdm.models.torch.train import (
    measure_inference_latency,
    set_full_determinism,
    train_classifier,
    train_regressor,
)
from pdm.preprocessing.pipeline import prepare_cmapss_subset

SWEEP_MAX_EPOCHS = 40
SWEEP_PATIENCE = 8
SWEEP_SEEDS = (0, 1)


def _train_one_rul_run(
    subset: str,
    processed_dir: Path,
    *,
    seed: int,
    window_size: int,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    rul_cap: int,
    max_epochs: int,
    patience: int,
) -> dict:
    """Fit one RUL regressor and return val-only metrics plus benchmark numbers — never touches
    the test split (sweeps are compared on validation, per the spec's explicit constraint)."""
    set_full_determinism(seed)
    ds = prepare_cmapss_subset(
        subset, processed_dir, seed=seed, window_size=window_size, rul_cap=rul_cap
    )
    n_features = len(ds.feature_spec.feature_names)

    train_loader = make_dataloader(ds.train.windows, ds.train.targets, shuffle=True, seed=seed)
    val_loader = make_dataloader(ds.val.windows, ds.val.targets, shuffle=False)

    model = LSTMRegressor(n_features=n_features, hidden_sizes=hidden_sizes, dropout=dropout)
    result = train_regressor(
        model, train_loader, val_loader, seed=seed, max_epochs=max_epochs, patience=patience
    )

    sample = torch.as_tensor(ds.val.windows[0], dtype=torch.float32)
    latency = measure_inference_latency(model, sample, n_calls=1000, n_warmup=50)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_size_bytes = buffer.tell()

    return {
        "val_rmse": result.best_val_score,
        "epochs_to_best": float(result.best_epoch),
        "train_wall_clock_seconds": result.train_wall_clock_seconds,
        "peak_rss_mb": result.peak_rss_bytes / 1e6,
        "parameter_count": float(count_parameters(model)),
        "model_size_bytes": float(model_size_bytes),
        "latency_p50_ms": latency["p50_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "latency_p99_ms": latency["p99_ms"],
        "diverged": float(result.diverged),
    }


def window_size_sweep(
    subset: str,
    processed_dir: Path,
    window_sizes: tuple[int, ...] = (10, 20, 30, 50),
    seeds: tuple[int, ...] = SWEEP_SEEDS,
) -> dict[int, dict]:
    results = {}
    for window_size in window_sizes:
        runs = [
            _train_one_rul_run(
                subset,
                processed_dir,
                seed=seed,
                window_size=window_size,
                hidden_sizes=settings.model.lstm_hidden_sizes,
                dropout=settings.model.dropout,
                rul_cap=settings.data.rul_cap,
                max_epochs=SWEEP_MAX_EPOCHS,
                patience=SWEEP_PATIENCE,
            )
            for seed in seeds
        ]
        results[window_size] = aggregate_over_seeds(runs, seeds)
    return results


ARCHITECTURE_CONFIGS: tuple[tuple[str, tuple[int, ...], float], ...] = (
    ("two_layer_with_dropout", (100, 50), 0.2),
    ("two_layer_no_dropout", (100, 50), 0.0),
    ("one_layer_with_dropout", (100,), 0.2),
    ("one_layer_no_dropout", (100,), 0.0),
)


def architecture_ablation(
    subset: str, processed_dir: Path, seeds: tuple[int, ...] = SWEEP_SEEDS
) -> dict[str, dict]:
    results = {}
    for name, hidden_sizes, dropout in ARCHITECTURE_CONFIGS:
        runs = [
            _train_one_rul_run(
                subset,
                processed_dir,
                seed=seed,
                window_size=settings.data.window_size,
                hidden_sizes=hidden_sizes,
                dropout=dropout,
                rul_cap=settings.data.rul_cap,
                max_epochs=SWEEP_MAX_EPOCHS,
                patience=SWEEP_PATIENCE,
            )
            for seed in seeds
        ]
        results[name] = aggregate_over_seeds(runs, seeds)
    return results


def rul_cap_ablation(
    subset: str, processed_dir: Path, seeds: tuple[int, ...] = SWEEP_SEEDS
) -> dict[str, dict]:
    # a cap far beyond any C-MAPSS unit's max cycle count (~543, per P01) is effectively "uncapped"
    uncapped_cap = 100_000
    conditions = {"capped": settings.data.rul_cap, "uncapped": uncapped_cap}
    results = {}
    for name, cap in conditions.items():
        runs = [
            _train_one_rul_run(
                subset,
                processed_dir,
                seed=seed,
                window_size=settings.data.window_size,
                hidden_sizes=settings.model.lstm_hidden_sizes,
                dropout=settings.model.dropout,
                rul_cap=cap,
                max_epochs=SWEEP_MAX_EPOCHS,
                patience=SWEEP_PATIENCE,
            )
            for seed in seeds
        ]
        results[name] = aggregate_over_seeds(runs, seeds)
    return results


def final_stability(subset: str, processed_dir: Path, seeds: tuple[int, ...] | None = None) -> dict:
    """The shipped configuration's headline numbers: val (for comparability with the sweeps
    above) and test (scored once per seed through the single-use harness), over the full
    5-seed configuration — never a single run."""
    seeds = settings.training.seeds if seeds is None else seeds
    val_runs = []
    test_runs = []
    for seed in seeds:
        set_full_determinism(seed)
        ds = prepare_cmapss_subset(subset, processed_dir, seed=seed)
        n_features = len(ds.feature_spec.feature_names)

        train_loader = make_dataloader(ds.train.windows, ds.train.targets, shuffle=True, seed=seed)
        val_loader = make_dataloader(ds.val.windows, ds.val.targets, shuffle=False)

        model = LSTMRegressor(
            n_features=n_features,
            hidden_sizes=settings.model.lstm_hidden_sizes,
            dropout=settings.model.dropout,
        )
        result = train_regressor(model, train_loader, val_loader, seed=seed)
        val_runs.append({"rmse": result.best_val_score, "epochs_to_best": float(result.best_epoch)})

        test_split = Split("test", ds.test.windows, ds.test.targets)

        def predict_fn(windows, _model=model):
            _model.eval()
            with torch.no_grad():
                return _model(torch.as_tensor(windows, dtype=torch.float32)).numpy()

        test_runs.append(evaluate(predict_fn, test_split, "rul", regression_metrics))

    return {
        "val": aggregate_over_seeds(val_runs, seeds),
        "test": aggregate_over_seeds(test_runs, seeds),
    }


def classifier_imbalance_comparison(
    subset: str, processed_dir: Path, seeds: tuple[int, ...] = SWEEP_SEEDS
) -> dict[str, dict]:
    """Failure-within-W classifier, with vs without ``pos_weight`` in ``BCEWithLogitsLoss``,
    on validation only — the deliverable #4 comparison, over multiple seeds like every other
    number in this project."""
    results: dict[str, list[dict]] = {"unweighted": [], "weighted": []}
    for seed in seeds:
        set_full_determinism(seed)
        ds = prepare_cmapss_subset(subset, processed_dir, seed=seed)
        n_features = len(ds.feature_spec.feature_names)
        train_targets = ds.train.targets["will_fail"]
        val_targets = ds.val.targets["will_fail"]

        train_loader = make_dataloader(ds.train.windows, ds.train.targets, shuffle=True, seed=seed)
        val_loader = make_dataloader(ds.val.windows, ds.val.targets, shuffle=False)

        n_pos = float(train_targets.sum())
        n_neg = float(len(train_targets) - n_pos)
        pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else None

        for name, weight in (("unweighted", None), ("weighted", pos_weight)):
            model = LSTMClassifier(
                n_features=n_features,
                hidden_sizes=settings.model.lstm_hidden_sizes,
                dropout=settings.model.dropout,
            )
            train_classifier(
                model,
                train_loader,
                val_loader,
                pos_weight=weight,
                seed=seed,
                max_epochs=SWEEP_MAX_EPOCHS,
                patience=SWEEP_PATIENCE,
            )
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.as_tensor(ds.val.windows, dtype=torch.float32))
            val_scores = torch.sigmoid(val_logits).numpy()
            val_pred = (val_scores >= 0.5).astype(int)
            results[name].append(classification_metrics(val_targets, val_pred, val_scores))

    return {name: aggregate_over_seeds(runs, seeds) for name, runs in results.items()}
