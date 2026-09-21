"""Runs every classical baseline over the configured seeds, writing per-run ``metrics.json``
files under ``artifacts/baselines/`` — the only input ``docs/RESULTS.md`` is generated from —
and logging one MLflow run per (model, seed) through ``pdm.tracking.mlflow_client`` (P07: no
training path may write metrics only to stdout).

Validation metrics may be computed freely (that's what they're for — model comparison and
threshold selection). Test metrics are scored **once per seed** through
:func:`pdm.evaluation.harness.evaluate`, using a threshold chosen on that seed's validation
split, never on test.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from pdm.config import settings
from pdm.evaluation.harness import Split, evaluate
from pdm.evaluation.metrics import classification_metrics, regression_metrics
from pdm.evaluation.stability import aggregate_over_seeds
from pdm.evaluation.thresholds import max_f1_threshold, threshold_at_precision
from pdm.models.baseline import ai4i as ai4i_baselines
from pdm.models.baseline.window_features import make_baseline_features
from pdm.preprocessing.pipeline import prepare_cmapss_subset
from pdm.tracking.mlflow_client import log_metrics, log_params, sha256_of_files, start_run

ARTIFACTS_DIR = Path("artifacts/baselines")
RUL_SUBSET = "FD001"
RUL_MODELS = ("dummy_mean", "dummy_cap", "ridge", "random_forest")
FAILURE_MODELS = ("dummy", "logistic_regression", "random_forest")
AI4I_MODELS = ("dummy", "logistic_regression", "random_forest")


def _write_metrics(run_id: str, payload: dict) -> Path:
    out_dir = ARTIFACTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _cmapss_dataset_hash(processed_dir: Path, subset: str) -> str:
    cmapss_dir = processed_dir / "cmapss"
    return sha256_of_files(
        [cmapss_dir / f"train_{subset}.parquet", cmapss_dir / f"test_{subset}.parquet"]
    )


# --- RUL regression -----------------------------------------------------------------------------


def _fit_predict_rul(
    model_name: str, train_windows: np.ndarray, train_targets: np.ndarray, seed: int
) -> Callable[[np.ndarray], np.ndarray]:
    if model_name == "dummy_mean":
        mean_value = float(np.mean(train_targets))
        return lambda w: np.full(w.shape[0], mean_value, dtype=float)
    if model_name == "dummy_cap":
        cap = float(settings.data.rul_cap)
        return lambda w: np.full(w.shape[0], cap, dtype=float)

    x_train = make_baseline_features(train_windows)
    if model_name == "ridge":
        model = Ridge(alpha=settings.training.baseline_ridge_alpha, random_state=seed)
    elif model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=settings.training.baseline_rf_n_estimators,
            max_depth=settings.training.baseline_rf_max_depth,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"unknown RUL model {model_name!r}")
    model.fit(x_train, train_targets)
    return lambda w: model.predict(make_baseline_features(w))


def run_rul_regression_baselines(processed_dir: Path, subset: str = RUL_SUBSET) -> dict[str, Path]:
    written = {}
    dataset_hash = _cmapss_dataset_hash(processed_dir, subset)

    for model_name in RUL_MODELS:
        val_results = []
        test_results = []
        mlflow_run_ids: dict[int, str] = {}

        for seed in settings.training.seeds:
            ds = prepare_cmapss_subset(subset, processed_dir, seed=seed)
            predict_fn = _fit_predict_rul(
                model_name, ds.train.windows, ds.train.targets["rul"], seed
            )

            val_metrics = regression_metrics(ds.val.targets["rul"], predict_fn(ds.val.windows))
            val_results.append(val_metrics)

            test_split = Split("test", ds.test.windows, ds.test.targets)
            test_metrics = evaluate(predict_fn, test_split, "rul", regression_metrics)
            test_results.append(test_metrics)

            with start_run(
                run_name=f"rul_regression__{model_name}__{subset}__seed{seed}",
                framework="sklearn",
                seed=seed,
                dataset_hash=dataset_hash,
                extra_tags={"task": "rul_regression", "model": model_name, "subset": subset},
            ) as mlflow_run:
                log_params({"model": model_name, "subset": subset})
                log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
                log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                mlflow_run_ids[seed] = mlflow_run.info.run_id

        val_summary = aggregate_over_seeds(val_results, settings.training.seeds)
        test_summary = aggregate_over_seeds(test_results, settings.training.seeds)

        run_id = f"rul_regression__{model_name}__{subset}"
        payload = {
            "run_id": run_id,
            "task": "rul_regression",
            "model": model_name,
            "subset": subset,
            "seeds": list(settings.training.seeds),
            "val": val_summary,
            "test": test_summary,
            "mlflow_run_ids": mlflow_run_ids,
        }
        written[run_id] = _write_metrics(run_id, payload)
    return written


# --- Failure-within-W classification -------------------------------------------------------------


def _fit_predict_score_failure(
    model_name: str, train_windows: np.ndarray, train_targets: np.ndarray, seed: int
) -> Callable[[np.ndarray], np.ndarray]:
    if model_name == "dummy":
        x_train = np.zeros((len(train_targets), 1))
        model = DummyClassifier(strategy="most_frequent", random_state=seed)
        model.fit(x_train, train_targets)
        return lambda w: model.predict_proba(np.zeros((w.shape[0], 1)))[:, 1]

    x_train = make_baseline_features(train_windows)
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=settings.training.baseline_rf_n_estimators,
            max_depth=settings.training.baseline_rf_max_depth,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"unknown failure-classification model {model_name!r}")
    model.fit(x_train, train_targets)
    return lambda w: model.predict_proba(make_baseline_features(w))[:, 1]


def run_failure_classification_baselines(
    processed_dir: Path, subset: str = RUL_SUBSET
) -> dict[str, Path]:
    written = {}
    dataset_hash = _cmapss_dataset_hash(processed_dir, subset)

    for model_name in FAILURE_MODELS:
        val_results = []
        test_results = []
        threshold_examples = []
        mlflow_run_ids: dict[int, str] = {}

        for seed in settings.training.seeds:
            ds = prepare_cmapss_subset(subset, processed_dir, seed=seed)
            train_targets = ds.train.targets["will_fail"]
            val_targets = ds.val.targets["will_fail"]
            score_fn = _fit_predict_score_failure(model_name, ds.train.windows, train_targets, seed)

            val_scores = score_fn(ds.val.windows)
            val_pred_default = (val_scores >= 0.5).astype(int)
            val_metrics = classification_metrics(val_targets, val_pred_default, val_scores)
            val_results.append(val_metrics)

            max_f1 = max_f1_threshold(val_targets, val_scores, split="validation")
            try:
                at_precision = threshold_at_precision(
                    val_targets,
                    val_scores,
                    settings.training.baseline_precision_target,
                    split="validation",
                )
            except ValueError as exc:
                # a weak model may genuinely be unable to reach the target precision at all —
                # that's a real, informative result, not a bug to hide by lowering the target.
                at_precision = {
                    "threshold": None,
                    "reachable": False,
                    "reason": str(exc),
                    "split": "validation",
                    "min_precision": settings.training.baseline_precision_target,
                }
            threshold_examples.append(
                {"seed": seed, "max_f1": max_f1, "at_precision": at_precision}
            )

            test_split = Split("test", ds.test.windows, ds.test.targets)

            def _metric_fn(y_true, y_score, _threshold: float = max_f1["threshold"]) -> dict:
                y_pred = (y_score >= _threshold).astype(int)
                return classification_metrics(y_true, y_pred, y_score)

            test_metrics = evaluate(score_fn, test_split, "will_fail", _metric_fn)
            test_results.append(test_metrics)

            with start_run(
                run_name=f"failure_classification__{model_name}__{subset}__seed{seed}",
                framework="sklearn",
                seed=seed,
                dataset_hash=dataset_hash,
                extra_tags={
                    "task": "failure_classification",
                    "model": model_name,
                    "subset": subset,
                },
            ) as mlflow_run:
                log_params({"model": model_name, "subset": subset})
                log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
                log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                mlflow_run_ids[seed] = mlflow_run.info.run_id

        val_summary = aggregate_over_seeds(val_results, settings.training.seeds)
        test_summary = aggregate_over_seeds(test_results, settings.training.seeds)

        run_id = f"failure_classification__{model_name}__{subset}"
        payload = {
            "run_id": run_id,
            "task": "failure_classification",
            "model": model_name,
            "subset": subset,
            "seeds": list(settings.training.seeds),
            "val": val_summary,
            "test": test_summary,
            "threshold_examples": threshold_examples,
            "mlflow_run_ids": mlflow_run_ids,
        }
        written[run_id] = _write_metrics(run_id, payload)
    return written


# --- AI4I -----------------------------------------------------------------------------------------


def run_ai4i_baselines(processed_dir: Path) -> dict[str, Path]:
    written = {}
    ai4i_path = processed_dir / "ai4i.parquet"
    ai4i_df = pd.read_parquet(ai4i_path)
    dataset_hash = sha256_of_files([ai4i_path])

    for model_name in AI4I_MODELS:
        val_results = []
        mlflow_run_ids: dict[int, str] = {}

        for seed in settings.training.seeds:
            train_df, val_df = ai4i_baselines.prepare_ai4i_splits(ai4i_df, seed=seed)
            if model_name == "dummy":
                val_metrics = ai4i_baselines.dummy_ai4i_baseline(train_df, val_df, seed=seed)
            elif model_name == "logistic_regression":
                val_metrics = ai4i_baselines.logistic_regression_ai4i_baseline(
                    train_df, val_df, seed=seed
                )
            else:
                val_metrics = ai4i_baselines.random_forest_ai4i_baseline(
                    train_df, val_df, seed=seed
                )
            val_results.append(val_metrics)

            with start_run(
                run_name=f"ai4i__{model_name}__seed{seed}",
                framework="sklearn",
                seed=seed,
                dataset_hash=dataset_hash,
                extra_tags={"task": "ai4i_failure_classification", "model": model_name},
            ) as mlflow_run:
                log_params({"model": model_name})
                log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
                mlflow_run_ids[seed] = mlflow_run.info.run_id

        val_summary = aggregate_over_seeds(val_results, settings.training.seeds)

        run_id = f"ai4i__{model_name}"
        payload = {
            "run_id": run_id,
            "task": "ai4i_failure_classification",
            "model": model_name,
            "subset": "ai4i",
            "seeds": list(settings.training.seeds),
            "val": val_summary,
            "mlflow_run_ids": mlflow_run_ids,
        }
        written[run_id] = _write_metrics(run_id, payload)
    return written


def run_all_baselines(processed_dir: Path) -> dict[str, Path]:
    written: dict[str, Path] = {}
    written.update(run_rul_regression_baselines(processed_dir))
    written.update(run_failure_classification_baselines(processed_dir))
    written.update(run_ai4i_baselines(processed_dir))
    return written
