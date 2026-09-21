"""Head-to-head PyTorch vs TensorFlow benchmark (P05, spec §6.4).

Both frameworks train the same architecture (parameter-count-identical, see
``pdm.models.tf.architecture``), with the same hyperparameters, same seeds, and the same
preprocessed windows/targets from ``pdm.preprocessing.pipeline.prepare_cmapss_subset`` — called
**once per seed** and its arrays handed to both frameworks unchanged, so neither side
re-derives its own view of the data (deliverable #3).
"""

import io
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from tensorflow import keras

from pdm.config import settings
from pdm.evaluation.harness import Split, evaluate
from pdm.evaluation.metrics import classification_metrics, regression_metrics
from pdm.evaluation.stability import aggregate_over_seeds
from pdm.models.tf.architecture import LSTMClassifier as TFClassifier
from pdm.models.tf.architecture import LSTMRegressor as TFRegressor
from pdm.models.tf.architecture import count_parameters as tf_count_parameters
from pdm.models.tf.dataset import make_tf_dataset
from pdm.models.tf.train import TrainResult as TFTrainResult
from pdm.models.tf.train import measure_inference_latency as tf_measure_latency
from pdm.models.tf.train import set_full_determinism as tf_set_seed
from pdm.models.tf.train import train_classifier as tf_train_classifier
from pdm.models.tf.train import train_regressor as tf_train_regressor
from pdm.models.torch.architecture import LSTMClassifier as TorchClassifier
from pdm.models.torch.architecture import LSTMRegressor as TorchRegressor
from pdm.models.torch.architecture import count_parameters as torch_count_parameters
from pdm.models.torch.dataset import make_dataloader
from pdm.models.torch.train import TrainResult as TorchTrainResult
from pdm.models.torch.train import measure_inference_latency as torch_measure_latency
from pdm.models.torch.train import set_full_determinism as torch_set_seed
from pdm.models.torch.train import train_classifier as torch_train_classifier
from pdm.models.torch.train import train_regressor as torch_train_regressor
from pdm.preprocessing.pipeline import PreparedDataset, prepare_cmapss_subset
from pdm.tracking.mlflow_client import log_metrics, log_params, sha256_of_files, start_run

FRAMEWORKS: tuple[str, ...] = ("pytorch", "tensorflow")

# Both frameworks' training loops return a dataclass with the same field names (best_epoch,
# best_val_score, train_wall_clock_seconds, peak_rss_bytes, diverged) by design — see
# pdm.models.tf.train's module docstring — so callers below read either through this alias
# rather than mypy narrowing `result`'s type to whichever branch assigned it first.
_TrainResult = TorchTrainResult | TFTrainResult


def _tf_model_size_bytes(model: keras.Model) -> int:
    """Serialized size of the model's own weights only — comparable to
    ``torch.save(model.state_dict())``'s pickle, which never includes optimizer state.

    ``model.save_weights()`` looked like the obvious choice here, but once a model has been
    compiled and trained it also serialises the **optimizer's** state (Adam's two moment
    estimates per parameter) into the same ``.weights.h5`` file — inflating a compiled model's
    "size on disk" to roughly 3x the raw parameter bytes for no reason a real deployment would
    ever pay (nothing ships an optimizer with an inference model). Confirmed by inspecting the
    saved file's HDF5 groups directly: an ``optimizer/vars/...`` group sized almost exactly 2x
    the model's own weights sat alongside the real ``layers/...`` weights. Using the public
    ``model.get_weights()`` (model-only, no optimizer) serialised through ``numpy.savez`` avoids
    the private saving internals while still measuring a real container format's overhead, not
    just summing raw array bytes. See ``docs/decisions/P05-tensorflow.md``.
    """
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "weights.npz"
        np.savez(path, *model.get_weights())
        return path.stat().st_size


def _train_one_regressor(
    framework: str,
    ds: PreparedDataset,
    *,
    seed: int,
    n_features: int,
    subset: str | None = None,
    dataset_hash: str | None = None,
) -> dict:
    """Train one RUL regressor in ``framework`` and return flat benchmark + test metrics.

    Scores test **once** through the single-use ``Split``/``evaluate`` harness, mirroring
    ``pdm.models.torch.experiments.final_stability``.
    """
    result: _TrainResult
    if framework == "pytorch":
        torch_set_seed(seed)
        train_loader = make_dataloader(ds.train.windows, ds.train.targets, shuffle=True, seed=seed)
        val_loader = make_dataloader(ds.val.windows, ds.val.targets, shuffle=False)
        model = TorchRegressor(
            n_features=n_features,
            hidden_sizes=settings.model.lstm_hidden_sizes,
            dropout=settings.model.dropout,
        )
        result = torch_train_regressor(model, train_loader, val_loader, seed=seed)
        sample = torch.as_tensor(ds.val.windows[0], dtype=torch.float32)
        latency = torch_measure_latency(model, sample, n_calls=1000, n_warmup=50)
        model_size_bytes = _torch_model_size_bytes(model)
        parameter_count = torch_count_parameters(model)

        def predict_fn(windows: np.ndarray) -> np.ndarray:
            model.eval()
            with torch.no_grad():
                return model(torch.as_tensor(windows, dtype=torch.float32)).numpy()

    elif framework == "tensorflow":
        tf_set_seed(seed)
        train_ds = make_tf_dataset(
            ds.train.windows, ds.train.targets, "rul", shuffle=True, seed=seed
        )
        val_ds = make_tf_dataset(ds.val.windows, ds.val.targets, "rul", shuffle=False)
        model = TFRegressor(
            n_features=n_features,
            hidden_sizes=settings.model.lstm_hidden_sizes,
            dropout=settings.model.dropout,
        )
        result = tf_train_regressor(model, train_ds, val_ds, seed=seed)
        latency = tf_measure_latency(model, ds.val.windows[0], n_calls=1000, n_warmup=50)
        model_size_bytes = _tf_model_size_bytes(model)
        parameter_count = tf_count_parameters(model)

        def predict_fn(windows: np.ndarray) -> np.ndarray:
            return model(tf.convert_to_tensor(windows, dtype=tf.float32), training=False).numpy()

    else:
        raise ValueError(f"unknown framework {framework!r}")

    test_split = Split("test", ds.test.windows, ds.test.targets)
    test_metrics = evaluate(predict_fn, test_split, "rul", regression_metrics)

    metrics = {
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_nasa_score": test_metrics["nasa_score"],
        "val_rmse": result.best_val_score,
        "train_wall_clock_seconds": result.train_wall_clock_seconds,
        "epochs_to_best": float(result.best_epoch),
        "peak_rss_mb": result.peak_rss_bytes / 1e6,
        "parameter_count": float(parameter_count),
        "model_size_bytes": float(model_size_bytes),
        "latency_p50_ms": latency["p50_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "latency_p99_ms": latency["p99_ms"],
        "diverged": float(result.diverged),
    }
    with start_run(
        run_name=f"framework_benchmark__regression__{framework}__seed{seed}",
        framework=framework,
        seed=seed,
        dataset_hash=dataset_hash,
        extra_tags={"task": "rul_regression", "subset": subset, "milestone": "P05"},
    ):
        log_params(
            {
                "framework": framework,
                "subset": subset,
                "n_features": n_features,
                "hidden_sizes": settings.model.lstm_hidden_sizes,
                "dropout": settings.model.dropout,
            }
        )
        log_metrics(metrics)
    return metrics


def _torch_model_size_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def _train_one_classifier(
    framework: str,
    ds: PreparedDataset,
    *,
    seed: int,
    n_features: int,
    subset: str | None = None,
    dataset_hash: str | None = None,
) -> dict:
    """Train one failure-within-W classifier (unweighted) in ``framework`` and return flat
    benchmark + test metrics — the PR-AUC/F1 half of spec §6.4's required comparison table."""
    result: _TrainResult
    if framework == "pytorch":
        torch_set_seed(seed)
        train_loader = make_dataloader(ds.train.windows, ds.train.targets, shuffle=True, seed=seed)
        val_loader = make_dataloader(ds.val.windows, ds.val.targets, shuffle=False)
        model = TorchClassifier(
            n_features=n_features,
            hidden_sizes=settings.model.lstm_hidden_sizes,
            dropout=settings.model.dropout,
        )
        result = torch_train_classifier(model, train_loader, val_loader, seed=seed)

        def predict_fn(windows: np.ndarray) -> np.ndarray:
            model.eval()
            with torch.no_grad():
                logits = model(torch.as_tensor(windows, dtype=torch.float32))
                return torch.sigmoid(logits).numpy()

    elif framework == "tensorflow":
        tf_set_seed(seed)
        train_ds = make_tf_dataset(
            ds.train.windows, ds.train.targets, "will_fail", shuffle=True, seed=seed
        )
        val_ds = make_tf_dataset(ds.val.windows, ds.val.targets, "will_fail", shuffle=False)
        model = TFClassifier(
            n_features=n_features,
            hidden_sizes=settings.model.lstm_hidden_sizes,
            dropout=settings.model.dropout,
        )
        result = tf_train_classifier(model, train_ds, val_ds, seed=seed)

        def predict_fn(windows: np.ndarray) -> np.ndarray:
            logits = model(tf.convert_to_tensor(windows, dtype=tf.float32), training=False)
            return tf.sigmoid(logits).numpy()

    else:
        raise ValueError(f"unknown framework {framework!r}")

    def _classification_metric_fn(y_true: np.ndarray, y_score: np.ndarray) -> dict:
        y_pred = (y_score >= 0.5).astype(int)
        return classification_metrics(y_true, y_pred, y_score)

    test_split = Split("test", ds.test.windows, ds.test.targets)
    test_metrics = evaluate(predict_fn, test_split, "will_fail", _classification_metric_fn)

    metrics = {
        "test_pr_auc": test_metrics["pr_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "train_wall_clock_seconds": result.train_wall_clock_seconds,
        "epochs_to_best": float(result.best_epoch),
        "diverged": float(result.diverged),
    }
    with start_run(
        run_name=f"framework_benchmark__classification__{framework}__seed{seed}",
        framework=framework,
        seed=seed,
        dataset_hash=dataset_hash,
        extra_tags={"task": "failure_classification", "subset": subset, "milestone": "P05"},
    ):
        log_params(
            {
                "framework": framework,
                "subset": subset,
                "n_features": n_features,
                "hidden_sizes": settings.model.lstm_hidden_sizes,
                "dropout": settings.model.dropout,
            }
        )
        log_metrics(metrics)
    return metrics


def time_input_pipeline(loader_or_dataset, n_epochs: int = 3) -> float:
    """Iterate ``loader_or_dataset`` (a PyTorch ``DataLoader`` or a ``tf.data.Dataset``) end to
    end ``n_epochs`` times with **no model involved**, isolating the input pipeline's own
    wall-clock cost from any framework op or training-loop cost — answers "is a train-time
    difference attributable to the framework or the data pipeline?"
    """
    start = time.perf_counter()
    for _ in range(n_epochs):
        for _ in loader_or_dataset:
            pass
    return time.perf_counter() - start


def compare_input_pipeline_timing(ds: PreparedDataset, *, seed: int, n_epochs: int = 3) -> dict:
    train_loader = make_dataloader(ds.train.windows, ds.train.targets, shuffle=True, seed=seed)
    train_tf_ds = make_tf_dataset(
        ds.train.windows, ds.train.targets, "rul", shuffle=True, seed=seed
    )
    return {
        "pytorch_seconds": time_input_pipeline(train_loader, n_epochs=n_epochs),
        "tensorflow_seconds": time_input_pipeline(train_tf_ds, n_epochs=n_epochs),
        "n_epochs": n_epochs,
        "n_train_windows": int(ds.train.windows.shape[0]),
    }


def export_and_measure_savedmodel(
    model: keras.Model,
    sample_window: np.ndarray,
    *,
    window_size: int,
    n_features: int,
    n_calls: int = 1000,
    n_warmup: int = 50,
) -> dict:
    """Export ``model`` to SavedModel format and measure single-sample inference latency
    through the reloaded, exported model — deliverable #6, "is SavedModel inference actually
    faster than eager PyTorch, and by how much."

    Keras 3's ``Model.export()`` traces a serving signature from whatever input shape the model
    was **last called with**, not a fully dynamic one — call the model once with the real
    serving shape (``window_size``) before exporting, or it silently freezes an unrelated
    shape (e.g. the dummy shape used at construction to make parameters countable). See
    ``docs/decisions/P05-tensorflow.md``.
    """
    model(tf.zeros((1, window_size, n_features)), training=False)  # retrace with the real shape

    with tempfile.TemporaryDirectory() as d:
        export_dir = Path(d) / "saved_model"
        model.export(str(export_dir))
        export_size_bytes = sum(f.stat().st_size for f in export_dir.rglob("*") if f.is_file())

        loaded = tf.saved_model.load(str(export_dir))
        infer = loaded.signatures["serving_default"]
        input_name = next(iter(infer.structured_input_signature[1]))

        single = sample_window[np.newaxis, ...] if sample_window.ndim == 2 else sample_window
        single_t = tf.convert_to_tensor(single, dtype=tf.float32)

        for _ in range(n_warmup):
            infer(**{input_name: single_t})

        timings_ms = []
        for _ in range(n_calls):
            start = time.perf_counter()
            infer(**{input_name: single_t})
            timings_ms.append((time.perf_counter() - start) * 1000)

    arr = np.array(timings_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "n_calls": n_calls,
        "export_size_bytes": export_size_bytes,
    }


def run_framework_benchmark(
    subset: str, processed_dir: Path, seeds: tuple[int, ...] | None = None
) -> dict:
    """The full P05 benchmark: for each seed, prepare one dataset and train both frameworks'
    regressor and classifier on the identical arrays, then aggregate mean/std over seeds —
    never fewer than the project's full 5-seed configuration for a reported headline number.
    """
    seeds = settings.training.seeds if seeds is None else seeds
    cmapss_dir = processed_dir / "cmapss"
    dataset_hash = sha256_of_files(
        [cmapss_dir / f"train_{subset}.parquet", cmapss_dir / f"test_{subset}.parquet"]
    )

    regression_runs: dict[str, list[dict]] = {fw: [] for fw in FRAMEWORKS}
    classification_runs: dict[str, list[dict]] = {fw: [] for fw in FRAMEWORKS}
    pipeline_timings: list[dict] = []
    export_results: dict | None = None

    for seed in seeds:
        ds = prepare_cmapss_subset(subset, processed_dir, seed=seed)
        n_features = len(ds.feature_spec.feature_names)

        for framework in FRAMEWORKS:
            regression_runs[framework].append(
                _train_one_regressor(
                    framework,
                    ds,
                    seed=seed,
                    n_features=n_features,
                    subset=subset,
                    dataset_hash=dataset_hash,
                )
            )
            classification_runs[framework].append(
                _train_one_classifier(
                    framework,
                    ds,
                    seed=seed,
                    n_features=n_features,
                    subset=subset,
                    dataset_hash=dataset_hash,
                )
            )

        pipeline_timings.append(compare_input_pipeline_timing(ds, seed=seed))

        if export_results is None:
            tf_set_seed(seed)
            export_model = TFRegressor(
                n_features=n_features,
                hidden_sizes=settings.model.lstm_hidden_sizes,
                dropout=settings.model.dropout,
            )
            eager_latency = tf_measure_latency(
                export_model, ds.val.windows[0], n_calls=1000, n_warmup=50
            )
            saved_model_latency = export_and_measure_savedmodel(
                export_model,
                ds.val.windows[0],
                window_size=ds.val.windows.shape[1],
                n_features=n_features,
            )
            export_results = {"eager": eager_latency, "saved_model": saved_model_latency}

    return {
        "subset": subset,
        "seeds": list(seeds),
        "regression": {
            fw: aggregate_over_seeds(runs, seeds) for fw, runs in regression_runs.items()
        },
        "classification": {
            fw: aggregate_over_seeds(runs, seeds) for fw, runs in classification_runs.items()
        },
        "input_pipeline_timing": pipeline_timings,
        "savedmodel_export": export_results,
    }
