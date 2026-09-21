"""Single CLI entry point (``pdm``). Subcommands are wired up as their milestone lands."""

import json
from pathlib import Path

import typer

try:
    # Must be imported before pandas (transitively, via any pdm.preprocessing/ingestion import)
    # gets a chance to load — on this Windows environment, pandas-then-torch reliably breaks
    # torch's native DLL loading. torch is an optional extra, so this is a no-op without it.
    # See docs/decisions/P04-pytorch.md.
    import torch  # noqa: F401
except ImportError:
    pass

try:
    # Same class of bug, same fix, for TensorFlow: pandas-before-tensorflow reliably breaks
    # TF's native DLL loading on this Windows environment too. See docs/decisions/P05-tensorflow.md.
    import tensorflow  # noqa: F401
except ImportError:
    pass

from pdm.config import settings

app = typer.Typer(name="pdm", help="PdM-Sentinel: predictive maintenance and anomaly detection.")

data_app = typer.Typer(help="Download, verify, and replay datasets.")
registry_app = typer.Typer(help="Compare and promote models in the MLflow registry.")
drift_app = typer.Typer(help="Data drift detection and drift-triggered retraining.")
app.add_typer(data_app, name="data")
app.add_typer(registry_app, name="registry")
app.add_typer(drift_app, name="drift")


@data_app.command("download")
def data_download() -> None:
    """Download and convert the raw datasets (C-MAPSS, AI4I, NAB) to parquet."""
    from pdm.ingestion.ai4i import load_and_convert_ai4i
    from pdm.ingestion.cmapss import load_and_convert_cmapss
    from pdm.ingestion.nab import load_and_convert_nab

    raw_dir = Path(settings.data.raw_dir)
    processed_dir = Path(settings.data.processed_dir)
    manifest_path = raw_dir / "manifest.json"

    typer.echo("== C-MAPSS ==")
    cmapss_profile = load_and_convert_cmapss(raw_dir / "cmapss", processed_dir, manifest_path)
    for subset, profile in cmapss_profile.items():
        summary = {k: v for k, v in profile.items() if k != "feature_std"}
        typer.echo(f"{subset}: {json.dumps(summary)}")

    typer.echo("== AI4I ==")
    ai4i_profile = load_and_convert_ai4i(raw_dir / "ai4i", processed_dir, manifest_path)
    typer.echo(json.dumps(ai4i_profile, indent=2))

    typer.echo("== NAB ==")
    nab_profile = load_and_convert_nab(raw_dir / "nab", processed_dir, manifest_path)
    typer.echo(json.dumps(nab_profile, indent=2))


@data_app.command("verify")
def data_verify() -> None:
    """Validate downloaded datasets against their schemas without re-downloading."""
    from pdm.ingestion.verify import verify_all

    processed_dir = Path(settings.data.processed_dir)
    report = verify_all(processed_dir)
    typer.echo(json.dumps(report, indent=2))


@data_app.command("replay")
def data_replay(
    path: Path = typer.Argument(..., help="Processed parquet file to replay."),
    rate_hz: float = typer.Option(10.0, help="Records per second (0 disables the delay)."),
    noise_std: float = typer.Option(0.0, help="Std-dev of Gaussian noise added to numeric cols."),
    dropout_prob: float = typer.Option(0.0, help="Probability of dropping a given record."),
    seed: int = typer.Option(0, help="RNG seed for noise/dropout."),
    limit: int = typer.Option(None, help="Stop after this many records."),
) -> None:
    """Replay a processed dataset as a simulated telemetry stream, one JSON record per line."""
    from pdm.ingestion.replay import replay as replay_stream

    for record in replay_stream(
        path,
        rate_hz=rate_hz,
        noise_std=noise_std,
        dropout_prob=dropout_prob,
        seed=seed,
        limit=limit,
    ):
        typer.echo(json.dumps(record, default=str))


@app.command("train")
def train(
    subset: str = typer.Option("FD001", help="C-MAPSS subset to train the RUL LSTM on."),
    seed: int = typer.Option(
        None,
        help="Override the training seed (default: settings.seed). Used by the drift-triggered "
        "retraining job (pdm.monitoring.trigger) so a candidate run never reuses the exact "
        "(subset, seed) pair the currently-served bundle occupies on disk.",
    ),
) -> str:
    """Train the PyTorch LSTM RUL regressor on one C-MAPSS subset, write its bundle, and log
    the run through MLflow. Returns the MLflow run id."""
    import torch

    from pdm.evaluation.harness import Split, evaluate
    from pdm.evaluation.metrics import regression_metrics
    from pdm.evaluation.plots import plot_training_curves
    from pdm.models.torch.architecture import LSTMRegressor, count_parameters
    from pdm.models.torch.bundle import save_torch_bundle
    from pdm.models.torch.dataset import make_dataloader
    from pdm.models.torch.train import set_full_determinism, train_regressor
    from pdm.preprocessing.pipeline import prepare_cmapss_subset
    from pdm.tracking.mlflow_client import (
        log_artifact,
        log_artifacts,
        log_epoch_history,
        log_metrics,
        log_params,
        sha256_of_files,
        start_run,
    )

    processed_dir = Path(settings.data.processed_dir)
    seed = settings.seed if seed is None else seed
    set_full_determinism(seed)
    cmapss_dir = processed_dir / "cmapss"
    dataset_hash = sha256_of_files(
        [cmapss_dir / f"train_{subset}.parquet", cmapss_dir / f"test_{subset}.parquet"]
    )

    ds = prepare_cmapss_subset(subset, processed_dir, seed=seed)
    n_features = len(ds.feature_spec.feature_names)

    train_loader = make_dataloader(ds.train.windows, ds.train.targets, shuffle=True, seed=seed)
    val_loader = make_dataloader(ds.val.windows, ds.val.targets, shuffle=False)

    model = LSTMRegressor(
        n_features=n_features,
        hidden_sizes=settings.model.lstm_hidden_sizes,
        dropout=settings.model.dropout,
    )
    typer.echo(model.summary())

    run_id = f"torch_rul_regressor__{subset}__seed{seed}"
    with start_run(
        run_name=run_id,
        framework="pytorch",
        seed=seed,
        dataset_hash=dataset_hash,
        extra_tags={"task": "rul_regression", "subset": subset},
    ) as mlflow_run:
        log_params(settings.model_dump())
        log_params({"subset": subset, "n_features": n_features})

        result = train_regressor(model, train_loader, val_loader, seed=seed, verbose=True)
        typer.echo(f"Best epoch {result.best_epoch}, val RMSE {result.best_val_score:.3f}")
        log_epoch_history(result.history)

        test_split = Split("test", ds.test.windows, ds.test.targets)

        def predict_fn(windows):
            model.eval()
            with torch.no_grad():
                return model(torch.as_tensor(windows, dtype=torch.float32)).numpy()

        test_metrics = evaluate(predict_fn, test_split, "rul", regression_metrics)
        typer.echo(f"Test metrics: {json.dumps(test_metrics)}")
        log_metrics({"val_rmse": result.best_val_score})
        log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        run_dir = Path("artifacts") / run_id
        save_torch_bundle(
            run_dir,
            model,
            scaler=ds.scaler,
            feature_spec=ds.feature_spec,
            train_windows=ds.train.windows,
            metrics={"val": {"rmse": result.best_val_score}, "test": test_metrics},
            thresholds={"failure_horizon_w": settings.data.failure_horizon_w},
            seed=seed,
            dataset_version=subset,
            parameter_count=count_parameters(model),
            mlflow_run_id=mlflow_run.info.run_id,
            architecture={
                "hidden_sizes": list(settings.model.lstm_hidden_sizes),
                "dropout": settings.model.dropout,
            },
        )
        typer.echo(f"Wrote bundle to {run_dir}")

        curves_path = Path("reports") / run_id / "training_curves.png"
        plot_training_curves([h.to_dict() for h in result.history], curves_path)
        typer.echo(f"Wrote training curves to {curves_path}")

        log_artifacts(run_dir, artifact_path="bundle")
        log_artifact(curves_path, artifact_path="plots")
        typer.echo(f"Logged MLflow run {mlflow_run.info.run_id}")
        return mlflow_run.info.run_id


@app.command("train-tf")
def train_tf(
    subset: str = typer.Option("FD001", help="C-MAPSS subset to train the TensorFlow RUL LSTM on."),
    seed: int = typer.Option(None, help="Override the training seed (default: settings.seed)."),
) -> str:
    """Train the TensorFlow LSTM RUL regressor on one C-MAPSS subset, write its bundle, and log
    the run through MLflow. Returns the MLflow run id."""
    import tensorflow as tf

    from pdm.evaluation.harness import Split, evaluate
    from pdm.evaluation.metrics import regression_metrics
    from pdm.evaluation.plots import plot_training_curves
    from pdm.models.tf.architecture import LSTMRegressor, count_parameters
    from pdm.models.tf.bundle import save_tf_bundle
    from pdm.models.tf.dataset import make_tf_dataset
    from pdm.models.tf.train import set_full_determinism, train_regressor
    from pdm.preprocessing.pipeline import prepare_cmapss_subset
    from pdm.tracking.mlflow_client import (
        log_artifact,
        log_artifacts,
        log_epoch_history,
        log_metrics,
        log_params,
        sha256_of_files,
        start_run,
    )

    processed_dir = Path(settings.data.processed_dir)
    seed = settings.seed if seed is None else seed
    set_full_determinism(seed)
    cmapss_dir = processed_dir / "cmapss"
    dataset_hash = sha256_of_files(
        [cmapss_dir / f"train_{subset}.parquet", cmapss_dir / f"test_{subset}.parquet"]
    )

    ds = prepare_cmapss_subset(subset, processed_dir, seed=seed)
    n_features = len(ds.feature_spec.feature_names)

    train_ds = make_tf_dataset(ds.train.windows, ds.train.targets, "rul", shuffle=True, seed=seed)
    val_ds = make_tf_dataset(ds.val.windows, ds.val.targets, "rul", shuffle=False)

    model = LSTMRegressor(
        n_features=n_features,
        hidden_sizes=settings.model.lstm_hidden_sizes,
        dropout=settings.model.dropout,
    )
    typer.echo(model.summary_text())

    run_id = f"tf_rul_regressor__{subset}__seed{seed}"
    with start_run(
        run_name=run_id,
        framework="tensorflow",
        seed=seed,
        dataset_hash=dataset_hash,
        extra_tags={"task": "rul_regression", "subset": subset},
    ) as mlflow_run:
        log_params(settings.model_dump())
        log_params({"subset": subset, "n_features": n_features})

        result = train_regressor(model, train_ds, val_ds, seed=seed, verbose=True)
        typer.echo(f"Best epoch {result.best_epoch}, val RMSE {result.best_val_score:.3f}")
        log_epoch_history(result.history)

        test_split = Split("test", ds.test.windows, ds.test.targets)

        def predict_fn(windows):
            return model(tf.convert_to_tensor(windows, dtype=tf.float32), training=False).numpy()

        test_metrics = evaluate(predict_fn, test_split, "rul", regression_metrics)
        typer.echo(f"Test metrics: {json.dumps(test_metrics)}")
        log_metrics({"val_rmse": result.best_val_score})
        log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        run_dir = Path("artifacts") / run_id
        save_tf_bundle(
            run_dir,
            model,
            scaler=ds.scaler,
            feature_spec=ds.feature_spec,
            train_windows=ds.train.windows,
            metrics={"val": {"rmse": result.best_val_score}, "test": test_metrics},
            thresholds={"failure_horizon_w": settings.data.failure_horizon_w},
            seed=seed,
            dataset_version=subset,
            parameter_count=count_parameters(model),
            mlflow_run_id=mlflow_run.info.run_id,
            architecture={
                "hidden_sizes": list(settings.model.lstm_hidden_sizes),
                "dropout": settings.model.dropout,
            },
        )
        typer.echo(f"Wrote bundle to {run_dir}")

        curves_path = Path("reports") / run_id / "training_curves.png"
        plot_training_curves([h.to_dict() for h in result.history], curves_path)
        typer.echo(f"Wrote training curves to {curves_path}")

        log_artifacts(run_dir, artifact_path="bundle")
        log_artifact(curves_path, artifact_path="plots")
        typer.echo(f"Logged MLflow run {mlflow_run.info.run_id}")
        return mlflow_run.info.run_id


@app.command("benchmark")
def benchmark(
    subset: str = typer.Option("FD001", help="C-MAPSS subset to benchmark on."),
) -> None:
    """Run the PyTorch vs TensorFlow framework benchmark (5 seeds each) and write its artifact."""
    from pdm.evaluation.framework_benchmark import run_framework_benchmark

    processed_dir = Path(settings.data.processed_dir)
    result = run_framework_benchmark(subset, processed_dir)

    out_dir = Path("artifacts/framework_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subset}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"Wrote {out_path}")

    for task in ("regression", "classification"):
        typer.echo(f"== {task} ==")
        for framework in ("pytorch", "tensorflow"):
            typer.echo(f"{framework}: {json.dumps(result[task][framework])}")


@app.command("evaluate")
def evaluate() -> None:
    """Run every classical baseline (5 seeds each) and regenerate docs/RESULTS.md."""
    from pdm.evaluation.results import generate_results_md
    from pdm.models.baseline.runner import ARTIFACTS_DIR, run_all_baselines

    processed_dir = Path(settings.data.processed_dir)
    written = run_all_baselines(processed_dir)
    typer.echo(f"Wrote {len(written)} metrics.json files under {ARTIFACTS_DIR}/")

    output_path = Path("docs/RESULTS.md")
    generate_results_md(ARTIFACTS_DIR, output_path)
    typer.echo(f"Regenerated {output_path}")


@app.command("anomaly")
def anomaly(
    subset: str = typer.Option("FD001", help="C-MAPSS subset to run anomaly detection on."),
) -> None:
    """Run the full P06 anomaly-detection pipeline: bottleneck ablation, Isolation Forest
    contamination sweep, fusion-weight selection, and the final 5-seed C-MAPSS + NAB evaluation.
    Writes JSON artifacts, plots, and a PyTorch + TensorFlow production bundle."""
    import numpy as np
    import tensorflow as tf

    from pdm.evaluation.metrics import classification_metrics
    from pdm.evaluation.plots import (
        plot_f1_vs_bottleneck,
        plot_pr_curve,
        plot_score_distribution,
        plot_sensor_heatmap,
    )
    from pdm.models.baseline.isolation_forest import fit_isolation_forest, isolation_forest_scores
    from pdm.models.tf.autoencoder import LSTMAutoencoder as TFAutoencoder
    from pdm.models.tf.autoencoder import reconstruction_error as tf_reconstruction_error
    from pdm.models.tf.bundle import save_tf_bundle
    from pdm.models.tf.dataset import make_tf_dataset
    from pdm.models.tf.train import set_full_determinism as tf_set_seed
    from pdm.models.torch.anomaly_experiments import (
        SUBSET,
        _cmapss_dataset_hash,
        _healthy_mask,
        _is_anomaly_labels,
        _load_cmapss_split,
        _train_ae,
        bottleneck_ablation,
        final_cmapss_evaluation,
        final_nab_evaluation,
        fusion_weight_sweep,
        isolation_forest_contamination_sweep,
    )
    from pdm.models.torch.autoencoder import (
        per_sensor_reconstruction_error,
        reconstruction_error,
    )
    from pdm.models.torch.bundle import save_torch_bundle
    from pdm.tracking.mlflow_client import log_artifacts, log_metrics, log_params, start_run

    subset = subset or SUBSET
    processed_dir = Path(settings.data.processed_dir)
    out_dir = Path("artifacts/anomaly_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports/anomaly")

    def _write(name: str, payload) -> None:
        (out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        typer.echo(f"Wrote {out_dir / name}")

    typer.echo("== bottleneck ablation ==")
    ablation = bottleneck_ablation(subset=subset, processed_dir=processed_dir)
    _write("bottleneck_ablation.json", ablation)
    plot_f1_vs_bottleneck(
        {dim: (m["f1"]["mean"], m["f1"]["std"]) for dim, m in ablation.items()},
        reports_dir / "f1_vs_bottleneck.png",
    )
    chosen_latent_dim = max(ablation, key=lambda d: ablation[d]["f1"]["mean"])
    typer.echo(f"Chosen bottleneck dim (max val F1): {chosen_latent_dim}")

    typer.echo("== Isolation Forest contamination sweep ==")
    contamination_sweep = isolation_forest_contamination_sweep(
        subset=subset, processed_dir=processed_dir
    )
    _write("isolation_forest_contamination_sweep.json", contamination_sweep)

    typer.echo("== fusion weight sweep ==")
    fusion_sweep = fusion_weight_sweep(
        chosen_latent_dim, subset=subset, processed_dir=processed_dir
    )
    _write("fusion_weight_sweep.json", fusion_sweep)
    chosen_weight = max(fusion_sweep, key=lambda w: fusion_sweep[w]["f1"]["mean"])
    typer.echo(f"Chosen fusion weight (max val F1): {chosen_weight}")

    typer.echo("== final C-MAPSS evaluation (5 seeds) ==")
    cmapss_result = final_cmapss_evaluation(
        chosen_latent_dim, chosen_weight, subset=subset, processed_dir=processed_dir
    )
    _write("final_cmapss_evaluation.json", cmapss_result)

    typer.echo("== final NAB evaluation (5 seeds) ==")
    nab_result = final_nab_evaluation(chosen_latent_dim, chosen_weight)
    _write("final_nab_evaluation.json", nab_result)

    payload = {
        "chosen_latent_dim": chosen_latent_dim,
        "chosen_fusion_weight": chosen_weight,
        "cmapss": cmapss_result,
        "nab": nab_result,
    }
    _write("summary.json", payload)

    # --- plots requiring one concrete trained model (seed 42) ------------------------------
    typer.echo("== plots + bundles (seed 42) ==")
    seed = settings.seed
    ds = _load_cmapss_split(seed, subset, processed_dir)
    n_features = len(ds.feature_spec.feature_names)
    window_size = ds.train.windows.shape[1]
    train_healthy = ds.train.windows[_healthy_mask(ds.train.targets)]
    val_healthy = ds.val.windows[_healthy_mask(ds.val.targets)]

    torch_ae = _train_ae(
        train_healthy,
        val_healthy,
        n_features=n_features,
        window_size=window_size,
        latent_dim=chosen_latent_dim,
        seed=seed,
        max_epochs=settings.training.max_epochs,
        patience=settings.training.early_stopping_patience,
    )
    if_model = fit_isolation_forest(train_healthy, seed=seed)

    test_labels = _is_anomaly_labels(ds.test.targets)
    ae_test_scores = reconstruction_error(torch_ae, ds.test.windows)
    if_test_scores = isolation_forest_scores(if_model, ds.test.windows)

    plot_score_distribution(
        ae_test_scores[test_labels == 0],
        ae_test_scores[test_labels == 1],
        float(np.percentile(reconstruction_error(torch_ae, train_healthy), 99)),
        reports_dir / "score_distribution_cmapss.png",
    )
    plot_pr_curve(test_labels, ae_test_scores, reports_dir / "pr_curve_ae.png", label="LSTM-AE")
    plot_pr_curve(
        test_labels,
        if_test_scores,
        reports_dir / "pr_curve_isolation_forest.png",
        label="Isolation Forest",
    )

    anomalous_idx = np.where(test_labels == 1)[0]
    top_anomalous = anomalous_idx[np.argsort(-ae_test_scores[anomalous_idx])[:15]]
    per_sensor_err = per_sensor_reconstruction_error(torch_ae, ds.test.windows[top_anomalous])
    plot_sensor_heatmap(
        per_sensor_err, ds.feature_spec.feature_names, reports_dir / "per_sensor_heatmap.png"
    )

    run_id = f"torch_lstm_ae__{subset}__seed{seed}"
    torch_ae_threshold = float(np.percentile(reconstruction_error(torch_ae, train_healthy), 99))
    torch_ae_test_metrics = classification_metrics(
        test_labels,
        (ae_test_scores >= torch_ae_threshold).astype(int),
        ae_test_scores,
        include_brier=False,
    )
    torch_ae_param_count = sum(p.numel() for p in torch_ae.parameters() if p.requires_grad)
    with start_run(
        run_name=run_id,
        framework="pytorch",
        seed=seed,
        dataset_hash=_cmapss_dataset_hash(processed_dir, subset),
        extra_tags={"task": "anomaly_detection", "dataset": "cmapss", "subset": subset},
    ) as torch_ae_run:
        log_params(
            {
                "latent_dim": chosen_latent_dim,
                "fusion_weight": chosen_weight,
                "subset": subset,
                "n_features": n_features,
            }
        )
        log_metrics({f"test_{k}": v for k, v in torch_ae_test_metrics.items()})
        bundle_dir = Path("artifacts") / run_id
        save_torch_bundle(
            bundle_dir,
            torch_ae,
            scaler=ds.scaler,
            feature_spec=ds.feature_spec,
            train_windows=ds.train.windows,
            metrics={"test": torch_ae_test_metrics},
            thresholds={
                "healthy_rul_threshold": settings.model.healthy_rul_threshold,
                "anomaly_score_threshold": torch_ae_threshold,
                "anomaly_score_threshold_method": "percentile-99-of-healthy-training-scores",
            },
            seed=seed,
            dataset_version=subset,
            parameter_count=torch_ae_param_count,
            mlflow_run_id=torch_ae_run.info.run_id,
            architecture={"latent_dim": chosen_latent_dim},
        )
        log_artifacts(bundle_dir, artifact_path="bundle")
    typer.echo(f"Wrote PyTorch AE bundle to artifacts/{run_id}")

    # TF production-mirror bundle: one seed, not benchmarked further (see decision log).
    tf_set_seed(seed)
    tf_ae = TFAutoencoder(
        n_features=n_features, window_size=window_size, latent_dim=chosen_latent_dim
    )
    train_ds = make_tf_dataset(
        train_healthy, {"windows": train_healthy}, "windows", shuffle=True, seed=seed
    )
    val_ds = make_tf_dataset(val_healthy, {"windows": val_healthy}, "windows", shuffle=False)
    tf_ae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings.training.learning_rate),
        loss="mse",
    )
    tf_ae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.training.max_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=settings.training.early_stopping_patience, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    tf_train_scores = tf_reconstruction_error(tf_ae, train_healthy)
    tf_test_scores = tf_reconstruction_error(tf_ae, ds.test.windows)
    tf_threshold = float(np.percentile(tf_train_scores, 99))
    tf_metrics = classification_metrics(
        test_labels,
        (tf_test_scores >= tf_threshold).astype(int),
        tf_test_scores,
        include_brier=False,
    )
    typer.echo(
        f"TF AE production-mirror test metrics (single seed {seed}): {json.dumps(tf_metrics)}"
    )

    tf_run_id = f"tf_lstm_ae__{subset}__seed{seed}"
    tf_ae_param_count = sum(int(tf.size(v)) for v in tf_ae.trainable_variables)
    with start_run(
        run_name=tf_run_id,
        framework="tensorflow",
        seed=seed,
        dataset_hash=_cmapss_dataset_hash(processed_dir, subset),
        extra_tags={"task": "anomaly_detection", "dataset": "cmapss", "subset": subset},
    ) as tf_ae_run:
        log_params(
            {
                "latent_dim": chosen_latent_dim,
                "fusion_weight": chosen_weight,
                "subset": subset,
                "n_features": n_features,
            }
        )
        log_metrics({f"test_{k}": v for k, v in tf_metrics.items()})
        tf_bundle_dir = Path("artifacts") / tf_run_id
        save_tf_bundle(
            tf_bundle_dir,
            tf_ae,
            scaler=ds.scaler,
            feature_spec=ds.feature_spec,
            train_windows=ds.train.windows,
            metrics={"test": tf_metrics},
            thresholds={
                "healthy_rul_threshold": settings.model.healthy_rul_threshold,
                "anomaly_score_threshold": tf_threshold,
                "anomaly_score_threshold_method": "percentile-99-of-healthy-training-scores",
            },
            seed=seed,
            dataset_version=subset,
            parameter_count=tf_ae_param_count,
            mlflow_run_id=tf_ae_run.info.run_id,
            architecture={"latent_dim": chosen_latent_dim},
        )
        log_artifacts(tf_bundle_dir, artifact_path="bundle")
    typer.echo(f"Wrote TensorFlow AE bundle to artifacts/{tf_run_id}")
    typer.echo("Done.")


@app.command("serve")
def serve() -> None:
    """Serve a trained bundle over the FastAPI inference API."""
    import uvicorn

    uvicorn.run("pdm.serving.app:app", host=settings.serving.host, port=settings.serving.port)


@app.command("load-test")
def load_test(
    base_url: str = typer.Option("http://localhost:8000", help="Base URL of a running API."),
    duration_seconds: float = typer.Option(
        120.0, help="How long to generate traffic for, in seconds."
    ),
    rate_hz: float = typer.Option(5.0, help="Request pairs (/predict/rul + /detect/anomaly) per second."),
    bundle_dir: str = typer.Option(
        None, help="Bundle whose feature_spec the traffic is windowed against (default: "
        "settings.serving.bundle_dir)."
    ),
    drifted_fraction: float = typer.Option(
        0.15,
        help="Fraction of requests drawn from real FD002/FD003 (genuinely drifted, per "
        "docs/decisions/P09-drift.md) traffic instead of FD001 -- so the anomaly-rate/alert "
        "panels and drift gauges have something real to show, not a flat line.",
    ),
) -> None:
    """P10 deliverable #6: replays real C-MAPSS telemetry against a running `pdm serve` (or
    `docker compose up`) API so the Grafana dashboards have real data. Unlike a synthetic noise
    generator, every window sent here is a real, already-labelled sensor reading this project
    has used throughout P01-P09 -- just replayed as live traffic rather than read in a batch."""
    import random
    import time

    import requests

    from pdm.preprocessing.pipeline import load_raw_cmapss_windows
    from pdm.preprocessing.windowing import FeatureSpec

    bundle_dir = bundle_dir or settings.serving.bundle_dir
    feature_spec = FeatureSpec.load(Path(bundle_dir) / "feature_spec.json")
    processed_dir = Path(settings.data.processed_dir)

    kwargs = {
        "feature_names": feature_spec.feature_names,
        "window_size": feature_spec.window_size,
        "stride": feature_spec.stride,
        "split": "test",
    }
    normal = load_raw_cmapss_windows("FD001", processed_dir, **kwargs)
    drifted = [
        load_raw_cmapss_windows(subset, processed_dir, **kwargs) for subset in ("FD002", "FD003")
    ]
    typer.echo(
        f"Loaded {len(normal.windows)} FD001 (normal) + "
        f"{sum(len(d.windows) for d in drifted)} FD002/FD003 (drifted) windows."
    )

    interval = 1.0 / rate_hz if rate_hz > 0 else 0.0
    end_time = time.time() + duration_seconds
    sent = errors = 0
    while time.time() < end_time:
        pool = random.choice(drifted) if random.random() < drifted_fraction else normal
        window = pool.windows[random.randrange(len(pool.windows))]
        payload = {"feature_names": feature_spec.feature_names, "window": window.tolist()}
        for path in ("/predict/rul", "/detect/anomaly"):
            try:
                r = requests.post(f"{base_url}{path}", json=payload, timeout=5)
                if r.status_code not in (200, 503):
                    errors += 1
            except requests.RequestException as exc:
                errors += 1
                typer.echo(f"request to {path} failed: {exc}")
        sent += 1
        if interval:
            time.sleep(interval)

    typer.echo(f"Sent {sent} request pairs ({errors} errors) to {base_url} over ~{duration_seconds:.0f}s")


@drift_app.command("check")
def drift_check(
    production_subset: str = typer.Option(
        ..., help="C-MAPSS subset to replay as 'production' traffic, e.g. FD002 or FD003."
    ),
    bundle_dir: str = typer.Option(
        None,
        help="RUL bundle providing reference_stats.json (default: settings.serving.bundle_dir).",
    ),
    anomaly_bundle_dir: str = typer.Option(
        None,
        help="Optional anomaly bundle, for the F1-degradation check (default: "
        "settings.serving.anomaly_bundle_dir).",
    ),
    split: str = typer.Option(
        "test", help="Which split of production_subset to check: train, val, or test."
    ),
    retrain_subset: str = typer.Option(
        None,
        help="Subset to retrain the candidate on if the trigger fires (default: the checked "
        "bundle's own training subset, from metadata.json's dataset_version).",
    ),
    threshold: float = typer.Option(
        None,
        help="Aggregate PSI threshold to trigger on (default: "
        "settings.monitoring.psi_significant_threshold).",
    ),
    no_retrain: bool = typer.Option(
        False,
        "--no-retrain",
        help="Report and log the trigger decision, but never launch a retraining run or touch "
        "the registry, even if the threshold is cleared.",
    ),
    noise_std: float = typer.Option(
        0.0,
        help="Std-dev of Gaussian noise the replay simulator adds to production traffic "
        "(pdm.ingestion.replay), simulating sensor jitter. 0 reproduces the raw parquet exactly.",
    ),
    dropout_prob: float = typer.Option(
        0.0,
        help="Probability the replay simulator drops a given production record, simulating a "
        "missed reading. 0 reproduces the raw parquet exactly.",
    ),
) -> None:
    """Check 'production' traffic — replayed through pdm.ingestion.replay, the same telemetry
    replay simulator `pdm data replay` uses — for input drift (PSI + corrected KS) against a
    bundle's frozen reference stats, measure the model's actual degradation on that traffic,
    decide whether to launch a retraining candidate (never auto-promoted), write a human-readable
    report, and emit Prometheus metrics."""
    from pdm.monitoring.drift import (
        compute_input_drift,
        compute_prediction_drift,
        evaluate_model_degradation,
    )
    from pdm.monitoring.metrics import write_drift_metrics
    from pdm.monitoring.reports import generate_feature_plots, render_markdown_report
    from pdm.monitoring.trigger import check_and_maybe_retrain
    from pdm.preprocessing.pipeline import load_raw_cmapss_windows
    from pdm.serving.bundle import load_anomaly_bundle, load_rul_bundle

    bundle_dir = bundle_dir or settings.serving.bundle_dir
    rul_bundle = load_rul_bundle(bundle_dir)
    reference_stats = json.loads(
        (Path(bundle_dir) / "reference_stats.json").read_text(encoding="utf-8")
    )
    feature_names = rul_bundle.feature_spec.feature_names
    reference_subset = rul_bundle.metadata.get("dataset_version")
    processed_dir = Path(settings.data.processed_dir)

    anomaly_bundle = None
    anomaly_dir = anomaly_bundle_dir or settings.serving.anomaly_bundle_dir
    if anomaly_dir:
        try:
            anomaly_bundle = load_anomaly_bundle(anomaly_dir)
        except Exception as exc:  # noqa: BLE001 - degrade, don't fail the whole check
            typer.echo(
                f"anomaly bundle at {anomaly_dir} failed to load, skipping F1 degradation: {exc}"
            )

    prod_raw = load_raw_cmapss_windows(
        production_subset,
        processed_dir,
        feature_names=feature_names,
        window_size=rul_bundle.feature_spec.window_size,
        stride=rul_bundle.feature_spec.stride,
        split=split,
        noise_std=noise_std,
        dropout_prob=dropout_prob,
    )
    production_windows = rul_bundle.scale(prod_raw.windows)

    input_drift = compute_input_drift(reference_stats, production_windows, feature_names)
    degradation = evaluate_model_degradation(
        rul_bundle, prod_raw.windows, prod_raw.targets, anomaly_bundle=anomaly_bundle
    )

    prediction_drift_result = None
    if reference_subset:
        # split="train", not "test": reference_stats (input drift's own reference) is computed
        # from the training split too, and using "train" here keeps the two forms of drift
        # anchored to the same reference distribution. It also matters for the FD001-holdout
        # false-positive check specifically: when production_subset == reference_subset,
        # comparing against the *test* split there would compare FD001-test's predictions
        # against themselves -- a tautological "zero drift" result that proves nothing.
        ref_raw = load_raw_cmapss_windows(
            reference_subset,
            processed_dir,
            feature_names=feature_names,
            window_size=rul_bundle.feature_spec.window_size,
            stride=rul_bundle.feature_spec.stride,
            split="train",
        )
        ref_predictions = rul_bundle.predict_rul(rul_bundle.scale(ref_raw.windows))
        prod_predictions = rul_bundle.predict_rul(production_windows)
        prediction_drift_result = compute_prediction_drift(ref_predictions, prod_predictions)

    retrain_subset = retrain_subset or reference_subset or production_subset
    target = f"{Path(bundle_dir).name}::{production_subset}:{split}"
    decision = check_and_maybe_retrain(
        input_drift["aggregate"]["mean_psi"],
        target=target,
        retrain_subset=retrain_subset,
        retrain=not no_retrain,
        threshold=threshold,
    )

    reports_dir = Path(settings.monitoring.reports_dir) / f"{production_subset}_{split}"
    plot_paths = generate_feature_plots(
        reference_stats, production_windows, feature_names, input_drift, out_dir=reports_dir
    )

    result_payload = {
        "target": target,
        "bundle_dir": str(bundle_dir),
        "production_subset": production_subset,
        "split": split,
        "reference_subset": reference_subset,
        "input_drift": input_drift,
        "prediction_drift": prediction_drift_result,
        "degradation": degradation,
        "trigger": {
            "triggered": decision.triggered,
            "reason": decision.reason,
            "aggregate_score": decision.aggregate_score,
            "threshold": decision.threshold,
            "cooldown_active": decision.cooldown_active,
            "candidate_run_id": decision.candidate_run_id,
            "candidate_model_name": decision.candidate_model_name,
            "candidate_version": decision.candidate_version,
        },
    }
    result_path = reports_dir / "drift_result.json"
    result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"Wrote {result_path}")

    md_path = render_markdown_report(
        input_drift,
        title=f"Drift check: {production_subset} ({split}) vs. {reference_subset or 'reference'}",
        plot_paths=plot_paths,
        prediction_drift=prediction_drift_result,
        degradation=degradation,
        trigger=result_payload["trigger"],
        out_path=reports_dir / "report.md",
    )
    typer.echo(f"Wrote {md_path}")

    write_drift_metrics(
        target=target,
        input_drift=input_drift,
        triggered=decision.triggered,
        cooldown_active=decision.cooldown_active,
        path=Path(settings.monitoring.metrics_textfile_path),
        prediction_drift=prediction_drift_result,
    )
    typer.echo(f"Wrote metrics to {settings.monitoring.metrics_textfile_path}")

    typer.echo(f"Aggregate PSI (mean): {input_drift['aggregate']['mean_psi']:.4f}")
    typer.echo(f"RUL RMSE on this traffic: {degradation['rul']['rmse']:.3f}")
    typer.echo(f"Trigger decision: {decision.reason}")
    if decision.candidate_run_id:
        typer.echo(
            f"Candidate run {decision.candidate_run_id} registered as "
            f"{decision.candidate_model_name} v{decision.candidate_version} (Staging) — NOT "
            f"promoted. Run `pdm registry promote --run-id {decision.candidate_run_id}` to "
            "attempt promotion through the gate."
        )


@drift_app.command("report")
def drift_report(
    result: Path = typer.Option(
        ..., "--result", help="Path to a drift_result.json previously written by `pdm drift check`."
    ),
    out: Path = typer.Option(
        None, help="Output markdown path (default: report.md alongside the result file)."
    ),
) -> None:
    """Re-render the human-readable markdown report from a previously saved `pdm drift check`
    result, without recomputing anything. Reuses whichever per-feature plots `drift check` left
    alongside the result file."""
    from pdm.monitoring.reports import render_markdown_report

    payload = json.loads(result.read_text(encoding="utf-8"))
    out_path = out or result.parent / "report.md"
    plot_paths = {p.stem: p for p in sorted(result.parent.glob("*.png"))}

    md_path = render_markdown_report(
        payload["input_drift"],
        title=(
            f"Drift check: {payload['production_subset']} ({payload['split']}) vs. "
            f"{payload.get('reference_subset') or 'reference'}"
        ),
        plot_paths=plot_paths,
        prediction_drift=payload.get("prediction_drift"),
        degradation=payload.get("degradation"),
        trigger=payload.get("trigger"),
        out_path=out_path,
    )
    typer.echo(f"Wrote {md_path}")


@registry_app.command("promote")
def registry_promote(
    run_id: str = typer.Option(..., "--run-id", help="MLflow run id of the candidate model."),
    model_name: str = typer.Option(
        None, help="Registered model name (default: inferred from the run's 'task' tag)."
    ),
    primary_metric: str = typer.Option(
        None, help="Metric to gate on (default: settings.tracking.promotion_primary_metric)."
    ),
    margin: float = typer.Option(
        None, help="Required relative improvement (default: settings.tracking.promotion_margin)."
    ),
) -> None:
    """Register (if needed) and attempt to promote a candidate run to Production. Refuses with a
    clear message and a non-zero exit code if the promotion gate fails — the registry is left
    untouched on refusal."""
    import mlflow

    from pdm.tracking.mlflow_client import configure_tracking
    from pdm.tracking.promotion import PromotionRefused, promote_to_production
    from pdm.tracking.registry import (
        ANOMALY_MODEL_NAME,
        RUL_MODEL_NAME,
        register_model_version,
        transition_to_staging,
    )

    configure_tracking()
    run = mlflow.get_run(run_id)
    if model_name is None:
        task = run.data.tags.get("task", "")
        model_name = ANOMALY_MODEL_NAME if "anomaly" in task else RUL_MODEL_NAME

    client = mlflow.tracking.MlflowClient()
    existing = [
        mv for mv in client.search_model_versions(f"run_id='{run_id}'") if mv.name == model_name
    ]
    if existing:
        version = existing[0].version
    else:
        version = register_model_version(run_id, model_name, artifact_path="bundle")
        typer.echo(f"Registered {model_name} version {version} from run {run_id}")
    transition_to_staging(model_name, version)

    try:
        decision = promote_to_production(
            model_name, version, primary_metric=primary_metric, margin=margin
        )
    except PromotionRefused as exc:
        typer.echo(f"Promotion REFUSED: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"Promoted {model_name} version {version} to Production: {decision.reason}")


@registry_app.command("compare")
def registry_compare(
    primary_metric: str = typer.Option(..., "--metric", help="Metric to rank by, e.g. rmse or f1."),
    experiment_name: str = typer.Option(
        None, help="Experiment to compare runs in (default: settings.tracking.experiment_name)."
    ),
    n: int = typer.Option(10, help="How many top runs to show."),
    higher_is_better: bool = typer.Option(
        None, help="Override ranking direction (default: inferred from the metric name)."
    ),
) -> None:
    """Show the top N runs by primary metric, with their key params/tags — the actual comparison
    behind "how did you decide which model to ship," not an assertion."""
    from pdm.tracking.promotion import is_higher_better
    from pdm.tracking.registry import compare_top_runs

    experiment_name = experiment_name or settings.tracking.experiment_name
    higher = is_higher_better(primary_metric) if higher_is_better is None else higher_is_better
    rows = compare_top_runs(experiment_name, primary_metric, n=n, higher_is_better=higher)
    if not rows:
        typer.echo(
            f"No runs with metric {primary_metric!r} found in experiment {experiment_name!r}."
        )
        return
    for i, r in enumerate(rows, 1):
        typer.echo(
            f"{i}. {r.run_name} ({r.run_id[:8]}) {primary_metric}={r.metric_value:.4g} "
            f"tags={r.tags}"
        )


if __name__ == "__main__":
    app()
