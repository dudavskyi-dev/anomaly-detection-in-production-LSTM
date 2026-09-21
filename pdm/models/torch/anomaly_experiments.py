"""The P06 required experiments: bottleneck ablation, Isolation Forest contamination
sensitivity, fusion-weight selection, and the final 5-seed evaluation on C-MAPSS and NAB.

Every detector here is trained on **healthy-only** windows (``rul >= healthy_rul_threshold``,
already computed by P02's capped-RUL labelling — a window's capped RUL equals the cap exactly
iff the engine still had at least that much life left when the window ended). Sweeps use a
**reduced budget** (mirroring P04's `SWEEP_MAX_EPOCHS`/`SWEEP_SEEDS` pattern, for the same
CPU-time reasons — see `docs/decisions/P04-pytorch.md`); the final headline numbers use the
project's full default budget and all 5 configured seeds.
"""

from pathlib import Path

import numpy as np

from pdm.config import settings
from pdm.evaluation.fusion import ScoreStandardizer, fuse_scores
from pdm.evaluation.metrics import classification_metrics
from pdm.evaluation.stability import aggregate_over_seeds
from pdm.evaluation.thresholds import max_f1_threshold, percentile_threshold, threshold_at_precision
from pdm.models.baseline.isolation_forest import fit_isolation_forest, isolation_forest_scores
from pdm.models.baseline.window_features import make_baseline_features
from pdm.models.torch.autoencoder import LSTMAutoencoder, reconstruction_error, train_autoencoder
from pdm.models.torch.dataset import make_dataloader
from pdm.models.torch.train import set_full_determinism
from pdm.preprocessing.nab_pipeline import prepare_nab_windows
from pdm.preprocessing.pipeline import PreparedDataset, prepare_cmapss_subset
from pdm.tracking.mlflow_client import log_metrics, log_params, sha256_of_files, start_run

SUBSET = "FD001"
SWEEP_MAX_EPOCHS = 40
SWEEP_PATIENCE = 8
SWEEP_SEEDS = (0, 1)
BOTTLENECK_DIMS = (2, 4, 8, 16, 32)
FUSION_WEIGHTS = tuple(round(i / 10, 1) for i in range(11))
CONTAMINATION_GRID = (0.02, 0.05, 0.1, 0.15, 0.2)


def _healthy_mask(targets: dict[str, np.ndarray]) -> np.ndarray:
    return targets["rul"] >= settings.model.healthy_rul_threshold


def _is_anomaly_labels(targets: dict[str, np.ndarray]) -> np.ndarray:
    return (targets["rul"] < settings.model.healthy_rul_threshold).astype(int)


def _numeric_only(d: dict) -> dict:
    """Drops non-numeric bookkeeping fields (``split``, etc.) from a threshold-selection
    result so it can be passed to :func:`aggregate_over_seeds`, which expects every value to be
    a plain float."""
    return {k: v for k, v in d.items() if isinstance(v, int | float) and not isinstance(v, bool)}


def _load_cmapss_split(seed: int, subset: str, processed_dir: Path) -> PreparedDataset:
    return prepare_cmapss_subset(subset, processed_dir, seed=seed)


def _cmapss_dataset_hash(processed_dir: Path, subset: str) -> str:
    cmapss_dir = processed_dir / "cmapss"
    return sha256_of_files(
        [cmapss_dir / f"train_{subset}.parquet", cmapss_dir / f"test_{subset}.parquet"]
    )


def _train_ae(
    train_healthy: np.ndarray,
    val_healthy: np.ndarray,
    *,
    n_features: int,
    window_size: int,
    latent_dim: int,
    seed: int,
    max_epochs: int,
    patience: int,
) -> LSTMAutoencoder:
    set_full_determinism(seed)
    model = LSTMAutoencoder(n_features=n_features, window_size=window_size, latent_dim=latent_dim)
    train_loader = make_dataloader(train_healthy, {}, shuffle=True, seed=seed)
    val_loader = make_dataloader(val_healthy, {}, shuffle=False)
    train_autoencoder(
        model, train_loader, val_loader, seed=seed, max_epochs=max_epochs, patience=patience
    )
    return model


# --- deliverable #2: bottleneck ablation ---------------------------------------------------------


def _run_one_bottleneck_config(
    latent_dim: int, seed: int, subset: str, processed_dir: Path
) -> dict:
    """One (latent_dim, seed) point of the ablation curve — split out from
    :func:`bottleneck_ablation` so ``tests/test_anomaly_experiments.py`` can monkeypatch this one
    function and check the sweep's orchestration/aggregation without training anything for real,
    the same pattern ``pdm.models.torch.experiments`` uses for the P04 sweeps."""
    ds = _load_cmapss_split(seed, subset, processed_dir)
    n_features = len(ds.feature_spec.feature_names)
    window_size = ds.train.windows.shape[1]
    train_healthy = ds.train.windows[_healthy_mask(ds.train.targets)]
    val_healthy = ds.val.windows[_healthy_mask(ds.val.targets)]

    model = _train_ae(
        train_healthy,
        val_healthy,
        n_features=n_features,
        window_size=window_size,
        latent_dim=latent_dim,
        seed=seed,
        max_epochs=SWEEP_MAX_EPOCHS,
        patience=SWEEP_PATIENCE,
    )
    train_scores = reconstruction_error(model, train_healthy)
    val_scores = reconstruction_error(model, ds.val.windows)
    val_labels = _is_anomaly_labels(ds.val.targets)
    threshold = percentile_threshold(train_scores, 99)["threshold"]
    val_pred = (val_scores >= threshold).astype(int)
    return classification_metrics(val_labels, val_pred, val_scores, include_brier=False)


def bottleneck_ablation(
    seeds: tuple[int, ...] = SWEEP_SEEDS,
    subset: str = SUBSET,
    processed_dir: Path | None = None,
) -> dict[int, dict]:
    """F1 (validation, percentile-99-of-healthy-train-scores threshold — the shippable,
    label-free operating point) for each bottleneck size, over the reduced-budget seeds. Looking
    for the point where a wider bottleneck starts reconstructing anomalies as well as healthy
    windows, collapsing detection."""
    processed_dir = Path(settings.data.processed_dir) if processed_dir is None else processed_dir
    results: dict[int, dict] = {}
    for latent_dim in BOTTLENECK_DIMS:
        runs = [
            _run_one_bottleneck_config(latent_dim, seed, subset, processed_dir) for seed in seeds
        ]
        results[latent_dim] = aggregate_over_seeds(runs, seeds)
    return results


# --- deliverable #3: Isolation Forest contamination sensitivity -------------------------------


def _run_one_contamination_config(
    contamination: float, seed: int, subset: str, processed_dir: Path
) -> dict:
    """One (contamination, seed) point of the sweep — see
    :func:`_run_one_bottleneck_config`'s docstring for why this is split out."""
    ds = _load_cmapss_split(seed, subset, processed_dir)
    train_healthy = ds.train.windows[_healthy_mask(ds.train.targets)]
    model = fit_isolation_forest(train_healthy, seed=seed, contamination=contamination)

    val_labels = _is_anomaly_labels(ds.val.targets)
    val_scores = isolation_forest_scores(model, ds.val.windows)
    hard_pred = (model.predict(make_baseline_features(ds.val.windows)) == -1).astype(int)
    return classification_metrics(val_labels, hard_pred, val_scores, include_brier=False)


def isolation_forest_contamination_sweep(
    seeds: tuple[int, ...] = SWEEP_SEEDS,
    grid: tuple[float, ...] = CONTAMINATION_GRID,
    subset: str = SUBSET,
    processed_dir: Path | None = None,
) -> dict[str, dict]:
    """F1 (validation, IF's own ``contamination``-implied operating point via ``model.predict``)
    for each contamination setting — how sensitive the classical baseline is to a hyperparameter
    that, unlike the AE's percentile threshold, is not derived from the score distribution."""
    processed_dir = Path(settings.data.processed_dir) if processed_dir is None else processed_dir
    results: dict[str, dict] = {}
    for contamination in grid:
        runs = [
            _run_one_contamination_config(contamination, seed, subset, processed_dir)
            for seed in seeds
        ]
        results[str(contamination)] = aggregate_over_seeds(runs, seeds)
    return results


# --- deliverable #4: fusion weight sweep -------------------------------------------------------


def fusion_weight_sweep(
    latent_dim: int,
    seeds: tuple[int, ...] = SWEEP_SEEDS,
    weights: tuple[float, ...] = FUSION_WEIGHTS,
    subset: str = SUBSET,
    processed_dir: Path | None = None,
) -> dict[float, dict]:
    """Max-F1-on-validation for each fusion weight (``1.0`` = LSTM-AE alone, ``0.0`` = Isolation
    Forest alone), both detectors' scores standardised on their own healthy-training-score
    statistics before combining."""
    processed_dir = Path(settings.data.processed_dir) if processed_dir is None else processed_dir
    runs_by_weight: dict[float, list[dict]] = {w: [] for w in weights}

    for seed in seeds:
        ds = _load_cmapss_split(seed, subset, processed_dir)
        n_features = len(ds.feature_spec.feature_names)
        window_size = ds.train.windows.shape[1]
        train_healthy = ds.train.windows[_healthy_mask(ds.train.targets)]
        val_healthy = ds.val.windows[_healthy_mask(ds.val.targets)]

        ae = _train_ae(
            train_healthy,
            val_healthy,
            n_features=n_features,
            window_size=window_size,
            latent_dim=latent_dim,
            seed=seed,
            max_epochs=SWEEP_MAX_EPOCHS,
            patience=SWEEP_PATIENCE,
        )
        if_model = fit_isolation_forest(train_healthy, seed=seed)

        ae_std = ScoreStandardizer.fit(reconstruction_error(ae, train_healthy))
        if_std = ScoreStandardizer.fit(isolation_forest_scores(if_model, train_healthy))

        val_labels = _is_anomaly_labels(ds.val.targets)
        ae_val = ae_std.transform(reconstruction_error(ae, ds.val.windows))
        if_val = if_std.transform(isolation_forest_scores(if_model, ds.val.windows))

        for w in weights:
            fused = fuse_scores(ae_val, if_val, w)
            best = _numeric_only(max_f1_threshold(val_labels, fused, split="validation"))
            runs_by_weight[w].append(best)

    return {w: aggregate_over_seeds(runs, seeds) for w, runs in runs_by_weight.items()}


# --- deliverables #5/#6: final 5-seed evaluation -----------------------------------------------


def final_cmapss_evaluation(
    latent_dim: int,
    fusion_weight: float,
    seeds: tuple[int, ...] | None = None,
    subset: str = SUBSET,
    processed_dir: Path | None = None,
) -> dict:
    """The shipped configuration's headline numbers on C-MAPSS: AE-alone and Isolation-Forest-
    alone (each at its own percentile-99 threshold), and the fused score under all three
    threshold styles (deliverable #5), all scored on **test**, over the full 5-seed budget."""
    seeds = settings.training.seeds if seeds is None else seeds
    processed_dir = Path(settings.data.processed_dir) if processed_dir is None else processed_dir

    ae_runs: list[dict] = []
    if_runs: list[dict] = []
    threshold_runs: dict[str, list[dict]] = {
        "max_f1": [],
        "precision_target": [],
        "percentile_99": [],
    }
    mlflow_run_ids: dict[int, str] = {}

    for seed in seeds:
        ds = _load_cmapss_split(seed, subset, processed_dir)
        n_features = len(ds.feature_spec.feature_names)
        window_size = ds.train.windows.shape[1]
        train_healthy = ds.train.windows[_healthy_mask(ds.train.targets)]
        val_healthy = ds.val.windows[_healthy_mask(ds.val.targets)]

        ae = _train_ae(
            train_healthy,
            val_healthy,
            n_features=n_features,
            window_size=window_size,
            latent_dim=latent_dim,
            seed=seed,
            max_epochs=settings.training.max_epochs,
            patience=settings.training.early_stopping_patience,
        )
        if_model = fit_isolation_forest(train_healthy, seed=seed)

        ae_train_scores = reconstruction_error(ae, train_healthy)
        if_train_scores = isolation_forest_scores(if_model, train_healthy)
        ae_std = ScoreStandardizer.fit(ae_train_scores)
        if_std = ScoreStandardizer.fit(if_train_scores)
        fused_train_scores = fuse_scores(
            ae_std.transform(ae_train_scores), if_std.transform(if_train_scores), fusion_weight
        )

        val_labels = _is_anomaly_labels(ds.val.targets)
        fused_val = fuse_scores(
            ae_std.transform(reconstruction_error(ae, ds.val.windows)),
            if_std.transform(isolation_forest_scores(if_model, ds.val.windows)),
            fusion_weight,
        )
        thr_max_f1 = max_f1_threshold(val_labels, fused_val, split="validation")["threshold"]
        thr_precision = threshold_at_precision(
            val_labels,
            fused_val,
            settings.training.baseline_precision_target,
            split="validation",
            include_brier=False,
        )["threshold"]
        thr_percentile = percentile_threshold(fused_train_scores, 99)["threshold"]

        test_labels = _is_anomaly_labels(ds.test.targets)
        ae_test = reconstruction_error(ae, ds.test.windows)
        if_test = isolation_forest_scores(if_model, ds.test.windows)
        fused_test = fuse_scores(
            ae_std.transform(ae_test), if_std.transform(if_test), fusion_weight
        )

        ae_thr = percentile_threshold(ae_train_scores, 99)["threshold"]
        if_thr = percentile_threshold(if_train_scores, 99)["threshold"]
        ae_runs.append(
            classification_metrics(
                test_labels, (ae_test >= ae_thr).astype(int), ae_test, include_brier=False
            )
        )
        if_runs.append(
            classification_metrics(
                test_labels, (if_test >= if_thr).astype(int), if_test, include_brier=False
            )
        )

        seed_metrics: dict[str, dict] = {
            "ae_alone": ae_runs[-1],
            "isolation_forest_alone": if_runs[-1],
        }
        for name, thr in (
            ("max_f1", thr_max_f1),
            ("precision_target", thr_precision),
            ("percentile_99", thr_percentile),
        ):
            pred = (fused_test >= thr).astype(int)
            fused_metrics = classification_metrics(
                test_labels, pred, fused_test, include_brier=False
            )
            threshold_runs[name].append(fused_metrics)
            seed_metrics[f"fused_{name}"] = fused_metrics

        with start_run(
            run_name=f"anomaly__cmapss__{subset}__seed{seed}",
            framework="pytorch",
            seed=seed,
            dataset_hash=_cmapss_dataset_hash(processed_dir, subset),
            extra_tags={"task": "anomaly_detection", "dataset": "cmapss", "subset": subset},
        ) as run:
            log_params(
                {
                    "latent_dim": latent_dim,
                    "fusion_weight": fusion_weight,
                    "subset": subset,
                    "n_features": n_features,
                }
            )
            for prefix, m in seed_metrics.items():
                log_metrics({f"{prefix}_{k}": v for k, v in m.items()})
            mlflow_run_ids[seed] = run.info.run_id

    return {
        "ae_alone": aggregate_over_seeds(ae_runs, seeds),
        "isolation_forest_alone": aggregate_over_seeds(if_runs, seeds),
        "fused_by_threshold_style": {
            name: aggregate_over_seeds(runs, seeds) for name, runs in threshold_runs.items()
        },
        # One MLflow run per seed covers ae_alone/isolation_forest_alone/fused_* together (all
        # logged inside the same `start_run` block above) — every number this function returns
        # is traceable to one of these, per the P07 spec constraint (docs/P07-mlflow.md).
        "mlflow_run_ids": mlflow_run_ids,
    }


def final_nab_evaluation(
    latent_dim: int,
    fusion_weight: float,
    seeds: tuple[int, ...] | None = None,
    nab_processed_path: Path | None = None,
) -> dict:
    """The shipped configuration's headline numbers on NAB: no validation split exists to tune a
    max-F1/precision-target threshold honestly (see `pdm.preprocessing.nab_pipeline`'s module
    docstring), so every detector here ships its percentile-99-of-healthy-training-scores
    threshold — the one threshold style that needs no labels at all."""
    seeds = settings.training.seeds if seeds is None else seeds
    nab_processed_path = (
        Path(settings.data.processed_dir) / "machine_temperature_system_failure.parquet"
        if nab_processed_path is None
        else nab_processed_path
    )
    nab = prepare_nab_windows(nab_processed_path)
    n_features = nab.train_windows.shape[2]
    window_size = nab.train_windows.shape[1]

    ae_runs: list[dict] = []
    if_runs: list[dict] = []
    fused_runs: list[dict] = []
    mlflow_run_ids: dict[int, str] = {}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(nab.train_windows))
        n_val = max(1, int(round(0.2 * len(idx))))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        train_healthy, val_healthy = nab.train_windows[train_idx], nab.train_windows[val_idx]

        ae = _train_ae(
            train_healthy,
            val_healthy,
            n_features=n_features,
            window_size=window_size,
            latent_dim=latent_dim,
            seed=seed,
            max_epochs=settings.training.max_epochs,
            patience=settings.training.early_stopping_patience,
        )
        if_model = fit_isolation_forest(train_healthy, seed=seed)

        ae_train_scores = reconstruction_error(ae, train_healthy)
        if_train_scores = isolation_forest_scores(if_model, train_healthy)
        ae_std = ScoreStandardizer.fit(ae_train_scores)
        if_std = ScoreStandardizer.fit(if_train_scores)

        ae_test = reconstruction_error(ae, nab.test_windows)
        if_test = isolation_forest_scores(if_model, nab.test_windows)
        fused_test = fuse_scores(
            ae_std.transform(ae_test), if_std.transform(if_test), fusion_weight
        )

        ae_thr = percentile_threshold(ae_train_scores, 99)["threshold"]
        if_thr = percentile_threshold(if_train_scores, 99)["threshold"]
        fused_train = fuse_scores(
            ae_std.transform(ae_train_scores), if_std.transform(if_train_scores), fusion_weight
        )
        fused_thr = percentile_threshold(fused_train, 99)["threshold"]

        ae_runs.append(
            classification_metrics(
                nab.test_labels, (ae_test >= ae_thr).astype(int), ae_test, include_brier=False
            )
        )
        if_runs.append(
            classification_metrics(
                nab.test_labels, (if_test >= if_thr).astype(int), if_test, include_brier=False
            )
        )
        fused_runs.append(
            classification_metrics(
                nab.test_labels,
                (fused_test >= fused_thr).astype(int),
                fused_test,
                include_brier=False,
            )
        )

        with start_run(
            run_name=f"anomaly__nab__seed{seed}",
            framework="pytorch",
            seed=seed,
            dataset_hash=sha256_of_files([nab_processed_path]),
            extra_tags={"task": "anomaly_detection", "dataset": "nab"},
        ) as run:
            log_params(
                {
                    "latent_dim": latent_dim,
                    "fusion_weight": fusion_weight,
                    "n_features": n_features,
                }
            )
            for prefix, m in (
                ("ae_alone", ae_runs[-1]),
                ("isolation_forest_alone", if_runs[-1]),
                ("fused", fused_runs[-1]),
            ):
                log_metrics({f"{prefix}_{k}": v for k, v in m.items()})
            mlflow_run_ids[seed] = run.info.run_id

    return {
        "ae_alone": aggregate_over_seeds(ae_runs, seeds),
        "isolation_forest_alone": aggregate_over_seeds(if_runs, seeds),
        "fused": aggregate_over_seeds(fused_runs, seeds),
        "mlflow_run_ids": mlflow_run_ids,
    }
