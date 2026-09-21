"""The versioned model-bundle contract (spec §5), shared by every training milestone (P04
PyTorch, P05 TensorFlow, P06 anomaly detection): a run writes one of these, and serving loads
exactly one, never recomputing anything from training data.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def compute_reference_stats(
    windows: np.ndarray,
    feature_names: list[str],
    *,
    n_bins: int = 10,
    reference_sample_size: int = 2000,
    sample_seed: int = 0,
) -> dict:
    """Per-feature distribution stats from the **train** split, for P09's drift comparison.

    Every timestep of every training window is treated as one sample of that feature — this is
    the same flattening ``pdm.monitoring.drift`` applies to production windows, so the two sides
    of a comparison are always in the same units.

    Two things are frozen here specifically because P09's drift checks must reuse them
    unchanged, potentially long after training and in a different process:

    - ``bin_edges``/``bin_counts``: a **quantile-based** (equal-frequency) histogram of the
      training column, computed once. Quantile binning — rather than equal-width — is what
      keeps every reference bin non-empty by construction (each holds ~1/n_bins of the training
      data), which matters because PSI's ratio has the reference proportion in the denominator;
      an empty equal-width bin at the tail of a skewed sensor would make that ratio divide by
      zero before any production data is even involved. The outermost edges are pinned to
      ``-inf``/``+inf`` so a production value from a genuinely new range is never silently
      dropped from the count instead of showing up as drift.
    - ``reference_sample``: a fixed-size random subsample of the raw (not binned) training
      values, seeded for reproducibility. The two-sample KS test needs actual samples on both
      sides, not just bin counts — storing the full training column would work too but bloats
      the bundle for no accuracy benefit past a few thousand points; ``reference_sample_size``
      is deliberately small and separate from ``bin_counts``, which uses the *full* training
      column for PSI's accuracy.
    """
    flat = windows.reshape(-1, windows.shape[-1])
    n_samples = int(flat.shape[0])
    rng = np.random.default_rng(sample_seed)
    features = {}
    for i, name in enumerate(feature_names):
        col = flat[:, i]
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(col, quantiles)
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(col, bins=edges)
        sample_size = min(reference_sample_size, col.shape[0])
        sample = rng.choice(col, size=sample_size, replace=False)
        features[name] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "q01": float(np.percentile(col, 1)),
            "q50": float(np.percentile(col, 50)),
            "q99": float(np.percentile(col, 99)),
            "bin_edges": [float(e) for e in edges],
            "bin_counts": [int(c) for c in counts],
            "reference_sample": [float(v) for v in sample],
        }
    return {
        "n_windows": int(windows.shape[0]),
        "n_samples_per_feature": n_samples,
        "n_bins": n_bins,
        "features": features,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_metadata(
    path: Path,
    *,
    framework: str,
    framework_version: str,
    seed: int,
    parameter_count: int,
    dataset_version: str,
    mlflow_run_id: str | None = None,
    architecture: dict | None = None,
) -> None:
    """``mlflow_run_id`` stays ``null`` until P07 wires up the tracking server — every bundle
    written before then is still fully valid and loadable, just not yet registered anywhere.

    ``architecture`` (added in P08) records exactly the hyperparameters needed to reconstruct
    the model class before loading its state dict (``hidden_sizes``/``dropout`` for the
    regressor/classifier, ``latent_dim`` for the autoencoder) — without it, a loader has to
    assume the *current* config's defaults still match what a given run was actually trained
    with, which is exactly the kind of silent training/serving skew this project's bundle
    contract exists to rule out. ``None`` for bundles written before this field existed;
    ``pdm.serving.bundle`` falls back to current config defaults for those, which is safe only
    because the regressor's architecture hasn't changed since — a real, documented limitation,
    not a robust general mechanism.
    """
    write_json(
        path,
        {
            "framework": framework,
            "framework_version": framework_version,
            "git_sha": git_sha(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "seed": seed,
            "parameter_count": parameter_count,
            "dataset_version": dataset_version,
            "mlflow_run_id": mlflow_run_id,
            "architecture": architecture,
        },
    )
