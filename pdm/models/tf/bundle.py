"""Writes/loads a complete model bundle (spec §5) for a trained TensorFlow model, mirroring
``pdm.models.torch.bundle`` field-for-field so both frameworks' bundles are interchangeable
inputs to later milestones (P07 tracking, P08 serving).

Saves weights only (``model_tf.weights.h5``), not a full SavedModel directory — the SavedModel
export (deliverable #6, the "production path") is a separate, larger artifact produced by
``pdm.evaluation.framework_benchmark``, and mixing the two into "model size on disk" would not
be a fair comparison against PyTorch's raw ``state_dict`` pickle.
"""

from pathlib import Path

import numpy as np
from tensorflow import __version__ as tf_version
from tensorflow import keras

from pdm.config import settings
from pdm.models.bundle import compute_reference_stats, write_json, write_metadata
from pdm.preprocessing.scaling import StandardScaler
from pdm.preprocessing.windowing import FeatureSpec


def save_tf_bundle(
    run_dir: Path,
    model: keras.Model,
    *,
    scaler: StandardScaler,
    feature_spec: FeatureSpec,
    train_windows: np.ndarray,
    metrics: dict,
    thresholds: dict,
    seed: int,
    dataset_version: str,
    parameter_count: int,
    mlflow_run_id: str | None = None,
    architecture: dict | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(run_dir / "model_tf.weights.h5"))
    scaler.save(run_dir / "scaler.json")
    feature_spec.save(run_dir / "feature_spec.json")
    write_json(run_dir / "thresholds.json", thresholds)
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "reference_stats.json",
        compute_reference_stats(
            train_windows,
            feature_spec.feature_names,
            n_bins=settings.monitoring.psi_n_bins,
            reference_sample_size=settings.monitoring.ks_reference_sample_size,
        ),
    )
    write_metadata(
        run_dir / "metadata.json",
        framework="tensorflow",
        framework_version=tf_version,
        seed=seed,
        parameter_count=parameter_count,
        dataset_version=dataset_version,
        mlflow_run_id=mlflow_run_id,
        architecture=architecture,
    )


def load_tf_model_weights(run_dir: Path, model: keras.Model) -> keras.Model:
    """Load a bundle's ``model_tf.weights.h5`` weights into an already-constructed, already-built
    model instance (the caller must build the model with the same architecture the bundle was
    saved with — ``feature_spec.json`` records the feature count needed to do that correctly)."""
    model.load_weights(str(run_dir / "model_tf.weights.h5"))
    return model
