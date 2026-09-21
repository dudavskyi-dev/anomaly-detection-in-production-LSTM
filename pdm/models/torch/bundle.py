"""Writes/loads a complete model bundle (spec §5) for a trained PyTorch model."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from pdm.config import settings
from pdm.models.bundle import compute_reference_stats, write_json, write_metadata
from pdm.preprocessing.scaling import StandardScaler
from pdm.preprocessing.windowing import FeatureSpec


def save_torch_bundle(
    run_dir: Path,
    model: nn.Module,
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
    torch.save(model.state_dict(), run_dir / "model_torch.pt")
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
        framework="pytorch",
        framework_version=torch.__version__,
        seed=seed,
        parameter_count=parameter_count,
        dataset_version=dataset_version,
        mlflow_run_id=mlflow_run_id,
        architecture=architecture,
    )


def load_torch_model_state(run_dir: Path, model: nn.Module) -> nn.Module:
    """Load a bundle's ``model_torch.pt`` weights into an already-constructed model instance
    (the caller must build the model with the same architecture the bundle was saved with —
    ``feature_spec.json`` records the feature count needed to do that correctly)."""
    state_dict = torch.load(run_dir / "model_torch.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model
