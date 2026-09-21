"""Bundle loading for the serving package (P08 deliverable #3).

Loads exactly the artifacts a training path wrote (spec §5's bundle contract) — nothing here
recomputes a scaler, a feature list, or a threshold from raw data. **Fails loudly** at load
time (raising :class:`BundleValidationError`) if a required file is missing, unreadable, or
internally inconsistent (a reconstructed model's parameter count disagreeing with
``metadata.json``'s recorded count, or a forward pass on a `feature_spec`-shaped dummy window
raising) — a service that boots with a broken model and serves garbage is worse than one that
won't boot at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pdm.config import settings
from pdm.preprocessing.scaling import StandardScaler
from pdm.preprocessing.windowing import FeatureSpec

REQUIRED_FILES = (
    "feature_spec.json",
    "scaler.json",
    "thresholds.json",
    "metrics.json",
    "metadata.json",
)


class BundleValidationError(RuntimeError):
    """A bundle directory is missing a required file, unreadable, or internally inconsistent.
    Always fatal to whoever is loading the bundle — never caught and silently downgraded into a
    partially-working model."""


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise BundleValidationError(f"missing required bundle file: {path}")
    return path


def _read_json(path: Path) -> dict:
    _require_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleValidationError(f"{path} is not valid JSON: {exc}") from exc


def _load_feature_spec(run_dir: Path) -> FeatureSpec:
    path = _require_file(run_dir / "feature_spec.json")
    try:
        return FeatureSpec.load(path)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BundleValidationError(f"{path} is not a valid feature_spec.json: {exc}") from exc


def _load_scaler(run_dir: Path) -> StandardScaler:
    path = _require_file(run_dir / "scaler.json")
    try:
        return StandardScaler.load(path)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BundleValidationError(f"{path} is not a valid scaler.json: {exc}") from exc


def _build_torch_regressor(n_features: int, architecture: dict | None):
    import torch

    from pdm.models.torch.architecture import LSTMRegressor, count_parameters

    hidden_sizes = tuple(
        (architecture or {}).get("hidden_sizes") or settings.model.lstm_hidden_sizes
    )
    dropout = (architecture or {}).get("dropout", settings.model.dropout)
    model = LSTMRegressor(n_features=n_features, hidden_sizes=hidden_sizes, dropout=dropout)
    model.eval()

    def predict(windows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.as_tensor(windows, dtype=torch.float32)).numpy()

    return model, predict, count_parameters(model)


def _build_tf_regressor(n_features: int, architecture: dict | None):
    import tensorflow as tf

    from pdm.models.tf.architecture import LSTMRegressor, count_parameters

    hidden_sizes = tuple(
        (architecture or {}).get("hidden_sizes") or settings.model.lstm_hidden_sizes
    )
    dropout = (architecture or {}).get("dropout", settings.model.dropout)
    model = LSTMRegressor(n_features=n_features, hidden_sizes=hidden_sizes, dropout=dropout)

    def predict(windows: np.ndarray) -> np.ndarray:
        return model(tf.convert_to_tensor(windows, dtype=tf.float32), training=False).numpy()

    return model, predict, count_parameters(model)


def _build_torch_autoencoder(n_features: int, window_size: int, architecture: dict | None):
    import torch

    from pdm.models.torch.architecture import count_parameters
    from pdm.models.torch.autoencoder import LSTMAutoencoder, per_sensor_reconstruction_error

    latent_dim = (architecture or {}).get("latent_dim", settings.model.ae_bottleneck_dim)
    model = LSTMAutoencoder(n_features=n_features, window_size=window_size, latent_dim=latent_dim)
    model.eval()

    def score(windows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(windows, dtype=torch.float32)
            recon = model(x)
            return ((recon - x) ** 2).mean(dim=(1, 2)).numpy()

    def per_sensor(windows: np.ndarray) -> np.ndarray:
        return per_sensor_reconstruction_error(model, windows)

    return model, score, per_sensor, count_parameters(model)


def _build_tf_autoencoder(n_features: int, window_size: int, architecture: dict | None):
    import tensorflow as tf

    from pdm.models.tf.architecture import count_parameters
    from pdm.models.tf.autoencoder import LSTMAutoencoder

    latent_dim = (architecture or {}).get("latent_dim", settings.model.ae_bottleneck_dim)
    model = LSTMAutoencoder(n_features=n_features, window_size=window_size, latent_dim=latent_dim)

    def score(windows: np.ndarray) -> np.ndarray:
        x = tf.convert_to_tensor(windows, dtype=tf.float32)
        recon = model(x, training=False)
        return tf.reduce_mean(tf.square(recon - x), axis=[1, 2]).numpy()

    def per_sensor(windows: np.ndarray) -> np.ndarray:
        x = tf.convert_to_tensor(windows, dtype=tf.float32)
        recon = model(x, training=False)
        return tf.reduce_mean(tf.square(recon - x), axis=1).numpy()

    return model, score, per_sensor, count_parameters(model)


@dataclass
class RulBundle:
    run_dir: Path
    framework: str
    feature_spec: FeatureSpec
    scaler: StandardScaler
    thresholds: dict
    metrics: dict
    metadata: dict
    _predict_fn: Callable[[np.ndarray], np.ndarray]

    @property
    def model_version(self) -> str:
        sha = (self.metadata.get("git_sha") or "nogit")[:8]
        return f"{self.metadata.get('framework')}-seed{self.metadata.get('seed')}-{sha}"

    @property
    def bundle_id(self) -> str:
        return self.run_dir.name

    def scale(self, window: np.ndarray) -> np.ndarray:
        return (window - self.scaler.mean_) / self.scaler.std_

    def predict_rul(self, windows: np.ndarray) -> np.ndarray:
        """``windows``: ``(N, window_size, n_features)``, already scaled. Returns ``(N,)``."""
        return self._predict_fn(windows)


@dataclass
class AnomalyBundle:
    run_dir: Path
    framework: str
    feature_spec: FeatureSpec
    scaler: StandardScaler
    thresholds: dict
    metrics: dict
    metadata: dict
    _score_fn: Callable[[np.ndarray], np.ndarray]
    _per_sensor_fn: Callable[[np.ndarray], np.ndarray]

    @property
    def model_version(self) -> str:
        sha = (self.metadata.get("git_sha") or "nogit")[:8]
        return f"{self.metadata.get('framework')}-seed{self.metadata.get('seed')}-{sha}"

    @property
    def threshold(self) -> float:
        value = self.thresholds.get("anomaly_score_threshold")
        if value is None:
            raise BundleValidationError(
                f"{self.run_dir / 'thresholds.json'} has no 'anomaly_score_threshold'"
            )
        return float(value)

    def scale(self, window: np.ndarray) -> np.ndarray:
        return (window - self.scaler.mean_) / self.scaler.std_

    def score(self, windows: np.ndarray) -> np.ndarray:
        return self._score_fn(windows)

    def per_sensor_error(self, windows: np.ndarray) -> np.ndarray:
        return self._per_sensor_fn(windows)


def _validate_feature_shape(
    feature_spec: FeatureSpec, predict_or_score: Callable[[np.ndarray], np.ndarray], run_dir: Path
) -> None:
    dummy = np.zeros(
        (1, feature_spec.window_size, len(feature_spec.feature_names)), dtype=np.float32
    )
    try:
        predict_or_score(dummy)
    except Exception as exc:  # noqa: BLE001 - any failure here means the bundle is unusable
        raise BundleValidationError(
            f"model in {run_dir} rejected a dummy window shaped "
            f"(1, {feature_spec.window_size}, {len(feature_spec.feature_names)}) built from its "
            f"own feature_spec.json: {exc}"
        ) from exc


def _validate_parameter_count(actual: int, metadata: dict, run_dir: Path) -> None:
    expected = metadata.get("parameter_count")
    if expected is not None and int(expected) != actual:
        raise BundleValidationError(
            f"{run_dir}: reconstructed model has {actual} parameters, but metadata.json "
            f"recorded {expected} — the architecture used to reconstruct this model (from "
            f"metadata.json's 'architecture' field, or the current config defaults if that "
            f"field is absent) does not match what this bundle was actually trained with."
        )


def load_rul_bundle(bundle_dir: str | Path) -> RulBundle:
    run_dir = Path(bundle_dir)
    if not run_dir.exists():
        raise BundleValidationError(f"bundle directory does not exist: {run_dir}")

    metadata = _read_json(run_dir / "metadata.json")
    feature_spec = _load_feature_spec(run_dir)
    scaler = _load_scaler(run_dir)
    thresholds = _read_json(run_dir / "thresholds.json")
    metrics = _read_json(run_dir / "metrics.json")

    framework = metadata.get("framework")
    n_features = len(feature_spec.feature_names)
    architecture = metadata.get("architecture")

    model: Any
    if framework == "pytorch":
        model, predict_fn, param_count = _build_torch_regressor(n_features, architecture)
        from pdm.models.torch.bundle import load_torch_model_state

        try:
            load_torch_model_state(run_dir, model)
        except Exception as exc:  # noqa: BLE001
            raise BundleValidationError(
                f"failed to load model weights from {run_dir}: {exc}"
            ) from exc
        model.eval()
    elif framework == "tensorflow":
        model, predict_fn, param_count = _build_tf_regressor(n_features, architecture)
        from pdm.models.tf.bundle import load_tf_model_weights

        try:
            load_tf_model_weights(run_dir, model)
        except Exception as exc:  # noqa: BLE001
            raise BundleValidationError(
                f"failed to load model weights from {run_dir}: {exc}"
            ) from exc
    else:
        raise BundleValidationError(f"{run_dir}/metadata.json: unknown framework {framework!r}")

    _validate_parameter_count(param_count, metadata, run_dir)
    _validate_feature_shape(feature_spec, predict_fn, run_dir)

    return RulBundle(
        run_dir=run_dir,
        framework=framework,
        feature_spec=feature_spec,
        scaler=scaler,
        thresholds=thresholds,
        metrics=metrics,
        metadata=metadata,
        _predict_fn=predict_fn,
    )


def load_anomaly_bundle(bundle_dir: str | Path) -> AnomalyBundle:
    run_dir = Path(bundle_dir)
    if not run_dir.exists():
        raise BundleValidationError(f"bundle directory does not exist: {run_dir}")

    metadata = _read_json(run_dir / "metadata.json")
    feature_spec = _load_feature_spec(run_dir)
    scaler = _load_scaler(run_dir)
    thresholds = _read_json(run_dir / "thresholds.json")
    metrics = _read_json(run_dir / "metrics.json")

    framework = metadata.get("framework")
    n_features = len(feature_spec.feature_names)
    window_size = feature_spec.window_size
    architecture = metadata.get("architecture")

    if framework == "pytorch":
        model, score_fn, per_sensor_fn, param_count = _build_torch_autoencoder(
            n_features, window_size, architecture
        )
        from pdm.models.torch.bundle import load_torch_model_state

        try:
            load_torch_model_state(run_dir, model)
        except Exception as exc:  # noqa: BLE001
            raise BundleValidationError(
                f"failed to load model weights from {run_dir}: {exc}"
            ) from exc
        model.eval()
    elif framework == "tensorflow":
        model, score_fn, per_sensor_fn, param_count = _build_tf_autoencoder(
            n_features, window_size, architecture
        )
        from pdm.models.tf.bundle import load_tf_model_weights

        try:
            load_tf_model_weights(run_dir, model)
        except Exception as exc:  # noqa: BLE001
            raise BundleValidationError(
                f"failed to load model weights from {run_dir}: {exc}"
            ) from exc
    else:
        raise BundleValidationError(f"{run_dir}/metadata.json: unknown framework {framework!r}")

    _validate_parameter_count(param_count, metadata, run_dir)
    _validate_feature_shape(feature_spec, score_fn, run_dir)
    if thresholds.get("anomaly_score_threshold") is None:
        raise BundleValidationError(f"{run_dir}/thresholds.json has no 'anomaly_score_threshold'")

    return AnomalyBundle(
        run_dir=run_dir,
        framework=framework,
        feature_spec=feature_spec,
        scaler=scaler,
        thresholds=thresholds,
        metrics=metrics,
        metadata=metadata,
        _score_fn=score_fn,
        _per_sensor_fn=per_sensor_fn,
    )
