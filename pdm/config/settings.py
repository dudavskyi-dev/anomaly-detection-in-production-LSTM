"""Nested, environment-driven application settings — the single source of defaults.

Every numeric default named in ``docs/SPEC.md`` lives here. Override any leaf value with an
environment variable of the form ``PDM__<SECTION>__<KEY>``, e.g. ``PDM__DATA__WINDOW_SIZE=50``.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseModel):
    """Windowing, labelling, and split parameters shared by preprocessing and every model."""

    window_size: int = 30
    stride: int = 1
    rul_cap: int = 125
    failure_horizon_w: int = 30
    val_fraction: float = 0.2
    n_operating_conditions: int = 6
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    constant_sensor_std_threshold: float = 1e-6


class ModelSettings(BaseModel):
    """Shared LSTM architecture, identical across the PyTorch and TensorFlow implementations."""

    lstm_hidden_sizes: tuple[int, int] = (100, 50)
    dropout: float = 0.2
    ae_bottleneck_dim: int = 8
    healthy_rul_threshold: int = 125


class TrainingSettings(BaseModel):
    """Optimisation hyperparameters, applied identically to both frameworks."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    baseline_ridge_alpha: float = 1.0
    baseline_rf_n_estimators: int = 100
    baseline_rf_max_depth: int = 12
    baseline_precision_target: float = 0.9


class ServingSettings(BaseModel):
    """FastAPI serving configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    bundle_dir: str = "artifacts/latest"
    # optional: the LSTM-autoencoder bundle backing /detect/anomaly. Unset (or unloadable) means
    # /detect/anomaly returns 503 while /predict/rul keeps working — the graceful-degradation
    # path the spec (P08 deliverable #4) explicitly asks for.
    anomaly_bundle_dir: str | None = None
    max_batch_size: int = 64


class MonitoringSettings(BaseModel):
    """Drift detection thresholds and retraining trigger configuration."""

    psi_moderate_threshold: float = 0.1
    psi_significant_threshold: float = 0.25
    # Quantile-based bins on the training distribution: equal training-side frequency per bin
    # is what keeps the reference side of the PSI ratio away from zero (see
    # pdm/monitoring/drift.py).
    psi_n_bins: int = 10
    # Two-sample KS needs actual samples, not just histogram counts — this many training values
    # (a fixed random subsample, seeded, drawn once at bundle-save time) are frozen into
    # reference_stats.json per feature so pdm.monitoring.drift can run scipy's ks_2samp without
    # re-reading the original training parquet.
    ks_reference_sample_size: int = 2000
    ks_alpha: float = 0.05
    ks_correction: str = "benjamini_hochberg"
    drift_check_interval_seconds: int = 3600
    retrain_cooldown_seconds: int = 86400
    # A retraining candidate must never reuse the exact (subset, seed) pair the currently-served
    # bundle was trained with — that would silently overwrite the live bundle's artifact
    # directory. Successive triggers add multiples of this offset to settings.seed.
    retrain_seed_offset: int = 1000
    state_path: str = "artifacts/monitoring/trigger_state.json"
    reports_dir: str = "reports/drift"
    metrics_textfile_path: str = "artifacts/monitoring/drift_metrics.prom"


class TrackingSettings(BaseModel):
    """MLflow tracking and registry configuration."""

    uri: str = "file:./mlruns"
    experiment_name: str = "pdm-sentinel"
    promotion_margin: float = 0.02
    # every training path logs metrics prefixed by the split that produced them (``val_rmse``,
    # ``test_rmse``, ...) — never a bare ``rmse`` — so the gate's default target must match.
    promotion_primary_metric: str = "test_rmse"


class Settings(BaseSettings):
    """Root settings object. Populated from defaults, overridable via ``PDM__SECTION__KEY``."""

    model_config = SettingsConfigDict(
        env_prefix="PDM__",
        env_nested_delimiter="__",
        extra="forbid",
    )

    seed: int = 42
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    serving: ServingSettings = Field(default_factory=ServingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)


settings = Settings()
