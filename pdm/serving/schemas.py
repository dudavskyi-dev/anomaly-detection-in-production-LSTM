"""Pydantic request/response models (P08 deliverable #2). Validation here catches shape and
numeric problems that are true regardless of which bundle is loaded (rectangular windows, no
NaN/Inf, non-empty); validation that depends on *which* bundle is loaded (exact window length,
feature count, and feature **order**) happens in the endpoint handlers against the loaded
bundle's ``feature_spec.json``, since a pydantic model can't see runtime state. Every rejection
names the offending field and what was expected — never a bare stack trace.
"""

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _ApiModel(BaseModel):
    """Every response model that has a ``model_version`` field inherits this instead of
    ``BaseModel`` directly — pydantic reserves the ``model_`` prefix for its own methods by
    default and warns on every field named that way otherwise; the field name matches this
    project's own vocabulary (used identically in bundle metadata) far too well to rename just
    to dodge the warning.
    """

    model_config = ConfigDict(protected_namespaces=())


def _check_rectangular_and_finite(
    window: list[list[float]], *, field_name: str
) -> list[list[float]]:
    if not window:
        raise ValueError(f"{field_name} must not be empty")
    row_len = len(window[0])
    if row_len == 0:
        raise ValueError(f"{field_name} rows must not be empty")
    for i, row in enumerate(window):
        if len(row) != row_len:
            raise ValueError(
                f"{field_name} is not rectangular: row 0 has {row_len} values, "
                f"row {i} has {len(row)}"
            )
        for j, value in enumerate(row):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{field_name}[{i}][{j}] is {value!r} — NaN/Inf is not allowed")
    return window


class WindowRequest(BaseModel):
    """A single sensor window: ``feature_names`` states the column order the client is
    claiming, and ``window`` (``window_size`` rows × ``len(feature_names)`` columns, oldest
    reading first) carries the values in that same order. The endpoint checks ``feature_names``
    against the loaded bundle's own ``feature_spec.json`` — sending the right values in the
    wrong order is a silent, dangerous bug this check exists specifically to catch loudly."""

    feature_names: list[str] = Field(
        ...,
        min_length=1,
        description="Feature column order, must match the bundle's feature_spec.json exactly.",
    )
    window: list[list[float]] = Field(
        ..., description="window_size rows x len(feature_names) columns, oldest reading first."
    )

    @field_validator("window")
    @classmethod
    def _validate_window(cls, value: list[list[float]]) -> list[list[float]]:
        return _check_rectangular_and_finite(value, field_name="window")


class BatchWindowRequest(BaseModel):
    feature_names: list[str] = Field(..., min_length=1)
    windows: list[list[list[float]]] = Field(..., min_length=1, max_length=10_000)

    @field_validator("windows")
    @classmethod
    def _validate_windows(cls, value: list[list[list[float]]]) -> list[list[list[float]]]:
        for i, window in enumerate(value):
            _check_rectangular_and_finite(window, field_name=f"windows[{i}]")
        return value


class RulPrediction(_ApiModel):
    predicted_rul: float
    failure_probability: float
    alert: bool
    model_version: str
    latency_ms: float


class BatchRulPredictionResponse(BaseModel):
    predictions: list[RulPrediction]
    latency_ms: float


class AnomalyDetection(_ApiModel):
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    per_sensor_contribution: dict[str, float]
    model_version: str
    latency_ms: float


class HealthResponse(_ApiModel):
    status: str
    model_version: str | None
    bundle_id: str | None
    anomaly_model_loaded: bool
    uptime_seconds: float


class ReadyResponse(BaseModel):
    ready: bool
    reason: str | None = None
