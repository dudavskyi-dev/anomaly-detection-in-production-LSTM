"""FastAPI inference service (P08 deliverable #1). Depends on the bundle format only —
``pdm.serving`` never imports a training-orchestration module (``pdm.preprocessing.pipeline``,
``pdm.models.baseline.runner``, ``pdm.models.torch.experiments``/``anomaly_experiments``,
``pdm.evaluation.framework_benchmark``, or anything under ``pdm.ingestion``); see
``tests/test_serving_import_graph.py`` and ``docs/decisions/P08-serving.md`` for exactly where
that line is drawn and why.
"""

import math
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pdm.config import settings
from pdm.serving.bundle import (
    AnomalyBundle,
    BundleValidationError,
    RulBundle,
    load_anomaly_bundle,
    load_rul_bundle,
)
from pdm.serving.logging_config import RequestLoggingMiddleware, configure_logging
from pdm.serving.metrics import (
    ALERTS_TOTAL,
    ANOMALIES_DETECTED,
    ANOMALY_SCORE,
    PREDICTED_RUL,
    PROCESS_START_TIME_SECONDS,
    render_metrics,
    set_model_info,
)
from pdm.serving.schemas import (
    AnomalyDetection,
    BatchRulPredictionResponse,
    BatchWindowRequest,
    HealthResponse,
    ReadyResponse,
    RulPrediction,
    WindowRequest,
)

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.start_time = time.time()
    PROCESS_START_TIME_SECONDS.set(app.state.start_time)
    app.state.rul_bundle = None
    app.state.anomaly_bundle = None
    app.state.anomaly_bundle_error = None

    # Required: fail loudly, refuse to start, if this doesn't load cleanly.
    app.state.rul_bundle = load_rul_bundle(settings.serving.bundle_dir)
    set_model_info(
        version=app.state.rul_bundle.model_version,
        framework=app.state.rul_bundle.framework,
        bundle_id=app.state.rul_bundle.bundle_id,
        trained_at_iso=app.state.rul_bundle.metadata.get("timestamp_utc"),
    )

    # Optional: missing or broken anomaly bundle degrades /detect/anomaly to a 503, never
    # blocks startup and never breaks /predict/rul.
    if settings.serving.anomaly_bundle_dir:
        try:
            app.state.anomaly_bundle = load_anomaly_bundle(settings.serving.anomaly_bundle_dir)
        except BundleValidationError as exc:
            app.state.anomaly_bundle_error = str(exc)
    else:
        app.state.anomaly_bundle_error = "PDM__SERVING__ANOMALY_BUNDLE_DIR is not configured"

    yield


app = FastAPI(title="PdM-Sentinel inference service", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """FastAPI's default 422 body is fine but verbose; this keeps the same status code and
    still names the offending field(s), just in a flatter, easier-to-read shape."""
    errors = [
        {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": "validation failed", "errors": errors})


def _require_feature_match(
    request_feature_names: list[str], bundle_feature_names: list[str]
) -> None:
    if request_feature_names != bundle_feature_names:
        raise HTTPException(
            status_code=422,
            detail={
                "field": "feature_names",
                "message": "feature order/name mismatch",
                "expected": bundle_feature_names,
                "received": request_feature_names,
            },
        )


def _require_window_shape(
    window: list[list[float]], feature_names: list[str], expected_window_size: int
) -> None:
    if len(window) != expected_window_size:
        raise HTTPException(
            status_code=422,
            detail={
                "field": "window",
                "message": f"expected {expected_window_size} rows (window_size), got {len(window)}",
            },
        )
    if len(window[0]) != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail={
                "field": "window",
                "message": f"expected {len(feature_names)} columns (matching feature_names), "
                f"got {len(window[0])}",
            },
        )


def _alert_severity(predicted_rul: float, horizon_w: float) -> str | None:
    """``None`` if no alert; otherwise "critical" (under half the failure horizon of predicted
    life remains) or "warning" (still within the horizon, but not that close) — a split on the
    same quantity the alert itself is defined on (predicted RUL vs. ``horizon_w``), rather than
    layering a second approximation (e.g. a `failure_probability` cutoff) on top of the
    Gaussian-residual approximation `/predict/rul` already makes. See
    docs/decisions/P10-monitoring.md."""
    if predicted_rul > horizon_w:
        return None
    return "critical" if predicted_rul <= horizon_w / 2 else "warning"


def _failure_probability(predicted_rul: float, horizon_w: float, test_rmse: float) -> float:
    """P(true RUL <= horizon_w) under a Gaussian-residual assumption: the regressor's own
    measured test RMSE (from the bundle's metrics.json — a real, measured number, never
    invented) is used as the residual standard deviation. This is a documented statistical
    approximation, not a calibrated classifier output — see docs/decisions/P08-serving.md for
    its limitations (assumes constant, Gaussian-distributed residual noise, which P04's own
    residual plots showed is not exactly true near the RUL cap).
    """
    if test_rmse <= 0:
        return 1.0 if predicted_rul <= horizon_w else 0.0
    z = (horizon_w - predicted_rul) / test_rmse
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    bundle: RulBundle | None = request.app.state.rul_bundle
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_version=bundle.model_version if bundle else None,
        bundle_id=bundle.bundle_id if bundle else None,
        anomaly_model_loaded=request.app.state.anomaly_bundle is not None,
        uptime_seconds=time.time() - request.app.state.start_time,
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready(request: Request, response: Response) -> ReadyResponse:
    if request.app.state.rul_bundle is None:
        response.status_code = 503
        return ReadyResponse(ready=False, reason="RUL bundle not loaded")
    return ReadyResponse(ready=True)


@app.get("/metrics")
async def metrics() -> Response:
    content, content_type = render_metrics()
    return Response(content=content, media_type=content_type)


@app.post("/predict/rul", response_model=RulPrediction)
async def predict_rul(payload: WindowRequest, request: Request) -> RulPrediction:
    start = time.perf_counter()
    bundle: RulBundle = request.app.state.rul_bundle
    _require_feature_match(payload.feature_names, bundle.feature_spec.feature_names)
    _require_window_shape(payload.window, payload.feature_names, bundle.feature_spec.window_size)

    window = np.array(payload.window, dtype=np.float32)
    scaled = bundle.scale(window)
    predicted_rul = float(bundle.predict_rul(scaled[np.newaxis, ...])[0])

    horizon_w = float(bundle.thresholds.get("failure_horizon_w", settings.data.failure_horizon_w))
    test_rmse = float(bundle.metrics.get("test", {}).get("rmse", 0.0))
    failure_probability = _failure_probability(predicted_rul, horizon_w, test_rmse)
    alert = predicted_rul <= horizon_w
    severity = _alert_severity(predicted_rul, horizon_w)

    latency_ms = (time.perf_counter() - start) * 1000
    PREDICTED_RUL.observe(predicted_rul)
    if severity is not None:
        ALERTS_TOTAL.labels(severity=severity).inc()

    request.state.model_version = bundle.model_version
    request.state.prediction_summary = {"predicted_rul": round(predicted_rul, 3), "alert": alert}

    return RulPrediction(
        predicted_rul=predicted_rul,
        failure_probability=failure_probability,
        alert=alert,
        model_version=bundle.model_version,
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchRulPredictionResponse)
async def predict_batch(
    payload: BatchWindowRequest, request: Request
) -> BatchRulPredictionResponse:
    start = time.perf_counter()
    bundle: RulBundle = request.app.state.rul_bundle

    if len(payload.windows) > settings.serving.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail={
                "field": "windows",
                "message": f"batch size {len(payload.windows)} exceeds configured max "
                f"{settings.serving.max_batch_size}",
            },
        )
    _require_feature_match(payload.feature_names, bundle.feature_spec.feature_names)
    for window in payload.windows:
        _require_window_shape(window, payload.feature_names, bundle.feature_spec.window_size)

    batch = np.array(payload.windows, dtype=np.float32)
    scaled = np.stack([bundle.scale(w) for w in batch])
    predicted = bundle.predict_rul(scaled)

    horizon_w = float(bundle.thresholds.get("failure_horizon_w", settings.data.failure_horizon_w))
    test_rmse = float(bundle.metrics.get("test", {}).get("rmse", 0.0))

    latency_ms = (time.perf_counter() - start) * 1000
    predictions = [
        RulPrediction(
            predicted_rul=float(p),
            failure_probability=_failure_probability(float(p), horizon_w, test_rmse),
            alert=float(p) <= horizon_w,
            model_version=bundle.model_version,
            latency_ms=latency_ms,
        )
        for p in predicted
    ]
    for p in predicted:
        PREDICTED_RUL.observe(float(p))
        severity = _alert_severity(float(p), horizon_w)
        if severity is not None:
            ALERTS_TOTAL.labels(severity=severity).inc()

    request.state.model_version = bundle.model_version
    request.state.prediction_summary = {"batch_size": len(predictions)}

    return BatchRulPredictionResponse(predictions=predictions, latency_ms=latency_ms)


@app.post("/detect/anomaly", response_model=AnomalyDetection)
async def detect_anomaly(payload: WindowRequest, request: Request) -> AnomalyDetection:
    start = time.perf_counter()
    bundle: AnomalyBundle | None = request.app.state.anomaly_bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "anomaly model not available",
                "reason": request.app.state.anomaly_bundle_error,
            },
        )

    _require_feature_match(payload.feature_names, bundle.feature_spec.feature_names)
    _require_window_shape(payload.window, payload.feature_names, bundle.feature_spec.window_size)

    window = np.array(payload.window, dtype=np.float32)
    scaled = bundle.scale(window)
    batch = scaled[np.newaxis, ...]
    score = float(bundle.score(batch)[0])
    per_sensor = bundle.per_sensor_error(batch)[0]
    threshold = bundle.threshold
    is_anomaly = score >= threshold

    latency_ms = (time.perf_counter() - start) * 1000
    ANOMALY_SCORE.observe(score)
    if is_anomaly:
        ANOMALIES_DETECTED.inc()

    request.state.model_version = bundle.model_version
    request.state.prediction_summary = {"anomaly_score": round(score, 4), "is_anomaly": is_anomaly}

    return AnomalyDetection(
        anomaly_score=score,
        threshold=threshold,
        is_anomaly=is_anomaly,
        per_sensor_contribution=dict(
            zip(bundle.feature_spec.feature_names, (float(x) for x in per_sensor), strict=True)
        ),
        model_version=bundle.model_version,
        latency_ms=latency_ms,
    )
