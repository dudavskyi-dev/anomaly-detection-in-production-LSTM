"""Prometheus exposition for the serving API (P08 deliverable #1's ``GET /metrics``; P10
deliverable #1 fleshes it out into the full metric list Prometheus/Grafana/alerting need).

A dedicated ``CollectorRegistry`` (not the global default one) so re-creating the FastAPI app —
which every test that spins up a fresh ``TestClient`` does — never hits a "duplicated
timeseries" registration error from Prometheus's client library.

**Drift metrics (``pdm_drift_score``/``pdm_drift_score_aggregate``) are deliberately not defined
here.** They're computed by ``pdm.monitoring`` (the P09 drift-check job), a periodic batch
process, not the live API — defining them in this registry would mean either the API recomputes
drift itself (which would require importing training/analysis code the P08 import-graph test
explicitly forbids: ``tests/test_serving_import_graph.py``) or silently going stale between
checks with no honest way to say so. They're emitted directly by
``pdm.monitoring.metrics.write_drift_metrics`` in the same Prometheus textfile format
node_exporter's textfile collector reads, and Prometheus scrapes that as its own target
(``docker/prometheus.yml``) — see ``docs/decisions/P10-monitoring.md`` for the full reasoning.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "pdm_requests_total",
    "Total requests by endpoint and status code",
    ["endpoint", "status"],
    registry=REGISTRY,
)

# Buckets chosen from docs/decisions/P08-serving.md's real, containerized (not in-process
# TestClient) latency measurements: p50 ~14.5ms, p95 ~16-20ms, p99 ~17.5-27.8ms, max ~28ms
# across /predict/rul, /detect/anomaly, and a 16-window /predict/batch. Every observation this
# project has ever actually measured falls under 30ms, so resolution is concentrated there
# (2.5-5ms steps); a handful of coarser buckets extend to 1s purely to still classify a cold
# start, a GC pause, or a genuine regression, without spending resolution on a range nothing
# real has ever produced.
REQUEST_LATENCY_SECONDS = Histogram(
    "pdm_request_latency_seconds",
    "Request latency in seconds, measured end-to-end by RequestLoggingMiddleware (parsing, "
    "validation, and serialization included, not just model inference time)",
    ["endpoint"],
    buckets=(
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.0125,
        0.015,
        0.02,
        0.025,
        0.03,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
    registry=REGISTRY,
)

PREDICTED_RUL = Histogram(
    "pdm_predicted_rul",
    "Distribution of predicted remaining useful life",
    registry=REGISTRY,
    buckets=(0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300),
)

# Buckets straddle the shipped AE bundle's own anomaly_score_threshold (0.8522, see
# artifacts/torch_lstm_ae__FD001__seed42/thresholds.json) at fine resolution -- what a dashboard
# actually needs is "how close is normal traffic sitting to the alarm line", which needs several
# buckets either side of 0.85, not evenly spaced buckets across whatever the max ever seen is.
# Coarse buckets beyond 1.0 still classify a genuinely drifted score (P09's FD002 scenario
# measured scores orders of magnitude past threshold) without wasting resolution there.
ANOMALY_SCORE = Histogram(
    "pdm_anomaly_score",
    "Distribution of anomaly reconstruction-error scores",
    registry=REGISTRY,
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8522, 0.9, 1.0, 1.5, 2.0, 5.0),
)

ANOMALIES_DETECTED = Counter(
    "pdm_anomalies_detected_total",
    "Total windows /detect/anomaly classified as anomalous (score >= threshold)",
    registry=REGISTRY,
)

# severity bands: "critical" is under half the failure horizon W (settings.data.failure_horizon_w)
# of predicted RUL remaining -- a simple, explainable split on the same quantity the alert itself
# is defined on, rather than layering a second approximation (e.g. a failure_probability cutoff)
# on top of the Gaussian-residual approximation /predict/rul already makes. See
# docs/decisions/P10-monitoring.md for why this split, not some other number.
ALERTS_TOTAL = Counter(
    "pdm_alerts_total",
    "Total RUL alerts raised (predicted_rul <= failure_horizon_w), by severity",
    ["severity"],
    registry=REGISTRY,
)

MODEL_INFO = Info(
    "pdm_model",
    "Currently loaded RUL model bundle (version, framework, bundle_id)",
    registry=REGISTRY,
)

# The default ``CollectorRegistry`` prometheus_client normally auto-populates includes a
# ``process_start_time_seconds`` gauge for free; the dedicated per-app ``REGISTRY`` this module
# uses instead (see module docstring) does not. Set once at startup so Grafana's "uptime" panel
# (P10 deliverable #4) can compute ``time() - pdm_process_start_time_seconds`` itself, the same
# way it would for any other Prometheus client library's process metrics.
PROCESS_START_TIME_SECONDS = Gauge(
    "pdm_process_start_time_seconds",
    "Unix timestamp this serving process started",
    registry=REGISTRY,
)

LAST_RETRAIN_TIMESTAMP = Gauge(
    "pdm_last_retrain_timestamp",
    "Unix timestamp the currently-served model bundle was trained (metadata.json's "
    "timestamp_utc) -- not when it was promoted or deployed, which this service has no way to "
    "know from the bundle alone",
    ["version"],
    registry=REGISTRY,
)

MODEL_AGE_SECONDS = Gauge(
    "pdm_model_age_seconds",
    "Seconds since the currently-served model bundle was trained",
    ["version"],
    registry=REGISTRY,
)


def set_model_info(
    *, version: str, framework: str, bundle_id: str, trained_at_iso: str | None
) -> None:
    """Called once at startup with the loaded ``RulBundle``'s own identity/provenance fields.
    ``MODEL_AGE_SECONDS`` is set from a live function (not a one-shot ``.set()``) so it keeps
    ticking upward correctly across the service's whole uptime without needing a background
    refresher thread. Both gauges carry a ``version`` label (redundant with ``pdm_model_info``,
    which has no numeric value to alert on) purely so `PdmModelStale`
    (``docker/prometheus/alerts.yml``) can name *which* model is stale in its annotation without
    a cross-metric join.
    """
    MODEL_INFO.info({"version": version, "framework": framework, "bundle_id": bundle_id})
    if trained_at_iso:
        trained_at = datetime.fromisoformat(trained_at_iso).timestamp()
        LAST_RETRAIN_TIMESTAMP.labels(version=version).set(trained_at)
        MODEL_AGE_SECONDS.labels(version=version).set_function(lambda: time.time() - trained_at)
    else:
        # A bundle written before metadata.json recorded a timestamp (pre-P04-decision-log
        # bundles, if any survive) -- 0 is a visibly-wrong sentinel a dashboard/alert would
        # immediately flag as stale, which is more honest than silently omitting the series.
        LAST_RETRAIN_TIMESTAMP.labels(version=version).set(0)
        MODEL_AGE_SECONDS.labels(version=version).set_function(
            lambda: time.time() - datetime.now(UTC).timestamp()
        )


def record_request(endpoint: str, status: str, latency_seconds: float) -> None:
    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint).observe(latency_seconds)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
