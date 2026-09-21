"""Contract tests for every serving endpoint: valid input, invalid shape, wrong feature order,
NaN, and empty — plus graceful degradation when the anomaly bundle isn't configured (P08
deliverable #4 and #7).
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pdm.config import settings
from tests.conftest import TINY_FEATURE_NAMES, TINY_WINDOW_SIZE

pytestmark = pytest.mark.fast


def _valid_window(seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return rng.normal(50, 5, size=(TINY_WINDOW_SIZE, len(TINY_FEATURE_NAMES))).tolist()


@pytest.fixture()
def client_with_both_bundles(tiny_rul_bundle_dir, tiny_anomaly_bundle_dir, monkeypatch):
    monkeypatch.setattr(settings.serving, "bundle_dir", str(tiny_rul_bundle_dir))
    monkeypatch.setattr(settings.serving, "anomaly_bundle_dir", str(tiny_anomaly_bundle_dir))
    from pdm.serving.app import app

    with TestClient(app) as client:
        yield client


@pytest.fixture()
def client_without_anomaly_bundle(tiny_rul_bundle_dir, monkeypatch):
    monkeypatch.setattr(settings.serving, "bundle_dir", str(tiny_rul_bundle_dir))
    monkeypatch.setattr(settings.serving, "anomaly_bundle_dir", None)
    from pdm.serving.app import app

    with TestClient(app) as client:
        yield client


# --- /health, /ready -----------------------------------------------------------------------


def test_health_reports_loaded_model_version_and_uptime(client_with_both_bundles):
    r = client_with_both_bundles.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"]
    assert body["bundle_id"]
    assert body["anomaly_model_loaded"] is True
    assert body["uptime_seconds"] >= 0


def test_health_reports_anomaly_model_not_loaded_when_not_configured(client_without_anomaly_bundle):
    r = client_without_anomaly_bundle.get("/health")
    assert r.json()["anomaly_model_loaded"] is False


def test_ready_is_200_once_the_bundle_is_loaded(client_with_both_bundles):
    r = client_with_both_bundles.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


# --- /predict/rul ----------------------------------------------------------------------------


def test_predict_rul_valid_request_returns_a_sensible_prediction(client_with_both_bundles):
    r = client_with_both_bundles.post(
        "/predict/rul",
        json={"feature_names": TINY_FEATURE_NAMES, "window": _valid_window()},
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["predicted_rul"], float)
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert isinstance(body["alert"], bool)
    assert body["model_version"]
    assert body["latency_ms"] >= 0


def test_predict_rul_wrong_feature_order_returns_422_with_useful_message(client_with_both_bundles):
    r = client_with_both_bundles.post(
        "/predict/rul",
        json={"feature_names": list(reversed(TINY_FEATURE_NAMES)), "window": _valid_window()},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["field"] == "feature_names"
    assert detail["expected"] == TINY_FEATURE_NAMES


def test_predict_rul_wrong_window_length_returns_422(client_with_both_bundles):
    r = client_with_both_bundles.post(
        "/predict/rul",
        json={"feature_names": TINY_FEATURE_NAMES, "window": _valid_window()[:-1]},
    )
    assert r.status_code == 422
    assert "window_size" in r.json()["detail"]["message"]


def test_predict_rul_wrong_feature_count_per_row_returns_422(client_with_both_bundles):
    window = [row + [1.0] for row in _valid_window()]  # one extra column per row
    r = client_with_both_bundles.post(
        "/predict/rul", json={"feature_names": TINY_FEATURE_NAMES, "window": window}
    )
    assert r.status_code == 422


def test_predict_rul_nan_in_window_returns_422(client_with_both_bundles):
    window = _valid_window()
    window[0][0] = float("nan")
    # a strict JSON encoder refuses to even serialise NaN (this is itself a real, documented
    # finding — see docs/decisions/P08-serving.md) — send it as raw bytes with a literal NaN
    # token instead, which Python's (and FastAPI's) JSON parser accepts as an extension.
    raw = (
        '{"feature_names": '
        + json.dumps(TINY_FEATURE_NAMES)
        + ', "window": '
        + json.dumps(window)
        + "}"
    )
    r = client_with_both_bundles.post(
        "/predict/rul", content=raw, headers={"content-type": "application/json"}
    )
    assert r.status_code == 422


def test_predict_rul_empty_window_returns_422(client_with_both_bundles):
    r = client_with_both_bundles.post(
        "/predict/rul", json={"feature_names": TINY_FEATURE_NAMES, "window": []}
    )
    assert r.status_code == 422


def test_predict_rul_empty_body_returns_422(client_with_both_bundles):
    r = client_with_both_bundles.post("/predict/rul", json={})
    assert r.status_code == 422


# --- /predict/batch --------------------------------------------------------------------------


def test_predict_batch_valid_request(client_with_both_bundles):
    windows = [_valid_window(0), _valid_window(1)]
    r = client_with_both_bundles.post(
        "/predict/batch", json={"feature_names": TINY_FEATURE_NAMES, "windows": windows}
    )
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_batch_rejects_a_batch_larger_than_configured_max(
    client_with_both_bundles, monkeypatch
):
    monkeypatch.setattr(settings.serving, "max_batch_size", 2)
    windows = [_valid_window(i) for i in range(3)]
    r = client_with_both_bundles.post(
        "/predict/batch", json={"feature_names": TINY_FEATURE_NAMES, "windows": windows}
    )
    assert r.status_code == 422


# --- /detect/anomaly -------------------------------------------------------------------------


def test_detect_anomaly_valid_request_returns_score_and_per_sensor_breakdown(
    client_with_both_bundles,
):
    r = client_with_both_bundles.post(
        "/detect/anomaly",
        json={"feature_names": TINY_FEATURE_NAMES, "window": _valid_window()},
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["anomaly_score"], float)
    assert isinstance(body["is_anomaly"], bool)
    assert set(body["per_sensor_contribution"]) == set(TINY_FEATURE_NAMES)


def test_detect_anomaly_returns_503_when_anomaly_bundle_not_configured(
    client_without_anomaly_bundle,
):
    r = client_without_anomaly_bundle.post(
        "/detect/anomaly",
        json={"feature_names": TINY_FEATURE_NAMES, "window": _valid_window()},
    )
    assert r.status_code == 503
    assert "anomaly model not available" in r.json()["detail"]["message"]


def test_predict_rul_still_works_when_anomaly_bundle_missing(client_without_anomaly_bundle):
    """The core of P08 deliverable #4: a missing anomaly model degrades /detect/anomaly only —
    /predict/rul must be completely unaffected."""
    r = client_without_anomaly_bundle.post(
        "/predict/rul",
        json={"feature_names": TINY_FEATURE_NAMES, "window": _valid_window()},
    )
    assert r.status_code == 200


# --- /metrics --------------------------------------------------------------------------------


def test_metrics_endpoint_is_prometheus_exposition_format(client_with_both_bundles):
    r = client_with_both_bundles.get("/metrics")
    assert r.status_code == 200
    assert "pdm_requests_total" in r.text


def test_metrics_endpoint_includes_the_full_p10_metric_list(client_with_both_bundles):
    """P10 deliverable #1's metric list -- every name must actually be exposed, not just the
    request counter the P08-era test above already checked."""
    client_with_both_bundles.post(
        "/predict/rul", json={"feature_names": TINY_FEATURE_NAMES, "window": _valid_window()}
    )
    client_with_both_bundles.post(
        "/detect/anomaly", json={"feature_names": TINY_FEATURE_NAMES, "window": _valid_window()}
    )
    text = client_with_both_bundles.get("/metrics").text
    for name in (
        "pdm_requests_total",
        "pdm_request_latency_seconds",
        "pdm_predicted_rul",
        "pdm_anomaly_score",
        "pdm_anomalies_detected_total",
        "pdm_alerts_total",
        "pdm_model_info",
        "pdm_last_retrain_timestamp",
        "pdm_model_age_seconds",
        "pdm_process_start_time_seconds",
    ):
        assert name in text, f"{name} missing from /metrics output"


def test_error_responses_are_counted_in_requests_total(client_with_both_bundles):
    """A real bug this project already hit once (P08): a handler that raises before its own
    manual REQUEST_COUNT.inc() silently excluded every error response. Instrumentation now lives
    in RequestLoggingMiddleware specifically so this can't regress."""
    bad = client_with_both_bundles.post(
        "/predict/rul", json={"feature_names": ["wrong"], "window": _valid_window()}
    )
    assert bad.status_code == 422
    text = client_with_both_bundles.get("/metrics").text
    assert 'endpoint="/predict/rul",status="422"' in text


def test_alert_severity_bands():
    from pdm.serving.app import _alert_severity

    assert _alert_severity(100.0, 30.0) is None  # well above the horizon: no alert at all
    assert _alert_severity(30.0, 30.0) == "warning"  # exactly at the horizon
    assert _alert_severity(20.0, 30.0) == "warning"  # inside it, but more than half remains
    assert _alert_severity(15.0, 30.0) == "critical"  # exactly half remains
    assert _alert_severity(1.0, 30.0) == "critical"  # almost none remains


def test_model_info_and_age_metrics_reflect_the_loaded_bundle(client_with_both_bundles):
    health = client_with_both_bundles.get("/health").json()
    text = client_with_both_bundles.get("/metrics").text
    assert f'version="{health["model_version"]}"' in text
    assert f'bundle_id="{health["bundle_id"]}"' in text
    assert "pdm_last_retrain_timestamp " in text or "pdm_last_retrain_timestamp{" in text
    assert "pdm_model_age_seconds " in text or "pdm_model_age_seconds{" in text
