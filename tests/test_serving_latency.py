"""Latency assertion (P08 deliverable #7): p50/p95/p99 over many calls, after warmup — never a
bare average. The budget here is generous on purpose: this measures in-process ASGI call
overhead through Starlette's ``TestClient`` (no real network socket, no real uvicorn worker), so
it is a *ceiling* sanity check ("did something regress by 10x"), not the number that belongs in
the decision log — that number comes from a real running server, measured and reported in
``docs/decisions/P08-serving.md``.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pdm.config import settings
from tests.conftest import TINY_FEATURE_NAMES, TINY_WINDOW_SIZE

pytestmark = pytest.mark.fast

# Generous on purpose — see module docstring. This is a regression ceiling, not a target.
P95_BUDGET_MS = 200.0


def test_predict_rul_p95_latency_is_under_the_documented_budget(tiny_rul_bundle_dir, monkeypatch):
    monkeypatch.setattr(settings.serving, "bundle_dir", str(tiny_rul_bundle_dir))
    monkeypatch.setattr(settings.serving, "anomaly_bundle_dir", None)
    from pdm.serving.app import app

    rng = np.random.default_rng(0)
    window = rng.normal(50, 5, size=(TINY_WINDOW_SIZE, len(TINY_FEATURE_NAMES))).tolist()
    payload = {"feature_names": TINY_FEATURE_NAMES, "window": window}

    with TestClient(app) as client:
        for _ in range(20):  # warmup — never measure the first, cold calls
            client.post("/predict/rul", json=payload)

        timings_ms = []
        for _ in range(200):
            start = time.perf_counter()
            response = client.post("/predict/rul", json=payload)
            timings_ms.append((time.perf_counter() - start) * 1000)
            assert response.status_code == 200

    arr = np.array(timings_ms)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    print(f"predict/rul in-process latency: p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms")
    assert p95 < P95_BUDGET_MS, f"p95 latency {p95:.2f}ms exceeded the {P95_BUDGET_MS}ms budget"
