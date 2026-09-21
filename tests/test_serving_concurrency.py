"""A concurrency smoke test (P08 deliverable #7): many requests in flight at once must all
succeed with internally-consistent responses — a cheap, real check against shared mutable state
bugs (e.g. accidentally caching something per-request in a module-level global)."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pdm.config import settings
from tests.conftest import TINY_FEATURE_NAMES, TINY_WINDOW_SIZE

pytestmark = pytest.mark.fast


def test_many_concurrent_predict_rul_requests_all_succeed(tiny_rul_bundle_dir, monkeypatch):
    monkeypatch.setattr(settings.serving, "bundle_dir", str(tiny_rul_bundle_dir))
    monkeypatch.setattr(settings.serving, "anomaly_bundle_dir", None)
    from pdm.serving.app import app

    with TestClient(app) as client:

        def call(seed: int):
            rng = np.random.default_rng(seed)
            window = rng.normal(50, 5, size=(TINY_WINDOW_SIZE, len(TINY_FEATURE_NAMES))).tolist()
            return client.post(
                "/predict/rul", json={"feature_names": TINY_FEATURE_NAMES, "window": window}
            )

        with ThreadPoolExecutor(max_workers=16) as pool:
            responses = list(pool.map(call, range(64)))

    assert all(r.status_code == 200 for r in responses)
    bodies = [r.json() for r in responses]
    assert all(isinstance(b["predicted_rul"], float) for b in bodies)
    # every response reports the same loaded model — no request accidentally saw a different
    # (or half-initialised) bundle than the others.
    assert len({b["model_version"] for b in bodies}) == 1
