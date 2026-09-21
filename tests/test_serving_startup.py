"""A service that boots with a broken model and serves garbage is worse than one that won't
boot — this checks the second half of that sentence: a corrupt/incomplete bundle must prevent
the app from starting at all, not just fail individual requests afterward."""

import pytest
from fastapi.testclient import TestClient

from pdm.config import settings
from pdm.serving.bundle import BundleValidationError

pytestmark = pytest.mark.fast


def test_app_refuses_to_start_with_a_missing_bundle_file(tiny_rul_bundle_dir, monkeypatch):
    (tiny_rul_bundle_dir / "scaler.json").unlink()
    monkeypatch.setattr(settings.serving, "bundle_dir", str(tiny_rul_bundle_dir))
    monkeypatch.setattr(settings.serving, "anomaly_bundle_dir", None)

    from pdm.serving.app import app

    with pytest.raises(BundleValidationError), TestClient(app):
        pass


def test_app_refuses_to_start_with_a_corrupt_weights_file(tiny_rul_bundle_dir, monkeypatch):
    (tiny_rul_bundle_dir / "model_torch.pt").write_bytes(b"not a real checkpoint")
    monkeypatch.setattr(settings.serving, "bundle_dir", str(tiny_rul_bundle_dir))
    monkeypatch.setattr(settings.serving, "anomaly_bundle_dir", None)

    from pdm.serving.app import app

    with pytest.raises(BundleValidationError), TestClient(app):
        pass


def test_app_refuses_to_start_pointed_at_a_nonexistent_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.serving, "bundle_dir", str(tmp_path / "nope"))
    monkeypatch.setattr(settings.serving, "anomaly_bundle_dir", None)

    from pdm.serving.app import app

    with pytest.raises(BundleValidationError), TestClient(app):
        pass


def test_app_starts_fine_when_only_the_optional_anomaly_bundle_is_broken(
    tiny_rul_bundle_dir, tiny_anomaly_bundle_dir, monkeypatch
):
    """A broken *anomaly* bundle must never take down the whole service — only /detect/anomaly
    degrades. This is the one bundle problem that must NOT raise at startup."""
    (tiny_anomaly_bundle_dir / "scaler.json").unlink()
    monkeypatch.setattr(settings.serving, "bundle_dir", str(tiny_rul_bundle_dir))
    monkeypatch.setattr(settings.serving, "anomaly_bundle_dir", str(tiny_anomaly_bundle_dir))

    from pdm.serving.app import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["anomaly_model_loaded"] is False
