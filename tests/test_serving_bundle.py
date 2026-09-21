"""Bundle loading: a valid bundle loads and matches its own feature_spec, and every way a
bundle can be broken (missing file, corrupt JSON, parameter-count mismatch, missing anomaly
threshold) fails loudly with :class:`BundleValidationError` rather than silently degrading."""

import json

import pytest

from pdm.serving.bundle import BundleValidationError, load_anomaly_bundle, load_rul_bundle

pytestmark = pytest.mark.fast


def test_load_rul_bundle_succeeds_on_a_valid_bundle(tiny_rul_bundle_dir):
    bundle = load_rul_bundle(tiny_rul_bundle_dir)
    assert bundle.feature_spec.feature_names == ["sensor_a", "sensor_b", "sensor_c"]
    assert bundle.thresholds["failure_horizon_w"] == 30
    assert bundle.model_version.startswith("pytorch-seed0-")
    assert bundle.bundle_id == tiny_rul_bundle_dir.name


def test_load_anomaly_bundle_succeeds_on_a_valid_bundle(tiny_anomaly_bundle_dir):
    bundle = load_anomaly_bundle(tiny_anomaly_bundle_dir)
    assert bundle.threshold == 0.5


def test_load_rul_bundle_rejects_a_nonexistent_directory(tmp_path):
    with pytest.raises(BundleValidationError, match="does not exist"):
        load_rul_bundle(tmp_path / "does_not_exist")


@pytest.mark.parametrize(
    "filename",
    ["feature_spec.json", "scaler.json", "thresholds.json", "metrics.json", "metadata.json"],
)
def test_load_rul_bundle_rejects_a_missing_required_file(tiny_rul_bundle_dir, filename):
    (tiny_rul_bundle_dir / filename).unlink()
    with pytest.raises(BundleValidationError, match=filename):
        load_rul_bundle(tiny_rul_bundle_dir)


def test_load_rul_bundle_rejects_corrupt_json(tiny_rul_bundle_dir):
    (tiny_rul_bundle_dir / "metadata.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(BundleValidationError, match="not valid JSON"):
        load_rul_bundle(tiny_rul_bundle_dir)


def test_load_rul_bundle_rejects_a_parameter_count_mismatch(tiny_rul_bundle_dir):
    metadata_path = tiny_rul_bundle_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["parameter_count"] = metadata["parameter_count"] + 1000
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="parameter"):
        load_rul_bundle(tiny_rul_bundle_dir)


def test_load_rul_bundle_rejects_an_unknown_framework(tiny_rul_bundle_dir):
    metadata_path = tiny_rul_bundle_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["framework"] = "some_unknown_framework"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="unknown framework"):
        load_rul_bundle(tiny_rul_bundle_dir)


def test_load_rul_bundle_rejects_a_corrupt_weights_file(tiny_rul_bundle_dir):
    (tiny_rul_bundle_dir / "model_torch.pt").write_bytes(b"not a real checkpoint")
    with pytest.raises(BundleValidationError, match="failed to load model weights"):
        load_rul_bundle(tiny_rul_bundle_dir)


def test_load_anomaly_bundle_rejects_a_missing_anomaly_threshold(tiny_anomaly_bundle_dir):
    thresholds_path = tiny_anomaly_bundle_dir / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    del thresholds["anomaly_score_threshold"]
    thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="anomaly_score_threshold"):
        load_anomaly_bundle(tiny_anomaly_bundle_dir)


def test_rul_bundle_scale_matches_the_scaler_directly(tiny_rul_bundle_dir):
    import numpy as np

    bundle = load_rul_bundle(tiny_rul_bundle_dir)
    window = np.ones((5, 3), dtype=np.float32) * 60
    scaled = bundle.scale(window)
    expected = (window - bundle.scaler.mean_) / bundle.scaler.std_
    np.testing.assert_allclose(scaled, expected)
