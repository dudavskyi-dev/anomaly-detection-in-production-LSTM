"""The MLflow wrapper: param flattening/secret-scrubbing (pure, no tracking server needed), and
a real run logged against a temporary local file store (the same backend `PDM__TRACKING__URI`
defaults to) to check the full start/log/tag contract end to end."""

import mlflow
import pytest

from pdm.config import settings
from pdm.tracking.mlflow_client import (
    flatten_params,
    log_artifact,
    log_epoch_history,
    log_metrics,
    log_params,
    sha256_of_files,
    start_run,
)

pytestmark = pytest.mark.fast


@pytest.fixture()
def local_tracking(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.tracking, "uri", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr(settings.tracking, "experiment_name", "test-experiment")
    yield tmp_path


def test_flatten_params_dots_nested_keys():
    flat = flatten_params({"a": {"b": {"c": 1}}, "d": 2})
    assert flat == {"a.b.c": 1, "d": 2}


def test_flatten_params_stringifies_tuples_and_lists():
    flat = flatten_params({"seeds": (0, 1, 2)})
    assert flat == {"seeds": "(0, 1, 2)"}


def test_flatten_params_drops_secret_looking_keys():
    flat = flatten_params({"db": {"password": "hunter2", "host": "localhost"}})
    assert "db.password" not in flat
    assert flat == {"db.host": "localhost"}


@pytest.mark.parametrize("marker", ["password", "secret", "token", "api_key", "apikey"])
def test_flatten_params_drops_every_known_secret_marker(marker):
    flat = flatten_params({f"my_{marker}": "value", "safe": "value"})
    assert flat == {"safe": "value"}


def test_flatten_params_scrubs_credentials_from_uri_shaped_values():
    flat = flatten_params({"uri": "postgresql://user:pw@db.internal:5432/mlflow"})
    assert flat == {"uri": "postgresql://db.internal:5432/mlflow"}


def test_flatten_params_leaves_uris_without_credentials_untouched():
    flat = flatten_params({"uri": "http://localhost:5000"})
    assert flat == {"uri": "http://localhost:5000"}


def test_sha256_of_files_is_order_independent(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello")
    b.write_text("world")
    assert sha256_of_files([a, b]) == sha256_of_files([b, a])


def test_sha256_of_files_changes_when_content_changes(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("v1")
    first = sha256_of_files([f])
    f.write_text("v2")
    second = sha256_of_files([f])
    assert first != second


def test_start_run_logs_params_metrics_and_tags(local_tracking):
    with start_run(run_name="unit-test-run", framework="pytorch", seed=7) as run:
        log_params({"model": {"hidden": 100}})
        log_metrics({"rmse": 12.5})
        run_id = run.info.run_id

    fetched = mlflow.get_run(run_id)
    assert fetched.data.params["model.hidden"] == "100"
    assert fetched.data.metrics["rmse"] == 12.5
    assert fetched.data.tags["framework"] == "pytorch"
    assert fetched.data.tags["seed"] == "7"
    assert "git_sha" in fetched.data.tags
    assert "git_dirty" in fetched.data.tags


def test_start_run_includes_dataset_hash_and_extra_tags_when_given(local_tracking):
    with start_run(
        run_name="unit-test-run-2",
        framework="tensorflow",
        seed=0,
        dataset_hash="deadbeef",
        extra_tags={"subset": "FD001"},
    ) as run:
        run_id = run.info.run_id

    fetched = mlflow.get_run(run_id)
    assert fetched.data.tags["dataset_hash"] == "deadbeef"
    assert fetched.data.tags["subset"] == "FD001"


def test_start_run_ends_the_run_even_if_the_body_raises(local_tracking):
    with pytest.raises(ValueError):
        with start_run(run_name="failing-run", framework="pytorch", seed=0):
            raise ValueError("boom")
    assert mlflow.active_run() is None


class _FakeEpoch:
    def __init__(self, epoch, train_loss, val_loss):
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss

    def to_dict(self):
        return {"epoch": self.epoch, "train_loss": self.train_loss, "val_loss": self.val_loss}


def test_log_epoch_history_logs_each_epoch_as_a_step_metric(local_tracking):
    history = [_FakeEpoch(1, 1.0, 2.0), _FakeEpoch(2, 0.5, 1.5)]
    with start_run(run_name="history-run", framework="pytorch", seed=0) as run:
        log_epoch_history(history)
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    train_loss_history = client.get_metric_history(run_id, "train_loss")
    steps = sorted(m.step for m in train_loss_history)
    assert steps == [1, 2]


def test_log_artifact_attaches_a_file_to_the_run(local_tracking, tmp_path):
    artifact = tmp_path / "note.txt"
    artifact.write_text("hello")
    with start_run(run_name="artifact-run", framework="pytorch", seed=0) as run:
        log_artifact(artifact)
        run_id = run.info.run_id

    files = mlflow.artifacts.list_artifacts(run_id=run_id)
    assert any(f.path == "note.txt" for f in files)
