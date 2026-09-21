"""The model registry and promotion gate against a real (temporary, local file-store) MLflow
tracking server — not just the pure gate logic (that's ``tests/test_promotion.py``), but the
full register -> stage -> promote -> archive lifecycle, and the guarantee that a refused
promotion leaves the registry untouched.
"""

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyRegressor

from pdm.config import settings
from pdm.tracking.mlflow_client import log_metrics, start_run
from pdm.tracking.promotion import PromotionRefused, promote_to_production
from pdm.tracking.registry import (
    archive,
    compare_top_runs,
    register_model_version,
    transition_to_staging,
)

pytestmark = pytest.mark.fast


@pytest.fixture()
def local_tracking(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.tracking, "uri", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr(settings.tracking, "experiment_name", "test-registry")
    yield tmp_path


def _log_candidate(run_name: str, rmse: float, seed: int = 0) -> str:
    with start_run(run_name=run_name, framework="sklearn", seed=seed) as run:
        log_metrics({"rmse": rmse})
        mlflow.sklearn.log_model(DummyRegressor(), "model")
        return run.info.run_id


def test_register_creates_a_new_version_in_stage_none(local_tracking):
    run_id = _log_candidate("run-1", 10.0)
    version = register_model_version(run_id, "model-a")

    client = MlflowClient()
    mv = client.get_model_version("model-a", version)
    assert mv.current_stage == "None"


def test_transition_to_staging_moves_the_version(local_tracking):
    run_id = _log_candidate("run-1", 10.0)
    version = register_model_version(run_id, "model-b")
    transition_to_staging("model-b", version)

    client = MlflowClient()
    mv = client.get_model_version("model-b", version)
    assert mv.current_stage == "Staging"


def test_first_promotion_succeeds_with_no_incumbent(local_tracking):
    run_id = _log_candidate("run-1", 10.0)
    version = register_model_version(run_id, "model-c")
    transition_to_staging("model-c", version)

    decision = promote_to_production("model-c", version, primary_metric="rmse", margin=0.02)
    assert decision.promoted

    client = MlflowClient()
    prod = client.get_latest_versions("model-c", stages=["Production"])
    assert len(prod) == 1 and prod[0].version == version


def test_promotion_refused_leaves_the_registry_untouched(local_tracking):
    run1 = _log_candidate("run-1", 10.0, seed=0)
    v1 = register_model_version(run1, "model-d")
    transition_to_staging("model-d", v1)
    promote_to_production("model-d", v1, primary_metric="rmse", margin=0.02)

    # only 0.5% better than incumbent; default-style 2% margin refuses it
    run2 = _log_candidate("run-2", 9.95, seed=1)
    v2 = register_model_version(run2, "model-d")
    transition_to_staging("model-d", v2)

    with pytest.raises(PromotionRefused):
        promote_to_production("model-d", v2, primary_metric="rmse", margin=0.02)

    client = MlflowClient()
    prod = client.get_latest_versions("model-d", stages=["Production"])
    assert len(prod) == 1 and prod[0].version == v1, "a refused promotion must not touch Production"


def test_a_comfortably_better_candidate_is_promoted_and_the_old_one_archived(local_tracking):
    run1 = _log_candidate("run-1", 10.0, seed=0)
    v1 = register_model_version(run1, "model-e")
    transition_to_staging("model-e", v1)
    promote_to_production("model-e", v1, primary_metric="rmse", margin=0.02)

    run2 = _log_candidate("run-2", 8.0, seed=1)  # 20% better, comfortably clears the margin
    v2 = register_model_version(run2, "model-e")
    transition_to_staging("model-e", v2)
    decision = promote_to_production("model-e", v2, primary_metric="rmse", margin=0.02)
    assert decision.promoted

    client = MlflowClient()
    prod = client.get_latest_versions("model-e", stages=["Production"])
    archived = client.get_latest_versions("model-e", stages=["Archived"])
    assert prod[0].version == v2
    assert any(v.version == v1 for v in archived)


def test_no_function_other_than_promote_to_production_sets_the_production_stage():
    """A structural guard for "make it impossible to promote by calling a single unguarded
    function": nothing in ``pdm.tracking.registry`` should ever pass ``stage="Production"``."""
    import inspect

    from pdm.tracking import registry

    source = inspect.getsource(registry)
    assert 'stage="Production"' not in source
    assert "stage='Production'" not in source


def test_compare_top_runs_ranks_by_the_primary_metric_correctly(local_tracking):
    _log_candidate("best", 5.0)
    _log_candidate("middle", 8.0)
    _log_candidate("worst", 12.0)

    ranked = compare_top_runs("test-registry", "rmse", n=10, higher_is_better=False)
    assert [r.run_name for r in ranked] == ["best", "middle", "worst"]


def test_archive_moves_a_version_out_of_its_current_stage(local_tracking):
    run_id = _log_candidate("run-1", 10.0)
    version = register_model_version(run_id, "model-f")
    transition_to_staging("model-f", version)
    archive("model-f", version)

    client = MlflowClient()
    mv = client.get_model_version("model-f", version)
    assert mv.current_stage == "Archived"
