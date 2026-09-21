"""Model registry helpers (P07 deliverable #3/#4): registration, every stage transition
*except* ``Production`` (that one only ever happens through
``pdm.tracking.promotion.promote_to_production`` — see that module's docstring for why), and the
run-comparison table behind ``pdm registry compare``.
"""

from dataclasses import dataclass, field

import mlflow
from mlflow.tracking import MlflowClient

RUL_MODEL_NAME = "pdm-sentinel-rul"
ANOMALY_MODEL_NAME = "pdm-sentinel-anomaly"


def register_model_version(run_id: str, model_name: str, *, artifact_path: str = "model") -> str:
    """Registers the model artifact logged at ``artifact_path`` under ``run_id`` as a new
    version of ``model_name`` — creating the registered model first if this is its first
    version. New versions start in stage ``None``; promote explicitly from there."""
    mv = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)
    return mv.version


def transition_to_staging(
    model_name: str, version: str, *, client: MlflowClient | None = None
) -> None:
    client = client or MlflowClient()
    client.transition_model_version_stage(model_name, version, stage="Staging")


def archive(model_name: str, version: str, *, client: MlflowClient | None = None) -> None:
    client = client or MlflowClient()
    client.transition_model_version_stage(model_name, version, stage="Archived")


@dataclass
class RunSummary:
    run_id: str
    run_name: str | None
    metric_value: float | None
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


def compare_top_runs(
    experiment_name: str,
    primary_metric: str,
    *,
    n: int = 10,
    higher_is_better: bool = True,
    client: MlflowClient | None = None,
) -> list[RunSummary]:
    """The top ``n`` runs in ``experiment_name`` ranked by ``primary_metric``, each with its key
    params and tags — ``pdm registry compare``'s data source, and the answer to "how did you
    decide which model to ship" as an actual table rather than an assertion.
    """
    client = client or MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    order = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metrics.{primary_metric} {order}"],
        max_results=n,
    )
    return [
        RunSummary(
            run_id=r.info.run_id,
            run_name=r.info.run_name,
            metric_value=r.data.metrics.get(primary_metric),
            params=dict(r.data.params),
            tags={k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
        )
        for r in runs
        if primary_metric in r.data.metrics
    ]
