"""The promotion gate (P07 deliverable #3): a candidate model version may reach the
``Production`` stage **only** by passing through :func:`promote_to_production`. There is no
other function anywhere in this codebase that transitions a model version to ``Production`` —
``pdm.tracking.registry`` exposes stage transitions for every other stage
(``None → Staging``, ``→ Archived``), deliberately not for ``Production``, so promoting a model
without going through the gate is not a matter of remembering to call the right function; the
unguarded function to skip the gate with does not exist.
"""

from dataclasses import dataclass

from mlflow.tracking import MlflowClient

from pdm.config import settings

# Metrics this project reports where a *lower* value is better. Anything not listed here is
# assumed higher-is-better (PR-AUC, ROC-AUC, F1, precision, recall) — the more common case among
# this project's own metrics (see pdm.evaluation.metrics).
_LOWER_IS_BETTER = {"rmse", "mae", "nasa_score", "brier"}

# Every logged metric in this project is prefixed by which split produced it (``val_rmse``,
# ``test_f1``, ``ae_alone_precision``, ...) — strip a leading split/detector qualifier down to
# the bare metric name before checking direction, rather than requiring every call site to pass
# the unprefixed name.
_SPLIT_PREFIXES = ("val_", "test_", "train_")


def is_higher_better(metric_name: str) -> bool:
    bare = metric_name
    for prefix in _SPLIT_PREFIXES:
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    return bare not in _LOWER_IS_BETTER


class PromotionRefused(RuntimeError):
    """Raised by :func:`promote_to_production` when the gate fails. Catch this specifically
    rather than a bare exception if you need to distinguish "gate said no" from a genuine error
    (a missing run, an unreachable tracking server)."""


@dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    candidate_value: float
    production_value: float | None
    primary_metric: str
    margin: float


def evaluate_gate(
    candidate_value: float,
    production_value: float | None,
    *,
    primary_metric: str,
    margin: float,
) -> PromotionDecision:
    """Pure decision logic, no MLflow involved — the part
    ``tests/test_promotion.py`` exercises directly, and the part
    :func:`promote_to_production` defers to rather than duplicating.

    No existing Production model (first-ever registration) always passes: there is nothing to
    beat, and refusing a first promotion because no incumbent exists would make the registry
    permanently empty.
    """
    higher_is_better = is_higher_better(primary_metric)
    if production_value is None:
        return PromotionDecision(
            True,
            "no existing Production model to beat",
            candidate_value,
            None,
            primary_metric,
            margin,
        )

    # A tiny relative tolerance absorbs floating-point noise at the boundary (e.g.
    # `0.80 * 1.02 == 0.8160000000000001`, not the mathematically exact `0.816`) — without it, a
    # candidate whose metric lands exactly on the required threshold can be wrongly refused
    # depending on summation order, which is not the kind of thing a promotion decision should
    # ever hinge on.
    if higher_is_better:
        required = production_value * (1 + margin)
        passed = candidate_value >= required * (1 - 1e-9)
        comparison = ">="
    else:
        required = production_value * (1 - margin)
        passed = candidate_value <= required * (1 + 1e-9)
        comparison = "<="

    reason = (
        f"candidate {primary_metric}={candidate_value:.6g} {comparison} required "
        f"{required:.6g} (production {primary_metric}={production_value:.6g}, "
        f"margin={margin:.1%}, higher_is_better={higher_is_better}) -> "
        f"{'PASSED' if passed else 'REFUSED'}"
    )
    return PromotionDecision(
        passed, reason, candidate_value, production_value, primary_metric, margin
    )


def promote_to_production(
    model_name: str,
    candidate_version: str,
    *,
    primary_metric: str | None = None,
    margin: float | None = None,
    client: MlflowClient | None = None,
) -> PromotionDecision:
    """The only sanctioned path to ``Production``. Compares the candidate version's logged
    ``primary_metric`` against the current Production version's (if any) by at least ``margin``
    (relative), and only transitions the registry if the gate passes — raising
    :class:`PromotionRefused` (registry left untouched) otherwise.

    On success, any existing Production version(s) are moved to ``Archived`` first — the
    registry never holds two ``Production`` versions of the same model at once.
    """
    primary_metric = (
        settings.tracking.promotion_primary_metric if primary_metric is None else primary_metric
    )
    margin = settings.tracking.promotion_margin if margin is None else margin
    client = client or MlflowClient()

    candidate_mv = client.get_model_version(model_name, candidate_version)
    candidate_run = client.get_run(candidate_mv.run_id)
    candidate_value = candidate_run.data.metrics.get(primary_metric)
    if candidate_value is None:
        raise PromotionRefused(
            f"candidate run {candidate_mv.run_id!r} (version {candidate_version} of "
            f"{model_name!r}) has no logged metric {primary_metric!r} — cannot evaluate the gate."
        )

    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    production_value = None
    if production_versions:
        production_run = client.get_run(production_versions[0].run_id)
        production_value = production_run.data.metrics.get(primary_metric)

    decision = evaluate_gate(
        candidate_value, production_value, primary_metric=primary_metric, margin=margin
    )
    if not decision.promoted:
        raise PromotionRefused(decision.reason)

    for old in production_versions:
        client.transition_model_version_stage(model_name, old.version, stage="Archived")
    client.transition_model_version_stage(model_name, candidate_version, stage="Production")
    return decision
