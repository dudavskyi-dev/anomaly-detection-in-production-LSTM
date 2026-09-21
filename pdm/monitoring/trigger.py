"""The retraining trigger (P09 deliverable #4): decides, from a drift score, whether to launch
a retraining run — never whether to ship it.

**No auto-promotion.** This module never imports ``pdm.tracking.promotion.promote_to_production``
and never transitions anything to the ``Production`` stage — a triggered retrain registers a
new model version and leaves it in ``Staging``. Whether it ever ships is entirely up to the P07
promotion gate (``pdm registry promote``), run by a human, later, against the candidate this
module produced. This split exists specifically because of the trap the P09 spec calls out:
retraining on drifted data and then auto-shipping the result would let exactly the failure mode
that produced the drift signal (a broken sensor, or the real onset of the fault the whole system
exists to catch) get taught back into production as "normal" — see
``docs/decisions/P09-drift.md`` for the full argument. Keeping "detect drift" -> "retrain" ->
"promote" as three separable steps, each independently gated, is the guard.

**The retraining step never trains on the raw drifted traffic either.** ``run_retraining_candidate``
retrains on a fixed, trusted, already-labelled C-MAPSS subset (by default, the subset the
currently-served bundle was itself trained on) using the exact same entry point
(``pdm.cli.train``) a human would use — not on whatever unlabelled/unvalidated production
windows triggered the check. Folding raw drifted data into a retraining set automatically is
exactly the trap above in a different shape; a real fix (curating and relabelling the drifted
traffic before it's trusted as training data) is a human decision this project deliberately does
not attempt to automate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pdm.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TriggerDecision:
    triggered: bool
    reason: str
    aggregate_score: float
    threshold: float
    cooldown_active: bool
    candidate_run_id: str | None = None
    candidate_model_name: str | None = None
    candidate_version: str | None = None


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _cooldown_remaining_seconds(
    state: dict, target: str, now: datetime, cooldown_seconds: int
) -> float:
    last = state.get(target, {}).get("last_trigger_utc")
    if last is None:
        return 0.0
    elapsed = (now - datetime.fromisoformat(last)).total_seconds()
    return max(0.0, cooldown_seconds - elapsed)


def decide_trigger(
    aggregate_score: float,
    *,
    target: str,
    threshold: float | None = None,
    cooldown_seconds: int | None = None,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> TriggerDecision:
    """Pure decision: does this drift score clear the threshold, and is ``target`` still in
    cooldown from a previous trigger? Does **not** retrain or touch the registry — kept separate
    from :func:`run_retraining_candidate` so this can be tested (and reasoned about) without
    MLflow or a real training run, the same way ``pdm.tracking.promotion.evaluate_gate`` is kept
    separate from ``promote_to_production``.

    ``target`` identifies what's being monitored (e.g. ``"<bundle>::<production_subset>:<split>"``)
    — cooldown is tracked independently per target, read from ``state_path`` (a small JSON file,
    since this check runs as a fresh process each time, not a long-lived daemon with in-memory
    state — see ``docs/decisions/P09-drift.md``).
    """
    threshold = settings.monitoring.psi_significant_threshold if threshold is None else threshold
    cooldown_seconds = (
        settings.monitoring.retrain_cooldown_seconds
        if cooldown_seconds is None
        else cooldown_seconds
    )
    state_path = Path(settings.monitoring.state_path) if state_path is None else state_path
    now = now or datetime.now(UTC)

    state = _load_state(state_path)
    remaining = _cooldown_remaining_seconds(state, target, now, cooldown_seconds)

    if aggregate_score < threshold:
        return TriggerDecision(
            triggered=False,
            reason=f"aggregate drift score {aggregate_score:.4f} below threshold {threshold:.4f}",
            aggregate_score=aggregate_score,
            threshold=threshold,
            cooldown_active=remaining > 0,
        )
    if remaining > 0:
        return TriggerDecision(
            triggered=False,
            reason=(
                f"aggregate drift score {aggregate_score:.4f} >= threshold {threshold:.4f}, but "
                f"{target!r} is in cooldown for {remaining:.0f} more seconds (last triggered "
                f"{state[target]['last_trigger_utc']})"
            ),
            aggregate_score=aggregate_score,
            threshold=threshold,
            cooldown_active=True,
        )
    return TriggerDecision(
        triggered=True,
        reason=(
            f"aggregate drift score {aggregate_score:.4f} >= threshold {threshold:.4f}, "
            "cooldown clear"
        ),
        aggregate_score=aggregate_score,
        threshold=threshold,
        cooldown_active=False,
    )


def run_retraining_candidate(subset: str, *, trigger_count: int = 0) -> tuple[str, str, str]:
    """Launches a fresh training run on ``subset`` and registers it as a **candidate** model
    version in stage ``Staging`` — never ``Production``. Uses ``pdm.cli.train`` directly, so a
    candidate produced this way is indistinguishable from one a human produced with
    ``pdm train --subset ...``.

    The candidate seed is offset from ``settings.seed`` by
    ``settings.monitoring.retrain_seed_offset * (trigger_count + 1)`` so its bundle directory
    (``artifacts/torch_rul_regressor__{subset}__seed{seed}``) never collides with — and can
    therefore never silently overwrite — the currently-served bundle's own directory.
    ``trigger_count`` (how many times this target has already triggered, from the cooldown
    state) keeps repeated triggers from colliding with each other too.

    Returns ``(run_id, model_name, version)``.
    """
    from pdm.cli import train as run_rul_training
    from pdm.tracking.mlflow_client import configure_tracking
    from pdm.tracking.registry import (
        RUL_MODEL_NAME,
        register_model_version,
        transition_to_staging,
    )

    candidate_seed = settings.seed + settings.monitoring.retrain_seed_offset * (trigger_count + 1)
    run_id = run_rul_training(subset=subset, seed=candidate_seed)

    configure_tracking()
    version = register_model_version(run_id, RUL_MODEL_NAME, artifact_path="bundle")
    transition_to_staging(RUL_MODEL_NAME, version)
    return run_id, RUL_MODEL_NAME, version


def check_and_maybe_retrain(
    aggregate_score: float,
    *,
    target: str,
    retrain_subset: str,
    retrain: bool = True,
    threshold: float | None = None,
    cooldown_seconds: int | None = None,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> TriggerDecision:
    """Ties :func:`decide_trigger` and :func:`run_retraining_candidate` together: the CLI's
    ``pdm drift check`` entry point. ``retrain=False`` (the ``--no-retrain`` flag) evaluates and
    logs the decision exactly as normal but never launches training or touches the registry —
    and, since nothing was actually retrained, never updates the cooldown state either.
    """
    state_path = Path(settings.monitoring.state_path) if state_path is None else state_path
    now = now or datetime.now(UTC)

    decision = decide_trigger(
        aggregate_score,
        target=target,
        threshold=threshold,
        cooldown_seconds=cooldown_seconds,
        state_path=state_path,
        now=now,
    )
    logger.info("drift trigger decision for %s: %s", target, decision.reason)

    if decision.triggered and retrain:
        state = _load_state(state_path)
        trigger_count = state.get(target, {}).get("trigger_count", 0)
        run_id, model_name, version = run_retraining_candidate(
            retrain_subset, trigger_count=trigger_count
        )
        decision.candidate_run_id = run_id
        decision.candidate_model_name = model_name
        decision.candidate_version = version

        state[target] = {
            "last_trigger_utc": now.isoformat(),
            "last_run_id": run_id,
            "trigger_count": trigger_count + 1,
        }
        _save_state(state_path, state)

    return decision
