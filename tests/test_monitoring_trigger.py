"""The retraining trigger's decision logic (P09 deliverable #6): fires once above threshold,
respects cooldown, and never launches a retraining run when told not to. No MLflow or real
training involved — ``run_retraining_candidate`` is monkeypatched out, the same way
``tests/test_promotion.py`` keeps the promotion gate's pure logic separate from a real registry.
"""

from datetime import UTC, datetime, timedelta

import pytest

from pdm.monitoring import trigger as trigger_module
from pdm.monitoring.trigger import TriggerDecision, check_and_maybe_retrain, decide_trigger

pytestmark = pytest.mark.fast


def test_below_threshold_never_triggers(tmp_path):
    decision = decide_trigger(
        0.05, target="t", threshold=0.25, cooldown_seconds=3600, state_path=tmp_path / "state.json"
    )
    assert not decision.triggered
    assert "below threshold" in decision.reason


def test_above_threshold_triggers_when_no_prior_trigger(tmp_path):
    decision = decide_trigger(
        0.5, target="t", threshold=0.25, cooldown_seconds=3600, state_path=tmp_path / "state.json"
    )
    assert decision.triggered
    assert not decision.cooldown_active


def test_cooldown_blocks_a_repeat_trigger(tmp_path):
    state_path = tmp_path / "state.json"
    now = datetime(2026, 1, 1, tzinfo=UTC)

    first = decide_trigger(
        0.5, target="t", threshold=0.25, cooldown_seconds=3600, state_path=state_path, now=now
    )
    assert first.triggered
    trigger_module._save_state(
        state_path, {"t": {"last_trigger_utc": now.isoformat(), "trigger_count": 1}}
    )

    soon_after = now + timedelta(seconds=60)
    second = decide_trigger(
        0.5,
        target="t",
        threshold=0.25,
        cooldown_seconds=3600,
        state_path=state_path,
        now=soon_after,
    )
    assert not second.triggered
    assert second.cooldown_active
    assert "cooldown" in second.reason


def test_cooldown_expires_after_the_configured_window(tmp_path):
    state_path = tmp_path / "state.json"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    trigger_module._save_state(
        state_path, {"t": {"last_trigger_utc": now.isoformat(), "trigger_count": 1}}
    )

    long_after = now + timedelta(seconds=7200)
    decision = decide_trigger(
        0.5,
        target="t",
        threshold=0.25,
        cooldown_seconds=3600,
        state_path=state_path,
        now=long_after,
    )
    assert decision.triggered
    assert not decision.cooldown_active


def test_cooldown_is_tracked_independently_per_target(tmp_path):
    state_path = tmp_path / "state.json"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    trigger_module._save_state(
        state_path, {"fd002": {"last_trigger_utc": now.isoformat(), "trigger_count": 1}}
    )

    decision = decide_trigger(
        0.5, target="fd003", threshold=0.25, cooldown_seconds=3600, state_path=state_path, now=now
    )
    assert decision.triggered
    assert not decision.cooldown_active


def test_no_retrain_flag_skips_retraining_and_never_touches_cooldown_state(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    called = []
    monkeypatch.setattr(
        trigger_module, "run_retraining_candidate", lambda *a, **k: called.append((a, k))
    )

    decision = check_and_maybe_retrain(
        0.5,
        target="t",
        retrain_subset="FD001",
        retrain=False,
        threshold=0.25,
        cooldown_seconds=3600,
        state_path=state_path,
    )
    assert decision.triggered
    assert decision.candidate_run_id is None
    assert not called
    assert not state_path.exists()  # nothing was actually retrained, so no cooldown was recorded


def test_check_and_maybe_retrain_registers_a_candidate_and_records_cooldown(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(
        trigger_module,
        "run_retraining_candidate",
        lambda subset, *, trigger_count=0: ("run-123", "pdm-sentinel-rul", "7"),
    )

    decision = check_and_maybe_retrain(
        0.5,
        target="t",
        retrain_subset="FD001",
        retrain=True,
        threshold=0.25,
        cooldown_seconds=3600,
        state_path=state_path,
    )
    assert decision.triggered
    assert decision.candidate_run_id == "run-123"
    assert decision.candidate_model_name == "pdm-sentinel-rul"
    assert decision.candidate_version == "7"

    state = trigger_module._load_state(state_path)
    assert state["t"]["last_run_id"] == "run-123"
    assert state["t"]["trigger_count"] == 1

    # a second call before cooldown expires must not retrain again
    second = check_and_maybe_retrain(
        0.5,
        target="t",
        retrain_subset="FD001",
        retrain=True,
        threshold=0.25,
        cooldown_seconds=3600,
        state_path=state_path,
    )
    assert not second.triggered
    assert second.candidate_run_id is None


def test_decision_is_a_dataclass_with_the_expected_fields():
    decision = TriggerDecision(
        triggered=True, reason="x", aggregate_score=0.5, threshold=0.25, cooldown_active=False
    )
    assert decision.candidate_run_id is None
