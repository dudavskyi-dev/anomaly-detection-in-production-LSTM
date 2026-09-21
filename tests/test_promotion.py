"""The promotion gate's pure decision logic — no MLflow server involved, so these run fast and
pin down the actual comparison arithmetic (margin direction for higher-is-better vs
lower-is-better metrics) that ``promote_to_production`` defers to.
"""

import pytest

from pdm.tracking.promotion import PromotionDecision, evaluate_gate, is_higher_better

pytestmark = pytest.mark.fast


def test_is_higher_better_classifies_known_metrics_correctly():
    assert not is_higher_better("rmse")
    assert not is_higher_better("mae")
    assert not is_higher_better("nasa_score")
    assert is_higher_better("pr_auc")
    assert is_higher_better("f1")
    assert is_higher_better("roc_auc")


def test_is_higher_better_strips_the_split_prefix_every_logged_metric_carries():
    """Every metric this project actually logs is prefixed by which split produced it
    (``val_rmse``, ``test_f1``, ...) — a real bug caught by running `pdm registry compare`
    against real data: without stripping the prefix, `is_higher_better("test_rmse")` fell
    through to the "assume higher is better" default and ranked the *worst* RMSE first."""
    assert not is_higher_better("val_rmse")
    assert not is_higher_better("test_rmse")
    assert not is_higher_better("train_mae")
    assert is_higher_better("test_f1")
    assert is_higher_better("val_pr_auc")


def test_first_ever_promotion_always_passes_when_no_production_exists():
    decision = evaluate_gate(999.0, None, primary_metric="rmse", margin=0.02)
    assert decision.promoted
    assert decision.production_value is None


def test_lower_is_better_metric_requires_at_least_the_margin_improvement():
    # production RMSE 10.0, margin 2% -> candidate must be <= 9.8 to pass
    just_short = evaluate_gate(9.81, 10.0, primary_metric="rmse", margin=0.02)
    exactly_at = evaluate_gate(9.8, 10.0, primary_metric="rmse", margin=0.02)
    comfortably_better = evaluate_gate(9.0, 10.0, primary_metric="rmse", margin=0.02)

    assert not just_short.promoted
    assert exactly_at.promoted
    assert comfortably_better.promoted


def test_lower_is_better_metric_refuses_a_worse_candidate():
    decision = evaluate_gate(11.0, 10.0, primary_metric="rmse", margin=0.02)
    assert not decision.promoted


def test_higher_is_better_metric_requires_at_least_the_margin_improvement():
    # production F1 0.80, margin 2% -> candidate must be >= 0.816 to pass
    just_short = evaluate_gate(0.815, 0.80, primary_metric="f1", margin=0.02)
    exactly_at = evaluate_gate(0.816, 0.80, primary_metric="f1", margin=0.02)
    comfortably_better = evaluate_gate(0.90, 0.80, primary_metric="f1", margin=0.02)

    assert not just_short.promoted
    assert exactly_at.promoted
    assert comfortably_better.promoted


def test_higher_is_better_metric_refuses_a_worse_candidate():
    decision = evaluate_gate(0.5, 0.80, primary_metric="f1", margin=0.02)
    assert not decision.promoted


def test_decision_reports_both_values_and_the_margin_used():
    decision = evaluate_gate(9.0, 10.0, primary_metric="rmse", margin=0.05)
    assert isinstance(decision, PromotionDecision)
    assert decision.candidate_value == 9.0
    assert decision.production_value == 10.0
    assert decision.margin == 0.05
    assert decision.primary_metric == "rmse"
    assert "REFUSED" not in decision.reason  # this one passes; sanity check the message exists
