"""The human-readable drift report (P09 deliverable #5): markdown renders the verdict table and
embeds whatever plots it's given, and the plot generator picks the most-drifted features."""

import numpy as np
import pytest

from pdm.models.bundle import compute_reference_stats
from pdm.monitoring.drift import compute_input_drift
from pdm.monitoring.reports import generate_feature_plots, render_markdown_report

pytestmark = pytest.mark.fast


def _drift_result():
    rng = np.random.default_rng(0)
    train_windows = rng.normal(50, 5, size=(200, 5, 3)).astype(np.float32)
    reference_stats = compute_reference_stats(train_windows, ["a", "b", "c"])
    production = np.concatenate(
        [
            rng.normal(50, 5, size=(50, 5, 1)),
            rng.normal(50, 5, size=(50, 5, 1)),
            rng.normal(90, 5, size=(50, 5, 1)),  # feature "c" drifted hard
        ],
        axis=2,
    ).astype(np.float32)
    return reference_stats, production


def test_render_markdown_report_includes_table_and_aggregate(tmp_path):
    reference_stats, production = _drift_result()
    input_drift = compute_input_drift(reference_stats, production, ["a", "b", "c"])

    out_path = tmp_path / "report.md"
    render_markdown_report(input_drift, title="Test report", out_path=out_path)

    text = out_path.read_text(encoding="utf-8")
    assert "# Test report" in text
    assert "| feature | PSI |" in text
    assert "Aggregate drift score" in text
    for name in ("a", "b", "c"):
        assert name in text


def test_render_markdown_report_includes_optional_sections(tmp_path):
    reference_stats, production = _drift_result()
    input_drift = compute_input_drift(reference_stats, production, ["a", "b", "c"])
    prediction_drift = {
        "psi": 0.3,
        "psi_band": "significant",
        "ks_statistic": 0.5,
        "ks_pvalue": 1e-9,
        "ks_reject": True,
        "reference_mean": 10.0,
        "production_mean": 20.0,
    }
    degradation = {"rul": {"rmse": 12.3, "mae": 9.1, "nasa_score": 200.0}}

    out_path = tmp_path / "report.md"
    render_markdown_report(
        input_drift,
        title="Test report",
        out_path=out_path,
        prediction_drift=prediction_drift,
        degradation=degradation,
    )
    text = out_path.read_text(encoding="utf-8")
    assert "## Prediction drift" in text
    assert "## Measured model degradation" in text
    assert "12.3" in text


def test_render_markdown_report_states_the_verdict_when_triggered(tmp_path):
    reference_stats, production = _drift_result()
    input_drift = compute_input_drift(reference_stats, production, ["a", "b", "c"])
    trigger = {
        "triggered": True,
        "reason": "aggregate drift score 0.5 >= threshold 0.25, cooldown clear",
        "candidate_run_id": "run-abc",
        "candidate_model_name": "pdm-sentinel-rul",
        "candidate_version": "3",
    }

    out_path = tmp_path / "report.md"
    render_markdown_report(input_drift, title="Test report", out_path=out_path, trigger=trigger)
    text = out_path.read_text(encoding="utf-8")
    assert "## Verdict: RETRAIN TRIGGERED" in text
    assert "run-abc" in text
    assert "pdm registry promote --run-id run-abc" in text


def test_render_markdown_report_states_no_action_when_not_triggered(tmp_path):
    reference_stats, production = _drift_result()
    input_drift = compute_input_drift(reference_stats, production, ["a", "b", "c"])
    trigger = {
        "triggered": False,
        "reason": "aggregate drift score 0.05 below threshold 0.25",
        "candidate_run_id": None,
    }

    out_path = tmp_path / "report.md"
    render_markdown_report(input_drift, title="Test report", out_path=out_path, trigger=trigger)
    text = out_path.read_text(encoding="utf-8")
    assert "## Verdict: no action" in text
    assert "pdm registry promote" not in text


def test_generate_feature_plots_ranks_by_psi_and_writes_files(tmp_path):
    reference_stats, production = _drift_result()
    input_drift = compute_input_drift(reference_stats, production, ["a", "b", "c"])

    paths = generate_feature_plots(
        reference_stats, production, ["a", "b", "c"], input_drift, out_dir=tmp_path, top_k=2
    )
    assert len(paths) == 2
    assert "c" in paths  # the deliberately-shifted feature must be in the top-2
    for p in paths.values():
        assert p.exists()
