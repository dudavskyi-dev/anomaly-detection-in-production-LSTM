"""docs/RESULTS.md generation: built purely from metrics.json files, never hand-typed numbers."""

import json
from pathlib import Path

import pytest

from pdm.evaluation.results import generate_results_md

pytestmark = pytest.mark.fast


def _metric_entry(mean: float, std: float = 0.1) -> dict:
    return {"mean": mean, "std": std, "values": [mean], "seeds": [0]}


def _write_run(artifacts_dir: Path, run_id: str, payload: dict) -> None:
    out_dir = artifacts_dir / run_id
    out_dir.mkdir(parents=True)
    (out_dir / "metrics.json").write_text(json.dumps(payload))


def _no_anomaly_summary(tmp_path: Path) -> Path:
    """A guaranteed-nonexistent path — passed explicitly to every test below that isn't
    exercising the anomaly section, so these tests stay isolated from whatever
    ``artifacts/anomaly_experiments/summary.json`` happens to exist in the real repo (the
    default path `generate_results_md` uses when this argument is omitted, for real `pdm
    evaluate` runs)."""
    return tmp_path / "no_such_anomaly_summary.json"


def test_generate_results_md_includes_rul_table(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    _write_run(
        artifacts_dir,
        "rul_regression__ridge__FD001",
        {
            "run_id": "rul_regression__ridge__FD001",
            "task": "rul_regression",
            "model": "ridge",
            "subset": "FD001",
            "val": {"rmse": _metric_entry(22.0), "mae": _metric_entry(18.0)},
            "test": {
                "rmse": _metric_entry(21.5),
                "mae": _metric_entry(17.5),
                "nasa_score": _metric_entry(500.0),
            },
        },
    )
    output_path = tmp_path / "docs" / "RESULTS.md"
    content = generate_results_md(artifacts_dir, output_path, _no_anomaly_summary(tmp_path))

    assert "RUL regression" in content
    assert "ridge" in content
    assert "21.500 ± 0.100" in content
    assert "rul_regression__ridge__FD001" in content
    assert output_path.read_text(encoding="utf-8") == content


def test_generate_results_md_sorts_rul_models_by_test_rmse(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    _write_run(
        artifacts_dir,
        "rul_regression__dummy_mean__FD001",
        {
            "run_id": "rul_regression__dummy_mean__FD001",
            "task": "rul_regression",
            "model": "dummy_mean",
            "subset": "FD001",
            "val": {"rmse": _metric_entry(40.0), "mae": _metric_entry(35.0)},
            "test": {
                "rmse": _metric_entry(40.0),
                "mae": _metric_entry(35.0),
                "nasa_score": _metric_entry(9000.0),
            },
        },
    )
    _write_run(
        artifacts_dir,
        "rul_regression__ridge__FD001",
        {
            "run_id": "rul_regression__ridge__FD001",
            "task": "rul_regression",
            "model": "ridge",
            "subset": "FD001",
            "val": {"rmse": _metric_entry(22.0), "mae": _metric_entry(18.0)},
            "test": {
                "rmse": _metric_entry(21.5),
                "mae": _metric_entry(17.5),
                "nasa_score": _metric_entry(500.0),
            },
        },
    )
    content = generate_results_md(
        artifacts_dir, tmp_path / "docs" / "RESULTS.md", _no_anomaly_summary(tmp_path)
    )
    # ridge (lower test RMSE) must appear before dummy_mean in the table
    assert content.index("| ridge |") < content.index("| dummy_mean |")


def test_generate_results_md_omits_sections_with_no_runs(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    content = generate_results_md(
        artifacts_dir, tmp_path / "docs" / "RESULTS.md", _no_anomaly_summary(tmp_path)
    )
    assert "RUL regression" not in content
    assert "AUTO-GENERATED" in content


def test_generate_results_md_never_has_more_than_one_blank_line_in_a_row(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    content = generate_results_md(
        artifacts_dir, tmp_path / "docs" / "RESULTS.md", _no_anomaly_summary(tmp_path)
    )
    assert "\n\n\n" not in content


def _anomaly_metric_entry(f1: float) -> dict:
    return {
        "f1": _metric_entry(f1),
        "precision": _metric_entry(f1 + 0.05),
        "recall": _metric_entry(f1 - 0.05),
        "pr_auc": _metric_entry(f1 + 0.1),
    }


def _write_anomaly_summary(path: Path) -> None:
    payload = {
        "chosen_latent_dim": 8,
        "chosen_fusion_weight": 0.6,
        "cmapss": {
            "ae_alone": _anomaly_metric_entry(0.5),
            "isolation_forest_alone": _anomaly_metric_entry(0.4),
            "fused_by_threshold_style": {
                "max_f1": _anomaly_metric_entry(0.6),
                "precision_target": _anomaly_metric_entry(0.3),
                "percentile_99": _anomaly_metric_entry(0.55),
            },
            "mlflow_run_ids": {"0": "seed0aaa1111111111", "1": "seed1bbb2222222222"},
        },
        "nab": {
            "ae_alone": _anomaly_metric_entry(0.2),
            "isolation_forest_alone": _anomaly_metric_entry(0.25),
            "fused": _anomaly_metric_entry(0.22),
            "mlflow_run_ids": {"0": "nabseed0xxxxxxxxxx", "1": "nabseed1yyyyyyyyyy"},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_results_md_includes_anomaly_section_when_summary_exists(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    summary_path = tmp_path / "anomaly" / "summary.json"
    _write_anomaly_summary(summary_path)

    content = generate_results_md(artifacts_dir, tmp_path / "docs" / "RESULTS.md", summary_path)

    assert "Anomaly detection" in content
    assert "0.500 ± 0.100" in content  # AE-alone C-MAPSS F1
    assert "machine_temperature_system_failure" in content
    assert "MLflow Runs" in content
    # Every C-MAPSS row shares the same per-seed runs (see _anomaly_table's docstring) -- the
    # exact same joined, 8-char-truncated cell (matching `pdm registry compare`'s truncation
    # convention) appears once per row: 5 C-MAPSS rows, 3 NAB rows.
    assert content.count("`seed0aaa`, `seed1bbb`") == 5
    assert content.count("`nabseed0`, `nabseed1`") == 3


def test_generate_results_md_omits_anomaly_section_when_summary_missing(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    content = generate_results_md(
        artifacts_dir, tmp_path / "docs" / "RESULTS.md", _no_anomaly_summary(tmp_path)
    )
    assert "Anomaly detection" not in content
