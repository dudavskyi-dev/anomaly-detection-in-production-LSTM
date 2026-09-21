"""Prometheus metrics for the drift-check batch job (P09 deliverable #4; P10 deliverable #1
names two of these metrics explicitly: ``pdm_drift_score``/``pdm_drift_score_aggregate``).

``pdm drift check`` runs as a short-lived process invoked on a schedule (a compose-managed cron
job or any simple scheduler — see ``docs/decisions/P09-drift.md`` for why not a long-lived
server), so there is no live process for Prometheus to scrape *from*. This writes the
Prometheus **textfile collector** format instead
(https://github.com/prometheus/node_exporter#textfile-collector) — the standard,
dependency-free pattern for batch/cron jobs: node_exporter (or anything textfile-collector
compatible) picks the file up on its next scrape, and the file's own mtime becomes a free
"has this job actually run recently" signal. Prometheus scrapes node_exporter (and therefore
this file) as its own target, separate from the API's own ``/metrics`` (``docker/prometheus.yml``)
— see ``docs/decisions/P10-monitoring.md`` for why these live here rather than in
``pdm.serving.metrics``: the API must never import training/analysis code
(``tests/test_serving_import_graph.py``), so it cannot compute or refresh these itself.
"""

from __future__ import annotations

from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, write_to_textfile


def write_drift_metrics(
    *,
    target: str,
    input_drift: dict,
    triggered: bool,
    cooldown_active: bool,
    path: Path,
    prediction_drift: dict | None = None,
) -> None:
    registry = CollectorRegistry()
    agg = input_drift["aggregate"]
    target_only = ["target"]

    def gauge(name: str, help_text: str, extra_labels: list[str] | None = None) -> Gauge:
        return Gauge(name, help_text, target_only + (extra_labels or []), registry=registry)

    # The exact names P10's spec asks for verbatim.
    drift_score = gauge("pdm_drift_score", "Per-feature PSI from this drift check", ["feature"])
    for feature, per_feature in input_drift["per_feature"].items():
        drift_score.labels(target, feature).set(per_feature["psi"])
    gauge(
        "pdm_drift_score_aggregate", "Aggregate (mean) PSI across features from this drift check"
    ).labels(target).set(agg["mean_psi"])

    # Additional diagnostic gauges beyond P10's minimum ask -- kept because they're genuinely
    # useful (max/significant-count/KS-reject give a dashboard more than the mean alone can) and
    # nothing about adding them conflicts with the two names above.
    gauge("pdm_drift_psi_mean", "Mean PSI across input features").labels(target).set(
        agg["mean_psi"]
    )
    gauge("pdm_drift_psi_max", "Max single-feature PSI").labels(target).set(agg["max_psi"])
    gauge(
        "pdm_drift_features_psi_significant",
        "Count of features with PSI at/above the significant threshold",
    ).labels(target).set(agg["n_features_psi_significant"])
    gauge(
        "pdm_drift_features_ks_reject",
        "Count of features with KS rejected after multiple-comparison correction",
    ).labels(target).set(agg["n_features_ks_reject"])
    gauge("pdm_drift_retrain_triggered", "1 if this check launched a retraining candidate").labels(
        target
    ).set(1 if triggered else 0)
    gauge(
        "pdm_drift_retrain_cooldown_active", "1 if the retrain trigger is currently in cooldown"
    ).labels(target).set(1 if cooldown_active else 0)
    if prediction_drift is not None:
        gauge(
            "pdm_drift_prediction_psi", "PSI of the model's output (prediction) distribution"
        ).labels(target).set(prediction_drift["psi"])

    path.parent.mkdir(parents=True, exist_ok=True)
    write_to_textfile(str(path), registry)
