"""Human-readable drift reports (P09 deliverable #5): a markdown PSI/KS table with a verdict,
plus overlaid before/after distribution plots for the most-drifted features.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_feature_distribution(
    reference_sample: list[float] | np.ndarray,
    production_sample: np.ndarray,
    feature_name: str,
    path: Path,
    *,
    bins: int = 30,
) -> None:
    """Overlaid histograms of ``reference_sample`` (the bundle's frozen training subsample —
    the "before") against ``production_sample`` (the "after") — density-normalised so the two
    are comparable regardless of how many points each side has."""
    fig, ax = plt.subplots()
    ax.hist(reference_sample, bins=bins, alpha=0.5, density=True, label="reference (train)")
    ax.hist(production_sample, bins=bins, alpha=0.5, density=True, label="production")
    ax.set_title(f"{feature_name}: reference vs. production")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("density")
    ax.legend()
    _save(fig, path)


def generate_feature_plots(
    reference_stats: dict,
    production_windows: np.ndarray,
    feature_names: list[str],
    input_drift: dict,
    *,
    out_dir: Path,
    top_k: int = 5,
) -> dict[str, Path]:
    """Plots the ``top_k`` features ranked by PSI (the ones worth a human actually looking at),
    not all of them — with 14+ sensors, a plot per feature on every check would mostly be noise
    nobody reads."""
    flat = production_windows.reshape(-1, production_windows.shape[-1])
    ranked = sorted(feature_names, key=lambda n: -input_drift["per_feature"][n]["psi"])[:top_k]
    paths = {}
    for name in ranked:
        i = feature_names.index(name)
        reference_sample = reference_stats["features"][name]["reference_sample"]
        production_sample = flat[:, i]
        path = out_dir / f"{name}.png"
        plot_feature_distribution(reference_sample, production_sample, name, path)
        paths[name] = path
    return paths


def render_markdown_report(
    input_drift: dict,
    *,
    title: str,
    out_path: Path,
    plot_paths: dict[str, Path] | None = None,
    prediction_drift: dict | None = None,
    degradation: dict | None = None,
    trigger: dict | None = None,
) -> Path:
    agg = input_drift["aggregate"]
    lines = [f"# {title}", ""]

    if trigger is not None:
        verdict = "RETRAIN TRIGGERED" if trigger.get("triggered") else "no action"
        lines.append(f"## Verdict: {verdict}")
        lines.append(f"- {trigger.get('reason', '')}")
        if trigger.get("candidate_run_id"):
            lines.append(
                f"- Candidate run `{trigger['candidate_run_id']}` registered as "
                f"`{trigger.get('candidate_model_name')}` v{trigger.get('candidate_version')} "
                "in stage **Staging** — NOT promoted. Run `pdm registry promote --run-id "
                f"{trigger['candidate_run_id']}` to attempt promotion through the gate."
            )
        lines.append("")

    lines.append(
        f"- Aggregate drift score (mean PSI across {agg['n_features']} features): "
        f"**{agg['mean_psi']:.4f}**\n"
        f"- Max single-feature PSI: **{agg['max_psi']:.4f}** ({agg['max_psi_feature']})\n"
        f"- Features with PSI at/above the significant threshold: "
        f"{agg['n_features_psi_significant']}\n"
        f"- Features with KS rejected after {input_drift['correction_method']} correction "
        f"(alpha={input_drift['alpha']}): {agg['n_features_ks_reject']}"
    )
    lines.append("")
    lines.append(
        "| feature | PSI | band | KS stat | KS p-value | KS reject | mean shift | std ratio |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, f in sorted(input_drift["per_feature"].items(), key=lambda kv: -kv[1]["psi"]):
        lines.append(
            f"| {name} | {f['psi']:.4f} | {f['psi_band']} | {f['ks_statistic']:.4f} | "
            f"{f['ks_pvalue']:.2e} | {f['ks_reject']} | {f['mean_shift']:+.4f} | "
            f"{f['std_ratio']:.3f} |"
        )
    lines.append("")

    if prediction_drift is not None:
        lines.append("## Prediction drift (model output distribution)")
        lines.append(
            f"PSI={prediction_drift['psi']:.4f} ({prediction_drift['psi_band']}), "
            f"KS stat={prediction_drift['ks_statistic']:.4f}, "
            f"p={prediction_drift['ks_pvalue']:.2e}, reject={prediction_drift['ks_reject']}. "
            f"Reference mean={prediction_drift['reference_mean']:.3f}, "
            f"production mean={prediction_drift['production_mean']:.3f}."
        )
        lines.append("")

    if degradation is not None:
        lines.append("## Measured model degradation on this traffic")
        rul = degradation.get("rul", {})
        lines.append(
            f"RUL RMSE: {rul.get('rmse', float('nan')):.3f}, "
            f"MAE: {rul.get('mae', float('nan')):.3f}, "
            f"NASA score: {rul.get('nasa_score', float('nan')):.3f}"
        )
        if "anomaly" in degradation:
            an = degradation["anomaly"]
            lines.append(
                f"Anomaly-detector F1: {an.get('f1', float('nan')):.3f} "
                f"(precision={an.get('precision', float('nan')):.3f}, "
                f"recall={an.get('recall', float('nan')):.3f})"
            )
        lines.append("")

    if plot_paths:
        lines.append("## Per-feature distribution shift (most drifted)")
        for name, p in plot_paths.items():
            lines.append(f"### {name}")
            lines.append(f"![{name}]({p.name})")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
