"""Data drift detection (P09 deliverable #2): Population Stability Index (PSI) and two-sample
Kolmogorov-Smirnov (KS) tests against a bundle's frozen ``reference_stats.json``, plus an
optional check on the model's *output* distribution ("prediction drift").

Both PSI and KS compare **production traffic** (whatever windows the serving pipeline actually
saw) against the **training distribution frozen at bundle-save time**
(:func:`pdm.models.bundle.compute_reference_stats`) — never against a distribution recomputed
from "current" data, which would make every comparison a moving target.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from pdm.config import settings

# Below this proportion, a bin is treated as this value instead of its true (possibly zero)
# proportion before taking the PSI ratio/log. Without this floor, a bin with zero count on
# either side of the comparison divides by zero or takes log(0) — the "naive PSI" trap the P09
# spec calls out explicitly. 1e-4 is the commonly cited floor in the credit-scoring literature
# PSI comes from; it is small enough not to mask a real large shift, large enough that a single
# empty bin does not produce +/-inf.
DEFAULT_PSI_EPSILON = 1e-4


def psi_band(
    psi: float,
    *,
    moderate: float = 0.1,
    significant: float = 0.25,
) -> str:
    """The standard PSI interpretation bands: <0.1 no material shift, 0.1-0.25 moderate,
    >0.25 significant (population-stability-index convention, not specific to this project)."""
    if psi < moderate:
        return "none"
    if psi < significant:
        return "moderate"
    return "significant"


def psi_from_reference(
    bin_edges: list[float],
    bin_counts: list[int],
    sample: np.ndarray,
    *,
    epsilon: float = DEFAULT_PSI_EPSILON,
) -> float:
    """PSI of ``sample`` against a frozen reference histogram (``bin_edges``/``bin_counts``,
    both written once at training time — see :func:`pdm.models.bundle.compute_reference_stats`).

    ``bin_edges``' outer edges are expected to already be ``-inf``/``+inf`` (as
    ``compute_reference_stats`` writes them) so every sample value lands in some bin — an
    out-of-range production value is exactly the kind of thing PSI should be able to see as
    drift, not something ``np.histogram`` silently drops from the count.
    """
    sample = np.asarray(sample, dtype=float)
    if sample.size == 0:
        raise ValueError("psi_from_reference: sample is empty")

    ref_counts = np.asarray(bin_counts, dtype=float)
    prod_counts, _ = np.histogram(sample, bins=np.asarray(bin_edges, dtype=float))

    ref_pct = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    prod_pct = np.clip(prod_counts / prod_counts.sum(), epsilon, None)

    contributions = (prod_pct - ref_pct) * np.log(prod_pct / ref_pct)
    return float(contributions.sum())


def ks_two_sample(
    reference_sample: list[float] | np.ndarray, sample: np.ndarray
) -> tuple[float, float]:
    """Two-sample KS statistic and asymptotic p-value (``scipy.stats.ks_2samp``) between a
    frozen reference sample and a production sample of the same feature."""
    result = ks_2samp(np.asarray(reference_sample, dtype=float), np.asarray(sample, dtype=float))
    return float(result.statistic), float(result.pvalue)


def bonferroni_correction(pvalues: list[float], alpha: float) -> list[bool]:
    """Reject feature ``i`` iff ``pvalues[i] <= alpha / n``. Controls the family-wise error rate
    (probability of *any* false positive across all features) — the strict, conservative choice.
    """
    n = len(pvalues)
    if n == 0:
        return []
    threshold = alpha / n
    return [p <= threshold for p in pvalues]


def benjamini_hochberg_correction(pvalues: list[float], alpha: float) -> list[bool]:
    """Reject the largest set of hypotheses whose sorted p-values all fall under the BH line
    ``(rank / n) * alpha``. Controls the *false discovery rate* (expected proportion of false
    positives *among rejections*) rather than the probability of any false positive at all —
    less conservative than Bonferroni, which matters here because this project's sensor features
    are correlated (several C-MAPSS sensors move together with the same degradation trend), so a
    real shift tends to show up as p-values dropping together across many features at once, not
    as one isolated small p-value Bonferroni is tuned to still catch.
    """
    n = len(pvalues)
    if n == 0:
        return []
    order = np.argsort(pvalues)
    sorted_p = np.asarray(pvalues)[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresholds
    reject_sorted = np.zeros(n, dtype=bool)
    if below.any():
        k_max = int(np.max(np.where(below)[0]))
        reject_sorted[: k_max + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject.tolist()


_CORRECTION_METHODS = {
    "bonferroni": bonferroni_correction,
    "benjamini_hochberg": benjamini_hochberg_correction,
}


def compute_input_drift(
    reference_stats: dict,
    production_windows: np.ndarray,
    feature_names: list[str],
    *,
    alpha: float | None = None,
    correction: str | None = None,
    psi_moderate: float | None = None,
    psi_significant: float | None = None,
    psi_epsilon: float = DEFAULT_PSI_EPSILON,
) -> dict:
    """Per-feature PSI + corrected KS between ``reference_stats`` (frozen at training time) and
    ``production_windows`` (whatever traffic is being checked now), plus an aggregate score.

    Every timestep of every production window counts as one sample per feature — the same
    flattening ``compute_reference_stats`` used when it built the reference, so both sides are
    measured in the same units.
    """
    alpha = settings.monitoring.ks_alpha if alpha is None else alpha
    correction = settings.monitoring.ks_correction if correction is None else correction
    psi_moderate = (
        settings.monitoring.psi_moderate_threshold if psi_moderate is None else psi_moderate
    )
    psi_significant = (
        settings.monitoring.psi_significant_threshold
        if psi_significant is None
        else psi_significant
    )
    if correction not in _CORRECTION_METHODS:
        raise ValueError(
            f"unknown correction method {correction!r}, expected one of "
            f"{sorted(_CORRECTION_METHODS)}"
        )

    flat = production_windows.reshape(-1, production_windows.shape[-1])
    ref_features = reference_stats["features"]

    per_feature: dict[str, dict] = {}
    pvalues: list[float] = []
    for i, name in enumerate(feature_names):
        ref = ref_features[name]
        sample = flat[:, i]
        psi = psi_from_reference(ref["bin_edges"], ref["bin_counts"], sample, epsilon=psi_epsilon)
        ks_stat, ks_p = ks_two_sample(ref["reference_sample"], sample)
        per_feature[name] = {
            "psi": psi,
            "psi_band": psi_band(psi, moderate=psi_moderate, significant=psi_significant),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "mean_shift": float(sample.mean() - ref["mean"]),
            "std_ratio": float(sample.std() / ref["std"]) if ref["std"] > 0 else float("nan"),
        }
        pvalues.append(ks_p)

    reject = _CORRECTION_METHODS[correction](pvalues, alpha)
    for name, r in zip(feature_names, reject, strict=True):
        per_feature[name]["ks_reject"] = bool(r)

    psi_values = [per_feature[n]["psi"] for n in feature_names]
    max_idx = int(np.argmax(psi_values))
    return {
        "correction_method": correction,
        "alpha": alpha,
        "per_feature": per_feature,
        "aggregate": {
            # The mean, not the max, is this report's headline "aggregate_score" (what the
            # trigger compares against a threshold) — a single noisy sensor spiking PSI should
            # not alone fire a retrain; the per-feature table right above this is exactly where
            # "one sensor moved" is still visible. max_psi/max_psi_feature are reported alongside
            # for that reason, not folded into the aggregate itself.
            "mean_psi": float(np.mean(psi_values)),
            "max_psi": float(psi_values[max_idx]),
            "max_psi_feature": feature_names[max_idx],
            "n_features": len(feature_names),
            "n_features_psi_significant": int(sum(1 for v in psi_values if v >= psi_significant)),
            "n_features_ks_reject": int(sum(reject)),
        },
    }


def compute_prediction_drift(
    reference_predictions: np.ndarray,
    production_predictions: np.ndarray,
    *,
    n_bins: int = 10,
    alpha: float | None = None,
    reference_sample_size: int = 2000,
    sample_seed: int = 0,
    psi_moderate: float | None = None,
    psi_significant: float | None = None,
    psi_epsilon: float = DEFAULT_PSI_EPSILON,
) -> dict:
    """Drift of the model's **output** distribution (predicted RUL, or an anomaly score),
    computed the same way as input drift but against ``reference_predictions`` (typically the
    model's own predictions on a held-out reference split, scored fresh at check time — unlike
    per-feature input drift, there is no bundle artifact persisting a frozen prediction
    histogram, so this recomputes reference bin edges from whatever reference batch the caller
    passes in, each time it's called).

    **Why check this in addition to per-feature input drift, not instead of it:** input drift
    can miss a real shift that only shows up jointly — several correlated sensors each moving a
    little, in a combination the model is sensitive to, without any single feature's marginal
    histogram moving enough to trip its own PSI/KS. Prediction drift catches that, because it
    looks at what the model actually did with the input rather than the input alone. The
    converse also holds: prediction drift can stay quiet even while input drift is real, if the
    model happens to be insensitive to whatever moved (e.g. a saturated output range) — so a
    quiet prediction-drift check does not clear the input side, and vice versa. Neither
    subsumes the other; :func:`compute_input_drift`'s per-feature table is also the only place
    that names *which sensor* moved, which prediction drift cannot do.
    """
    alpha = settings.monitoring.ks_alpha if alpha is None else alpha
    psi_moderate = (
        settings.monitoring.psi_moderate_threshold if psi_moderate is None else psi_moderate
    )
    psi_significant = (
        settings.monitoring.psi_significant_threshold
        if psi_significant is None
        else psi_significant
    )

    reference_predictions = np.asarray(reference_predictions, dtype=float)
    production_predictions = np.asarray(production_predictions, dtype=float)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference_predictions, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference_predictions, bins=edges)

    psi = psi_from_reference(
        edges.tolist(), ref_counts.tolist(), production_predictions, epsilon=psi_epsilon
    )

    rng = np.random.default_rng(sample_seed)
    sample_size = min(reference_sample_size, reference_predictions.shape[0])
    ref_sample = rng.choice(reference_predictions, size=sample_size, replace=False)
    ks_stat, ks_p = ks_two_sample(ref_sample, production_predictions)

    return {
        "psi": psi,
        "psi_band": psi_band(psi, moderate=psi_moderate, significant=psi_significant),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "ks_reject": bool(ks_p <= alpha),
        "reference_mean": float(reference_predictions.mean()),
        "production_mean": float(production_predictions.mean()),
    }


def evaluate_model_degradation(
    rul_bundle,
    windows: np.ndarray,
    targets: dict,
    *,
    anomaly_bundle=None,
) -> dict:
    """The other half of the "is this drift real" question: not just "does the input look
    different" but "did the model actually get worse on it". Reuses the exact bundle objects
    ``pdm.serving.bundle`` already knows how to load and score — no separate model
    reconstruction here.

    RMSE always comes from the RUL regressor. F1 (an anomaly-detection classification metric)
    is only computed if ``anomaly_bundle`` is given, since not every drift check has one
    configured — mirrors ``/detect/anomaly``'s graceful-degradation contract in serving.
    """
    from pdm.evaluation.metrics import classification_metrics, regression_metrics

    scaled = rul_bundle.scale(windows)
    predicted_rul = rul_bundle.predict_rul(scaled)
    true_rul = np.asarray(targets["rul"], dtype=float)
    result = {"rul": regression_metrics(true_rul, predicted_rul)}

    if anomaly_bundle is not None:
        anomaly_scaled = anomaly_bundle.scale(windows)
        scores = anomaly_bundle.score(anomaly_scaled)
        is_anomaly = (true_rul < settings.model.healthy_rul_threshold).astype(int)
        predicted_anomaly = (scores >= anomaly_bundle.threshold).astype(int)
        result["anomaly"] = classification_metrics(
            is_anomaly, predicted_anomaly, scores, include_brier=False
        )

    return result
