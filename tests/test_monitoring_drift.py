"""PSI/KS drift math (P09 deliverable #6): PSI against a hand-computed value, the zero-bin
handling the naive formula gets wrong, a no-drift case producing a low score, a synthetic shifted
distribution producing a high one, and the two correction methods disagreeing on a realistic
correlated-shift scenario.
"""

import numpy as np
import pytest

from pdm.models.bundle import compute_reference_stats
from pdm.monitoring.drift import (
    benjamini_hochberg_correction,
    bonferroni_correction,
    compute_input_drift,
    compute_prediction_drift,
    evaluate_model_degradation,
    ks_two_sample,
    psi_band,
    psi_from_reference,
)
from pdm.serving.bundle import load_rul_bundle

pytestmark = pytest.mark.fast


def test_psi_is_zero_for_identical_distributions():
    edges = [-np.inf, 1, 2, 3, np.inf]
    counts = [25, 25, 25, 25]
    sample = np.array([0.5, 1.5, 2.5, 3.5] * 25)
    assert psi_from_reference(edges, counts, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_hand_computed_value_with_zero_count_bins():
    """Hand-derived expected value (see docs/decisions/P09-drift.md): a production sample that
    lands entirely in one reference bin, leaving the other three at zero count. Naive PSI would
    divide by zero / take log(0) on those; the epsilon floor keeps it finite and large."""
    edges = [-np.inf, 1, 2, 3, np.inf]
    counts = [25, 25, 25, 25]
    sample = np.full(100, 0.5)
    psi = psi_from_reference(edges, counts, sample)
    assert psi == pytest.approx(6.90540806517888, rel=1e-6)
    assert psi_band(psi) == "significant"


def test_psi_from_reference_rejects_empty_sample():
    with pytest.raises(ValueError):
        psi_from_reference([-np.inf, 0, np.inf], [50, 50], np.array([]))


def test_psi_band_thresholds():
    assert psi_band(0.05) == "none"
    assert psi_band(0.1) == "moderate"
    assert psi_band(0.2) == "moderate"
    assert psi_band(0.25) == "significant"
    assert psi_band(0.5) == "significant"


def test_compute_reference_stats_freezes_quantile_bins_and_reference_sample():
    rng = np.random.default_rng(0)
    windows = rng.normal(50, 5, size=(40, 5, 2))
    stats = compute_reference_stats(
        windows, ["a", "b"], n_bins=4, reference_sample_size=10, sample_seed=0
    )
    assert stats["n_windows"] == 40
    assert stats["n_samples_per_feature"] == 40 * 5
    for name in ("a", "b"):
        feat = stats["features"][name]
        assert len(feat["bin_edges"]) == 5
        assert feat["bin_edges"][0] == -np.inf
        assert feat["bin_edges"][-1] == np.inf
        assert sum(feat["bin_counts"]) == 40 * 5
        assert len(feat["reference_sample"]) == 10


def test_compute_reference_stats_caps_reference_sample_to_available_data():
    windows = np.random.default_rng(0).normal(size=(3, 2, 1))  # only 6 samples for this feature
    stats = compute_reference_stats(windows, ["x"], reference_sample_size=2000)
    assert len(stats["features"]["x"]["reference_sample"]) == 6


def test_no_drift_case_produces_a_low_aggregate_psi():
    rng = np.random.default_rng(1)
    train_windows = rng.normal(50, 5, size=(500, 5, 3)).astype(np.float32)
    reference_stats = compute_reference_stats(train_windows, ["a", "b", "c"])

    holdout = rng.normal(50, 5, size=(100, 5, 3)).astype(np.float32)
    result = compute_input_drift(reference_stats, holdout, ["a", "b", "c"])

    assert result["aggregate"]["mean_psi"] < 0.1
    assert result["aggregate"]["n_features_psi_significant"] == 0


def test_shifted_distribution_produces_a_high_aggregate_psi():
    rng = np.random.default_rng(1)
    train_windows = rng.normal(50, 5, size=(500, 5, 3)).astype(np.float32)
    reference_stats = compute_reference_stats(train_windows, ["a", "b", "c"])

    shifted = rng.normal(90, 5, size=(100, 5, 3)).astype(np.float32)  # 8 std devs away
    result = compute_input_drift(reference_stats, shifted, ["a", "b", "c"])

    assert result["aggregate"]["mean_psi"] > 0.25
    assert result["aggregate"]["n_features_psi_significant"] == 3
    assert result["aggregate"]["n_features_ks_reject"] == 3


def test_ks_two_sample_rejects_a_clearly_different_distribution():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, size=1000)
    same = rng.normal(0, 1, size=1000)
    shifted = rng.normal(5, 1, size=1000)

    _, p_same = ks_two_sample(reference, same)
    _, p_shifted = ks_two_sample(reference, shifted)
    assert p_same > 0.05
    assert p_shifted < 0.001


def test_bonferroni_correction_basic():
    pvalues = [0.5, 0.5, 0.5]
    assert bonferroni_correction(pvalues, 0.05) == [False, False, False]
    pvalues = [0.001, 0.5, 0.5]
    assert bonferroni_correction(pvalues, 0.05) == [True, False, False]


def test_benjamini_hochberg_correction_basic():
    pvalues = [0.5, 0.5, 0.5]
    assert benjamini_hochberg_correction(pvalues, 0.05) == [False, False, False]
    pvalues = [0.001, 0.5, 0.5]
    assert benjamini_hochberg_correction(pvalues, 0.05) == [True, False, False]


def test_benjamini_hochberg_is_less_conservative_than_bonferroni_on_correlated_shift():
    """The real-world case the two methods disagree on: many features whose p-values drop
    together (a correlated shift, like several C-MAPSS sensors moving with the same degradation
    trend) rather than one isolated small p-value. Bonferroni's flat alpha/n threshold catches
    none of these; BH's rank-dependent line catches several."""
    pvalues = [0.01] * 10 + [0.5] * 4
    alpha = 0.05
    bonf = bonferroni_correction(pvalues, alpha)
    bh = benjamini_hochberg_correction(pvalues, alpha)
    assert sum(bonf) == 0
    assert sum(bh) == 10
    assert bonf != bh


def test_compute_input_drift_rejects_unknown_correction_method():
    reference_stats = compute_reference_stats(
        np.random.default_rng(0).normal(size=(20, 5, 1)), ["a"]
    )
    with pytest.raises(ValueError):
        compute_input_drift(
            reference_stats,
            np.random.default_rng(0).normal(size=(5, 5, 1)),
            ["a"],
            correction="not_a_real_method",
        )


def test_prediction_drift_low_for_matched_reference_and_production():
    rng = np.random.default_rng(3)
    reference_predictions = rng.normal(80, 10, size=2000)
    production_predictions = rng.normal(80, 10, size=300)
    result = compute_prediction_drift(reference_predictions, production_predictions)
    assert result["psi"] < 0.1
    assert not result["ks_reject"]


def test_prediction_drift_high_for_shifted_predictions():
    rng = np.random.default_rng(3)
    reference_predictions = rng.normal(80, 10, size=500)
    production_predictions = rng.normal(20, 10, size=100)
    result = compute_prediction_drift(reference_predictions, production_predictions)
    assert result["psi"] > 0.25
    assert result["ks_reject"]


def test_evaluate_model_degradation_uses_the_bundles_own_rmse_math(tiny_rul_bundle_dir):
    bundle = load_rul_bundle(tiny_rul_bundle_dir)
    n = 15
    rng = np.random.default_rng(0)
    windows = rng.normal(
        size=(n, bundle.feature_spec.window_size, len(bundle.feature_spec.feature_names))
    ).astype(np.float32)
    targets = {"rul": rng.uniform(0, 125, size=n)}

    result = evaluate_model_degradation(bundle, windows, targets)
    assert "rul" in result
    assert result["rul"]["rmse"] >= 0.0
    assert "anomaly" not in result  # no anomaly_bundle passed -> graceful omission
