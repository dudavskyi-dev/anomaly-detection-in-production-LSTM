"""End-to-end pipeline smoke test against the real, locally cached C-MAPSS data.

Skipped if `pdm data download` hasn't been run — this deliberately exercises the full pipeline
against real data rather than a fixture, so it needs `data/processed/` to be populated first,
exactly as P11's CI smoke-training job will do before running the suite.
"""

from pathlib import Path

import numpy as np
import pytest

from pdm.preprocessing.pipeline import load_raw_cmapss_windows, prepare_cmapss_subset

PROCESSED_DIR = Path("data/processed")

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "cmapss" / "train_FD001.parquet").exists(),
    reason="requires `pdm data download` (data/processed/cmapss/*.parquet) to have been run",
)


def test_fd001_pipeline_produces_finite_windows_with_expected_shape():
    ds = prepare_cmapss_subset("FD001", PROCESSED_DIR)
    assert ds.train.windows.ndim == 3
    assert ds.train.windows.shape[1] == 30
    assert ds.train.windows.shape[2] == len(ds.feature_spec.feature_names)
    assert np.isfinite(ds.train.windows).all()
    assert np.isfinite(ds.val.windows).all()
    assert np.isfinite(ds.test.windows).all()
    assert set(ds.train.targets) == {"rul", "will_fail"}
    assert ds.train.targets["rul"].max() <= 125
    assert ds.feature_spec.per_condition_normalization is False


def test_fd002_pipeline_applies_per_condition_normalisation():
    ds = prepare_cmapss_subset("FD002", PROCESSED_DIR)
    assert ds.feature_spec.per_condition_normalization is True
    assert np.isfinite(ds.train.windows).all()
    assert np.isfinite(ds.val.windows).all()
    assert np.isfinite(ds.test.windows).all()


def test_load_raw_cmapss_windows_is_unscaled_unlike_prepare_cmapss_subset():
    """P09's drift checks need genuinely raw sensor values (to feed through an *already
    trained* bundle's own scaler) — not FD001's own re-fit StandardScaler output, which is what
    prepare_cmapss_subset always applies. The two must disagree in scale on the same data."""
    fd001 = prepare_cmapss_subset("FD001", PROCESSED_DIR)
    feature_names = fd001.feature_spec.feature_names

    raw = load_raw_cmapss_windows(
        "FD001", PROCESSED_DIR, feature_names=feature_names, window_size=30, stride=1, split="test"
    )
    assert raw.windows.shape[1:] == (30, len(feature_names))
    assert np.isfinite(raw.windows).all()
    assert set(raw.targets) == {"rul", "will_fail"}
    # prepare_cmapss_subset's own test split is standardised (~unit variance); the raw loader's
    # is not -- if this ever matched, load_raw_cmapss_windows would be silently re-scaling.
    assert raw.windows.std() > 5 * fd001.test.windows.std()


def test_load_raw_cmapss_windows_selects_the_given_feature_list_across_subsets():
    """The genuine-drift scenario (docs/decisions/P09-drift.md): replay FD002 using FD001's own
    feature list, not FD002's own constant-column-filtered one -- the two subsets don't
    necessarily agree on which sensors are constant."""
    fd001 = prepare_cmapss_subset("FD001", PROCESSED_DIR)
    feature_names = fd001.feature_spec.feature_names

    raw_fd002 = load_raw_cmapss_windows(
        "FD002", PROCESSED_DIR, feature_names=feature_names, window_size=30, stride=1, split="test"
    )
    assert raw_fd002.windows.shape[-1] == len(feature_names)
    assert np.isfinite(raw_fd002.windows).all()


def test_load_raw_cmapss_windows_rejects_unknown_split():
    fd001 = prepare_cmapss_subset("FD001", PROCESSED_DIR)
    with pytest.raises(ValueError):
        load_raw_cmapss_windows(
            "FD001",
            PROCESSED_DIR,
            feature_names=fd001.feature_spec.feature_names,
            window_size=30,
            stride=1,
            split="bogus",
        )


def test_load_raw_cmapss_windows_genuinely_flows_through_the_replay_simulator():
    """P09 deliverable #3 asks for production traffic to be fed 'through the replay simulator'
    (pdm.ingestion.replay), not just a direct parquet read that happens to produce equivalent
    numbers. noise_std=0/dropout_prob=0 (the default) must reproduce the raw data exactly; a
    nonzero noise_std must perturb it -- proving the values actually passed through replay()'s
    per-record noise injection, not around it."""
    fd001 = prepare_cmapss_subset("FD001", PROCESSED_DIR)
    feature_names = fd001.feature_spec.feature_names

    clean = load_raw_cmapss_windows(
        "FD001", PROCESSED_DIR, feature_names=feature_names, window_size=30, stride=1, split="test"
    )
    noisy = load_raw_cmapss_windows(
        "FD001",
        PROCESSED_DIR,
        feature_names=feature_names,
        window_size=30,
        stride=1,
        split="test",
        noise_std=50.0,
        replay_seed=0,
    )
    assert not np.allclose(clean.windows, noisy.windows)
    assert noisy.windows.shape == clean.windows.shape


def test_load_raw_cmapss_windows_dropout_reduces_available_rows():
    fd001 = prepare_cmapss_subset("FD001", PROCESSED_DIR)
    feature_names = fd001.feature_spec.feature_names

    full = load_raw_cmapss_windows(
        "FD001", PROCESSED_DIR, feature_names=feature_names, window_size=30, stride=1, split="test"
    )
    with_dropout = load_raw_cmapss_windows(
        "FD001",
        PROCESSED_DIR,
        feature_names=feature_names,
        window_size=30,
        stride=1,
        split="test",
        dropout_prob=0.3,
        replay_seed=0,
    )
    # Dropped records shrink each unit's row count, which can only reduce (never grow) the
    # number of windows the same stride/window_size produce.
    assert with_dropout.windows.shape[0] <= full.windows.shape[0]
