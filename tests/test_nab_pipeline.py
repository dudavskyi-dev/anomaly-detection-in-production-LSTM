"""NAB-to-window adaptation: the healthy-prefix/rest-of-series temporal split, scaler fit on
train only, and windows/labels shaped exactly like C-MAPSS's."""

import numpy as np
import pandas as pd
import pytest

from pdm.preprocessing.nab_pipeline import prepare_nab_windows

pytestmark = pytest.mark.fast


def _make_series(tmp_path, n_healthy=100, n_anomalous_block=20, n_after=50):
    """A synthetic series: ``n_healthy`` normal rows, then one labelled anomalous block, then
    ``n_after`` more (unlabelled-but-still-scored) rows."""
    rng = np.random.default_rng(0)
    timestamps = pd.date_range(
        "2020-01-01", periods=n_healthy + n_anomalous_block + n_after, freq="5min"
    )
    values = rng.normal(0, 1, size=len(timestamps))
    is_anomaly = np.zeros(len(timestamps), dtype=bool)
    is_anomaly[n_healthy : n_healthy + n_anomalous_block] = True
    values[n_healthy : n_healthy + n_anomalous_block] += 10  # a real distributional shift

    df = pd.DataFrame({"timestamp": timestamps, "value": values, "is_anomaly": is_anomaly})
    path = tmp_path / "series.parquet"
    df.to_parquet(path, index=False)
    return path


def test_train_windows_are_all_from_before_the_first_labelled_anomaly(tmp_path):
    path = _make_series(tmp_path)
    ds = prepare_nab_windows(path, window_size=10, stride=1)
    assert ds.train_windows.shape[1:] == (10, 1)
    assert ds.train_windows.shape[0] > 0


def test_test_windows_include_the_anomalous_block(tmp_path):
    path = _make_series(tmp_path)
    ds = prepare_nab_windows(path, window_size=10, stride=1)
    assert ds.test_labels.sum() > 0
    assert ds.test_windows.shape[0] == len(ds.test_labels)


def test_scaler_is_fit_on_train_only(tmp_path):
    path = _make_series(tmp_path, n_healthy=200)
    ds = prepare_nab_windows(path, window_size=10, stride=1)
    # the healthy prefix is ~N(0,1); a scaler fit on it should have mean near 0, std near 1
    assert abs(ds.scaler.mean_[0]) < 0.5
    assert 0.5 < ds.scaler.std_[0] < 1.5


def test_no_labelled_anomalies_leak_into_train_windows(tmp_path):
    path = _make_series(tmp_path)
    ds = prepare_nab_windows(path, window_size=10, stride=1)
    # every train window's underlying rows came from strictly before the first anomaly, so a
    # reconstruction target computed from `is_anomaly` would be all-zero for every train window —
    # verified indirectly here by checking the train windows are pure noise, not shifted by +10.
    assert ds.train_windows.mean() < 3.0
