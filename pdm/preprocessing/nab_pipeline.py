"""Adapts the NAB ``machine_temperature_system_failure`` univariate series into the same
``(N, window_size, n_features)`` window format C-MAPSS uses (P06 deliverable #6), so it can be
scored by the exact same trained detectors and threshold logic — no NAB-specific model code.

There is no natural train/validation/test split for a single real time series the way there is
for C-MAPSS's many engine units. The split used here is temporal and deliberately conservative:
**train** is every row strictly before the first labelled anomaly window starts (the only
stretch of the series that is *known* healthy), and **test** is everything from the training
cutoff onward — including, honestly, the labelled anomalies. There is no NAB validation split:
tuning a max-F1 or precision-target threshold would require labelled anomalies disjoint from the
ones scored as test, and this series only has four labelled events total, none of which can be
set aside without either starving training or leaking the only labels test would be scored
against. See ``docs/decisions/P06-anomaly.md`` for what this implies about which threshold style
is honestly usable on NAB.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.config import settings
from pdm.preprocessing.scaling import StandardScaler
from pdm.preprocessing.windowing import make_windows

FEATURE_NAME = "value"


@dataclass
class NabDataset:
    train_windows: np.ndarray  # healthy-only, for training a detector
    test_windows: np.ndarray
    test_labels: np.ndarray  # 0/1 per window, from the window's last timestep
    scaler: StandardScaler


def _to_single_series_df(df: pd.DataFrame) -> pd.DataFrame:
    """``make_windows`` groups by ``unit``/sorts by ``cycle``; a single time series is exactly
    one unit whose cycle is its row order."""
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["unit"] = 0
    df["cycle"] = np.arange(len(df))
    df["is_anomaly"] = df["is_anomaly"].astype(float)
    return df


def prepare_nab_windows(
    processed_path: Path,
    *,
    window_size: int | None = None,
    stride: int | None = None,
) -> NabDataset:
    window_size = settings.data.window_size if window_size is None else window_size
    stride = settings.data.stride if stride is None else stride

    df = pd.read_parquet(processed_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    first_anomaly_start = df.loc[df["is_anomaly"], "timestamp"].min()
    train_df = df[df["timestamp"] < first_anomaly_start].copy()
    test_df = df[df["timestamp"] >= train_df["timestamp"].max()].copy()

    scaler = StandardScaler().fit(train_df, [FEATURE_NAME])
    train_df[[FEATURE_NAME]] = scaler.transform(train_df[[FEATURE_NAME]]).to_numpy()
    test_df[[FEATURE_NAME]] = scaler.transform(test_df[[FEATURE_NAME]]).to_numpy()

    train_df = _to_single_series_df(train_df)
    test_df = _to_single_series_df(test_df)

    train_windows, _, _ = make_windows(
        train_df, [FEATURE_NAME], ["is_anomaly"], window_size, stride, pad_short_units=False
    )
    test_windows, test_targets, _ = make_windows(
        test_df, [FEATURE_NAME], ["is_anomaly"], window_size, stride, pad_short_units=False
    )
    test_labels = (test_targets["is_anomaly"] >= 0.5).astype(int)

    return NabDataset(
        train_windows=train_windows,
        test_windows=test_windows,
        test_labels=test_labels,
        scaler=scaler,
    )
