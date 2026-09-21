"""Sliding windows: correct shapes, last-timestep targets, unit grouping, and padding policy."""

import numpy as np
import pandas as pd
import pytest

from pdm.preprocessing.windowing import FeatureSpec, make_windows

pytestmark = pytest.mark.fast


def _make_df(unit_lengths: dict[int, int]) -> pd.DataFrame:
    frames = []
    for unit, n in unit_lengths.items():
        cycles = np.arange(1, n + 1)
        frames.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "cycle": cycles,
                    "f1": cycles.astype(float),
                    "rul": (n - cycles).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_window_shape_and_count():
    df = _make_df({1: 10, 2: 8})
    windows, targets, index_info = make_windows(df, ["f1"], ["rul"], window_size=5, stride=1)
    assert windows.shape == (10, 5, 1)  # (10-5+1) + (8-5+1) = 6 + 4
    assert targets["rul"].shape == (10,)
    assert len(index_info) == 10


def test_target_is_taken_from_last_timestep():
    df = _make_df({1: 6})
    _, targets, index_info = make_windows(df, ["f1"], ["rul"], window_size=3, stride=1)
    assert [c for (_, c) in index_info] == [3, 4, 5, 6]
    df_indexed = df.set_index("cycle")
    for target, cycle in zip(targets["rul"], [3, 4, 5, 6], strict=True):
        assert target == df_indexed.loc[cycle, "rul"]


def test_no_window_spans_two_units():
    df = _make_df({1: 5, 2: 5})
    windows, _, index_info = make_windows(df, ["f1"], ["rul"], window_size=5, stride=1)
    for w, (_unit, end_cycle) in zip(windows, index_info, strict=True):
        expected = np.arange(end_cycle - 4, end_cycle + 1).astype(float)
        np.testing.assert_array_equal(w[:, 0], expected)


def test_short_unit_is_dropped_when_padding_disabled():
    df = _make_df({1: 3, 2: 10})
    _, _, index_info = make_windows(
        df, ["f1"], ["rul"], window_size=5, stride=1, pad_short_units=False
    )
    assert {u for u, _ in index_info} == {2}


def test_short_unit_is_left_padded_when_enabled():
    df = _make_df({1: 3})
    windows, _, _ = make_windows(df, ["f1"], ["rul"], window_size=5, stride=1, pad_short_units=True)
    assert windows.shape == (1, 5, 1)
    np.testing.assert_array_equal(windows[0][:, 0], [1.0, 1.0, 1.0, 2.0, 3.0])


def test_stride_skips_windows():
    df = _make_df({1: 10})
    w1, _, _ = make_windows(df, ["f1"], ["rul"], window_size=5, stride=1)
    w2, _, _ = make_windows(df, ["f1"], ["rul"], window_size=5, stride=2)
    assert w1.shape[0] == 6
    assert w2.shape[0] == 3


def test_feature_spec_round_trip(tmp_path):
    spec = FeatureSpec(
        feature_names=["sensor_2", "sensor_3"],
        window_size=30,
        stride=1,
        pad_short_units=True,
        per_condition_normalization=False,
    )
    path = tmp_path / "feature_spec.json"
    spec.save(path)
    assert FeatureSpec.load(path) == spec
