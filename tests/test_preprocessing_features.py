"""Constant-sensor dropping (reading a P01-style profile) and per-condition normalisation."""

import json

import numpy as np
import pandas as pd
import pytest

from pdm.preprocessing.features import (
    apply_condition_normalisation,
    drop_ai4i_leakage_columns,
    drop_constant_columns,
    fit_condition_stats,
    fit_operating_condition_clusters,
    load_constant_columns,
)

pytestmark = pytest.mark.fast


def test_drop_constant_columns_removes_only_named_columns():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [5, 5, 5]})
    out = drop_constant_columns(df, ["a", "c"])
    assert list(out.columns) == ["b"]


def test_load_constant_columns_reads_p01_profile_format(tmp_path):
    profile = {
        "FD001": {"constant_columns": ["sensor_1", "sensor_5"], "feature_std": {}},
        "FD002": {"constant_columns": [], "feature_std": {}},
    }
    path = tmp_path / "sensor_profile.json"
    path.write_text(json.dumps(profile))
    assert load_constant_columns(path, "FD001") == ["sensor_1", "sensor_5"]
    assert load_constant_columns(path, "FD002") == []


def _synthetic_two_condition_df(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 200
    condition = rng.integers(0, 2, size=n)
    op1 = np.where(condition == 0, 10.0, 90.0) + rng.normal(0, 0.1, n)
    sensor = np.where(condition == 0, 500.0, 700.0) + rng.normal(0, 1.0, n)
    return pd.DataFrame({"op_setting_1": op1, "sensor_2": sensor}), condition


def test_fit_operating_condition_clusters_recovers_known_conditions():
    df, true_condition = _synthetic_two_condition_df()
    clusterer = fit_operating_condition_clusters(df, ["op_setting_1"], n_clusters=2, seed=0)
    labels = clusterer.predict(df[["op_setting_1"]].to_numpy())
    # cluster ids may be swapped relative to the synthetic ground truth; compare the partition
    agree = (labels == true_condition).mean()
    assert agree > 0.99 or agree < 0.01


def test_apply_condition_normalisation_removes_between_condition_offset():
    df, true_condition = _synthetic_two_condition_df()
    df = df.assign(condition=true_condition)
    stats = fit_condition_stats(df, ["sensor_2"])
    normalised = apply_condition_normalisation(df, ["sensor_2"], stats)
    group_means = normalised.groupby("condition")["sensor_2"].mean()
    np.testing.assert_allclose(group_means.to_numpy(), [0.0, 0.0], atol=1e-6)


def test_drop_ai4i_leakage_columns_removes_all_five_flags():
    df = pd.DataFrame(
        {
            "type": ["L"],
            "machine_failure": [1],
            "twf": [0],
            "hdf": [1],
            "pwf": [0],
            "osf": [0],
            "rnf": [0],
        }
    )
    out = drop_ai4i_leakage_columns(df)
    assert set(out.columns) == {"type", "machine_failure"}
