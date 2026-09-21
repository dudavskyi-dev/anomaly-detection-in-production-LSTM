"""The central leakage test suite (P02 acceptance: `pytest tests/test_leakage.py -v`).

Covers every leakage failure mode named in the spec: the scaler must actually differ when
fit on train-only vs. all data (proving it isn't silently cheating), splits must never share a
unit, no window may cross a unit boundary, AI4I's leakage flags must be absent from features,
the C-MAPSS test-set RUL offset must be applied, and — kept as living documentation — a naive
row-wise split must score implausibly better than a proper grouped split.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

from pdm.preprocessing.features import drop_ai4i_leakage_columns
from pdm.preprocessing.labelling import add_rul, add_rul_test
from pdm.preprocessing.scaling import StandardScaler
from pdm.preprocessing.splits import naive_row_split, split_by_unit
from pdm.preprocessing.windowing import make_windows

pytestmark = pytest.mark.fast


def _make_units(unit_lengths: dict[int, int]) -> pd.DataFrame:
    frames = []
    for unit, n in unit_lengths.items():
        cycles = np.arange(1, n + 1)
        frames.append(
            pd.DataFrame({"unit": unit, "cycle": cycles, "sensor_1": cycles.astype(float) * unit})
        )
    return pd.concat(frames, ignore_index=True)


# --- scaler: train-only fit must differ from a fit on everything -----------------------------


def test_scaler_fit_on_train_only_differs_from_fit_on_everything():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"sensor_1": rng.normal(0, 1, 200)})
    # validation drawn from a shifted distribution, as real held-out engines would be
    val = pd.DataFrame({"sensor_1": rng.normal(5, 1, 50)})
    everything = pd.concat([train, val], ignore_index=True)

    scaler_train_only = StandardScaler().fit(train, ["sensor_1"])
    scaler_everything = StandardScaler().fit(everything, ["sensor_1"])

    assert scaler_train_only.mean_ is not None and scaler_everything.mean_ is not None
    assert not np.allclose(scaler_train_only.mean_, scaler_everything.mean_)


# --- splits: no unit id in both sides ----------------------------------------------------------


def test_split_by_unit_has_no_shared_units():
    df = _make_units({u: 20 for u in range(1, 11)})
    train_df, val_df = split_by_unit(df, val_fraction=0.3, seed=0)
    assert set(train_df["unit"]).isdisjoint(set(val_df["unit"]))
    assert set(train_df["unit"]) | set(val_df["unit"]) == set(df["unit"])


# --- windowing: no window crosses a unit boundary ----------------------------------------------


def test_no_window_crosses_a_unit_boundary():
    df = _make_units({1: 12, 2: 9, 3: 15})
    windows, _, index_info = make_windows(
        df, ["sensor_1"], ["sensor_1"], window_size=6, stride=1, pad_short_units=False
    )
    for w, (unit, end_cycle) in zip(windows, index_info, strict=True):
        start_cycle = end_cycle - 6 + 1
        expected = np.arange(start_cycle, end_cycle + 1).astype(float) * unit
        np.testing.assert_array_equal(w[:, 0], expected)


# --- AI4I: leakage flags absent from features ---------------------------------------------------


def test_ai4i_failure_mode_flags_absent_after_feature_selection():
    df = pd.DataFrame(
        {
            "type": ["L", "M"],
            "machine_failure": [0, 1],
            "twf": [0, 0],
            "hdf": [0, 1],
            "pwf": [0, 0],
            "osf": [0, 0],
            "rnf": [0, 0],
        }
    )
    out = drop_ai4i_leakage_columns(df)
    assert not {"twf", "hdf", "pwf", "osf", "rnf"} & set(out.columns)


# --- C-MAPSS: test-set RUL offset must be applied -----------------------------------------------


def test_test_set_rul_offset_is_applied_correctly():
    df = pd.DataFrame({"unit": [1, 1, 1], "cycle": [1, 2, 3], "true_rul": [50, 50, 50]})
    correct = add_rul_test(df)
    # the classic bug: treating test data like train data, ignoring true_rul entirely
    naive_wrong = add_rul(df.drop(columns=["true_rul"]))

    assert correct["rul"].tolist() == [52, 51, 50]
    assert naive_wrong["rul"].tolist() == [2, 1, 0]
    # the naive version is wrong by exactly the offset that was dropped
    assert (correct["rul"] - naive_wrong["rul"] == 50).all()


# --- the naive-split demonstration, kept as documentation ---------------------------------------


def _make_synthetic_degradation_units(n_units: int = 6, n_cycles: int = 20, seed: int = 0):
    """Each unit has a distinct, well-separated sensor offset, so a model can only do well on a
    held-out unit by generalising — not by recognising rows it has effectively already seen."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_units + 1):
        cycles = np.arange(1, n_cycles + 1)
        rul = (n_cycles - cycles).astype(float)
        offset = unit * 10.0
        sensor_1 = offset - 0.5 * rul + rng.normal(0, 0.05, size=n_cycles)
        frames.append(
            pd.DataFrame({"unit": unit, "cycle": cycles, "sensor_1": sensor_1, "rul": rul})
        )
    return pd.concat(frames, ignore_index=True)


def _rmse_of_1nn(train: pd.DataFrame, val: pd.DataFrame) -> float:
    model = KNeighborsRegressor(n_neighbors=1)
    model.fit(train[["sensor_1"]], train["rul"])
    preds = model.predict(val[["sensor_1"]])
    return float(mean_squared_error(val["rul"], preds) ** 0.5)


def test_naive_row_split_scores_implausibly_better_than_grouped_split():
    """Documents *why* grouped splitting matters: a naive row-wise split lets a 1-NN model
    "cheat" via near-duplicate rows from the same engine, scoring far better than it has any
    right to. A grouped split, which never lets a unit appear on both sides, doesn't allow that.
    """
    df = _make_synthetic_degradation_units()

    train_grouped, val_grouped = split_by_unit(df, val_fraction=0.34, seed=0)
    train_naive, val_naive = naive_row_split(df, val_fraction=0.34, seed=0)

    grouped_rmse = _rmse_of_1nn(train_grouped, val_grouped)
    naive_rmse = _rmse_of_1nn(train_naive, val_naive)

    assert naive_rmse < grouped_rmse * 0.5, (
        f"expected the naive split to look implausibly better (leakage), got "
        f"naive={naive_rmse:.2f} vs grouped={grouped_rmse:.2f}"
    )
