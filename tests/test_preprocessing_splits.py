"""Grouped vs. naive splits: unit-level disjointness, determinism, and the naive split's leak."""

import pandas as pd
import pytest

from pdm.preprocessing.splits import naive_row_split, split_by_unit

pytestmark = pytest.mark.fast


def _make_df(n_units: int = 10, rows_per_unit: int = 5) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"unit": u, "cycle": range(1, rows_per_unit + 1)})
        for u in range(1, n_units + 1)
    ]
    return pd.concat(frames, ignore_index=True)


def test_split_by_unit_is_disjoint_and_covers_all_units():
    df = _make_df()
    train, val = split_by_unit(df, val_fraction=0.3, seed=1)
    assert set(train["unit"]).isdisjoint(val["unit"])
    assert set(train["unit"]) | set(val["unit"]) == set(df["unit"])


def test_split_by_unit_is_deterministic_given_seed():
    df = _make_df()
    _, val1 = split_by_unit(df, val_fraction=0.3, seed=7)
    _, val2 = split_by_unit(df, val_fraction=0.3, seed=7)
    assert set(val1["unit"]) == set(val2["unit"])


def test_naive_row_split_can_put_the_same_unit_on_both_sides():
    df = _make_df(n_units=3, rows_per_unit=20)
    train, val = naive_row_split(df, val_fraction=0.5, seed=0)
    assert set(train["unit"]) & set(val["unit"]), "overlap expected — that's the whole point"
