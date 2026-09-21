"""Grouped train/validation splits by engine unit — an engine's rows never appear on both
sides of a split. Also carries a deliberately wrong naive row-wise split used only to
demonstrate, in ``tests/test_leakage.py``, why the grouped split is necessary in the first place.
"""

import numpy as np
import pandas as pd


def split_by_unit(
    df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train/validation by engine unit id, so no unit's rows are split
    across both sides."""
    unit_ids = np.sort(df["unit"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unit_ids)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_units = set(shuffled[:n_val].tolist())

    val_df = df[df["unit"].isin(val_units)].copy()
    train_df = df[~df["unit"].isin(val_units)].copy()
    return train_df, val_df


def naive_row_split(
    df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random **row-wise** split, ignoring unit grouping entirely.

    This leaks information across train/validation, because consecutive cycles of the same
    engine are nearly identical — a model can partly "memorise" a unit from training rows and
    then recognise its held-out rows as near-duplicates. It exists *only* so
    ``tests/test_leakage.py`` can demonstrate that failure mode concretely and is never used by
    any real preprocessing or training path.
    """
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(df.index.to_numpy())
    n_val = int(round(len(shuffled_idx) * val_fraction))
    val_idx = shuffled_idx[:n_val]
    train_idx = shuffled_idx[n_val:]
    return df.loc[train_idx].copy(), df.loc[val_idx].copy()
