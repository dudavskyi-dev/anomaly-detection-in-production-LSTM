"""AI4I baselines: leakage columns are dropped from the split (and never in the model's
feature allowlist to begin with), and real signal beats the dummy floor."""

import numpy as np
import pandas as pd
import pytest

from pdm.models.baseline.ai4i import (
    dummy_ai4i_baseline,
    logistic_regression_ai4i_baseline,
    prepare_ai4i_splits,
    random_forest_ai4i_baseline,
)

pytestmark = pytest.mark.fast


def _make_synthetic_ai4i(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    torque = rng.normal(40, 10, n)
    failure = (torque > 55).astype(int)  # clear, learnable signal
    return pd.DataFrame(
        {
            "type": rng.choice(["L", "M", "H"], size=n),
            "air_temperature": rng.normal(300, 2, n),
            "process_temperature": rng.normal(310, 2, n),
            "rotational_speed": rng.normal(1500, 100, n),
            "torque": torque,
            "tool_wear": rng.integers(0, 250, n),
            "machine_failure": failure,
            "twf": 0,
            "hdf": 0,
            "pwf": failure,  # deliberately 1:1 with the target, to prove it's excluded
            "osf": 0,
            "rnf": 0,
        }
    )


def test_prepare_ai4i_splits_drops_leakage_columns():
    df = _make_synthetic_ai4i()
    train_df, val_df = prepare_ai4i_splits(df, seed=0)
    for leak_col in ("twf", "hdf", "pwf", "osf", "rnf"):
        assert leak_col not in train_df.columns
        assert leak_col not in val_df.columns


def test_prepare_ai4i_splits_is_disjoint_and_roughly_the_right_size():
    df = _make_synthetic_ai4i()
    train_df, val_df = prepare_ai4i_splits(df, val_fraction=0.2, seed=0)
    assert set(train_df.index).isdisjoint(val_df.index)
    assert len(val_df) == pytest.approx(len(df) * 0.2, abs=1)


def test_logistic_regression_beats_dummy_on_learnable_ai4i_data():
    df = _make_synthetic_ai4i()
    train_df, val_df = prepare_ai4i_splits(df, seed=0)

    dummy = dummy_ai4i_baseline(train_df, val_df, seed=0)
    logreg = logistic_regression_ai4i_baseline(train_df, val_df, seed=0)

    assert logreg["pr_auc"] > dummy["pr_auc"]


def test_random_forest_beats_dummy_on_learnable_ai4i_data():
    df = _make_synthetic_ai4i()
    train_df, val_df = prepare_ai4i_splits(df, seed=0)

    dummy = dummy_ai4i_baseline(train_df, val_df, seed=0)
    rf = random_forest_ai4i_baseline(train_df, val_df, seed=0)

    assert rf["pr_auc"] > dummy["pr_auc"]
