"""StandardScaler: fit-then-transform round trip, a strict column contract, and JSON persistence."""

import numpy as np
import pandas as pd
import pytest

from pdm.preprocessing.scaling import (
    FeatureColumnMismatchError,
    ScalerNotFittedError,
    StandardScaler,
)

pytestmark = pytest.mark.fast


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})


def test_transform_then_inverse_round_trips():
    df = _sample_df()
    scaler = StandardScaler().fit(df, ["a", "b"])
    restored = scaler.inverse_transform(scaler.transform(df))
    np.testing.assert_allclose(restored.to_numpy(), df.to_numpy(), atol=1e-10)


def test_transform_produces_zero_mean_unit_std_on_the_fitted_data():
    df = _sample_df()
    scaler = StandardScaler().fit(df, ["a", "b"])
    scaled = scaler.transform(df)
    np.testing.assert_allclose(scaled.mean().to_numpy(), [0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(scaled.std(ddof=0).to_numpy(), [1.0, 1.0], atol=1e-10)


def test_transform_rejects_wrong_column_order():
    df = _sample_df()
    scaler = StandardScaler().fit(df, ["a", "b"])
    with pytest.raises(FeatureColumnMismatchError):
        scaler.transform(df[["b", "a"]])


def test_transform_rejects_extra_or_missing_columns():
    df = _sample_df()
    scaler = StandardScaler().fit(df, ["a", "b"])
    with pytest.raises(FeatureColumnMismatchError):
        scaler.transform(df[["a"]])


def test_use_before_fit_raises():
    scaler = StandardScaler()
    with pytest.raises(ScalerNotFittedError):
        scaler.transform(_sample_df())


def test_save_and_load_round_trip(tmp_path):
    df = _sample_df()
    scaler = StandardScaler().fit(df, ["a", "b"])
    path = tmp_path / "scaler.json"
    scaler.save(path)

    loaded = StandardScaler.load(path)
    pd.testing.assert_frame_equal(scaler.transform(df), loaded.transform(df))


def test_constant_column_does_not_divide_by_zero():
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0]})
    scaler = StandardScaler().fit(df, ["a", "b"])
    scaled = scaler.transform(df)
    assert np.isfinite(scaled.to_numpy()).all()
