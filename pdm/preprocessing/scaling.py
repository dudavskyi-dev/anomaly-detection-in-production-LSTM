"""Feature scaling: fit on the train split only, serialised as human-readable JSON (never a
pickle), with a strict contract on which columns `transform`/`inverse_transform` will accept.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class ScalerNotFittedError(RuntimeError):
    pass


class FeatureColumnMismatchError(ValueError):
    pass


class StandardScaler:
    """Zero-mean, unit-variance standardisation with an explicit, ordered feature contract.

    ``transform``/``inverse_transform`` require the input dataframe's columns to equal the
    fitted ``feature_names`` exactly, in the same order — a caller must select the right
    columns itself rather than relying on this scaler to reorder or subset for them. This is
    deliberate: silently reindexing would hide a bug where the feature order drifted between
    training and serving.
    """

    def __init__(self) -> None:
        self.feature_names: list[str] | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, feature_names: list[str]) -> "StandardScaler":
        self.feature_names = list(feature_names)
        values = df[self.feature_names].to_numpy(dtype=float)
        self.mean_ = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        # a genuinely constant column would otherwise divide by zero; it should have been
        # dropped upstream (features.py), but guard rather than crash if one slips through.
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None or self.feature_names is None:
            raise ScalerNotFittedError("StandardScaler.fit() must be called before use")

    def _check_columns(self, df: pd.DataFrame) -> None:
        assert self.feature_names is not None
        if list(df.columns) != self.feature_names:
            raise FeatureColumnMismatchError(
                f"Column mismatch: scaler was fit on {self.feature_names}, got "
                f"{list(df.columns)}. Select and order columns explicitly before calling "
                "transform/inverse_transform."
            )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._check_columns(df)
        values = (df.to_numpy(dtype=float) - self.mean_) / self.std_
        return pd.DataFrame(values, columns=self.feature_names, index=df.index)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._check_columns(df)
        values = df.to_numpy(dtype=float) * self.std_ + self.mean_
        return pd.DataFrame(values, columns=self.feature_names, index=df.index)

    def to_dict(self) -> dict:
        self._check_fitted()
        assert self.mean_ is not None and self.std_ is not None
        return {
            "feature_names": self.feature_names,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "StandardScaler":
        obj = cls()
        obj.feature_names = list(data["feature_names"])
        obj.mean_ = np.array(data["mean"], dtype=float)
        obj.std_ = np.array(data["std"], dtype=float)
        return obj

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
