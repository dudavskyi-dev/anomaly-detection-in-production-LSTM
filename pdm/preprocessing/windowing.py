"""Leakage-free sliding windows, grouped by engine unit so a window never spans two engines,
plus the `feature_spec.json` contract that records exactly how they were built.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    """Everything needed to reproduce this dataset's windows from raw preprocessed rows."""

    feature_names: list[str]
    window_size: int
    stride: int
    pad_short_units: bool
    per_condition_normalization: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> "FeatureSpec":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def make_windows(
    df: pd.DataFrame,
    feature_names: list[str],
    target_cols: list[str],
    window_size: int,
    stride: int,
    *,
    pad_short_units: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[tuple[int, int]]]:
    """Build ``(N, window_size, n_features)`` windows and aligned targets, grouped by unit.

    A window never spans two units — each unit's rows are windowed independently. Units with
    fewer than ``window_size`` rows are either left-padded (repeating the unit's first row,
    the reading closest to "still healthy" available) or dropped, per ``pad_short_units``.
    Every target in ``target_cols`` is read from the window's **last** timestep.

    Returns the window array, a dict of ``{target_col: (N,) array}``, and a parallel list of
    ``(unit, end_cycle)`` so any window can be traced back to the row that produced its target.
    """
    windows: list[np.ndarray] = []
    targets: dict[str, list[float]] = {col: [] for col in target_cols}
    index_info: list[tuple[int, int]] = []

    for unit, group in df.groupby("unit", sort=True):
        group = group.sort_values("cycle")
        features = group[feature_names].to_numpy(dtype=float)
        cycles = group["cycle"].to_numpy()
        target_arrays = {col: group[col].to_numpy(dtype=float) for col in target_cols}
        n = len(group)

        if n < window_size:
            if not pad_short_units:
                continue
            pad_count = window_size - n
            features = np.vstack([np.repeat(features[:1], pad_count, axis=0), features])
            cycles = np.concatenate([np.repeat(cycles[:1], pad_count), cycles])
            target_arrays = {
                col: np.concatenate([np.repeat(arr[:1], pad_count), arr])
                for col, arr in target_arrays.items()
            }
            n = window_size

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            windows.append(features[start:end])
            index_info.append((int(unit), int(cycles[end - 1])))
            for col in target_cols:
                targets[col].append(float(target_arrays[col][end - 1]))

    if not windows:
        empty_windows = np.empty((0, window_size, len(feature_names)))
        empty_targets = {col: np.empty((0,)) for col in target_cols}
        return empty_windows, empty_targets, []

    stacked = np.stack(windows)
    target_arrays_out = {col: np.array(values) for col, values in targets.items()}
    return stacked, target_arrays_out, index_info
