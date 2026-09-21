"""Hand-crafted window feature engineering for the classical (non-deep) baselines: flattened
raw values plus per-sensor summary statistics — the flat, tabular input Ridge/RandomForest need,
built from the same ``(N, window_size, n_features)`` arrays the deep models will consume later.
"""

import numpy as np


def _slope(window: np.ndarray) -> np.ndarray:
    """Per-column linear-trend slope over the window's time axis (ordinary least squares)."""
    n = window.shape[0]
    t = np.arange(n, dtype=float)
    t_centered = t - t.mean()
    denom = np.sum(t_centered**2)
    if denom == 0:
        return np.zeros(window.shape[1])
    return (t_centered @ (window - window.mean(axis=0))) / denom


def window_summary_features(windows: np.ndarray) -> np.ndarray:
    """``(N, window_size, n_features)`` -> ``(N, n_features * 5)``: per feature, the window's
    mean, std, min, max, and linear-trend slope."""
    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    mins = windows.min(axis=1)
    maxs = windows.max(axis=1)
    slopes = np.stack([_slope(w) for w in windows])
    return np.concatenate([means, stds, mins, maxs, slopes], axis=1)


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """``(N, window_size, n_features)`` -> ``(N, window_size * n_features)``."""
    return windows.reshape(windows.shape[0], -1)


def make_baseline_features(windows: np.ndarray) -> np.ndarray:
    """Flattened raw values concatenated with per-sensor summary statistics (spec §6, P03)."""
    return np.concatenate([flatten_windows(windows), window_summary_features(windows)], axis=1)
