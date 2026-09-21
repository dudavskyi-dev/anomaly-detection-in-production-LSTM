"""Feature selection: dropping constant sensors (using P01's derived profile, never hardcoded)
and per-condition operating-condition normalisation for the multi-condition C-MAPSS subsets.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

AI4I_LEAKAGE_COLUMNS = ("twf", "hdf", "pwf", "osf", "rnf")


def load_constant_columns(sensor_profile_path: Path, subset: str) -> list[str]:
    """Read the constant-sensor list P01 derived for ``subset`` — never hardcoded from memory."""
    profile = json.loads(sensor_profile_path.read_text(encoding="utf-8"))
    return list(profile[subset]["constant_columns"])


def drop_constant_columns(df: pd.DataFrame, constant_columns: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in constant_columns if c in df.columns])


def choose_n_operating_conditions(
    df: pd.DataFrame,
    op_setting_cols: list[str],
    k_range: range,
    seed: int,
    sample_size: int = 5000,
) -> dict[int, float]:
    """Silhouette score for each candidate k, so the "six operating conditions" claim for
    FD002/FD004 is verified against the data rather than assumed from the spec text.

    KMeans is fit on the full column set; the silhouette score itself (O(n^2) pairwise
    distances) is computed on a fixed-seed subsample so this stays practical on tens of
    thousands of rows.
    """
    X = df[list(op_setting_cols)].to_numpy()
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = float(
            silhouette_score(X, labels, sample_size=min(sample_size, len(X)), random_state=seed)
        )
    return scores


def fit_operating_condition_clusters(
    df: pd.DataFrame, op_setting_cols: list[str], n_clusters: int, seed: int
) -> KMeans:
    """Fit KMeans on operating settings. Fit on train only; reuse via `.predict()` for val/test
    to avoid leaking validation/test condition assignments into the clustering itself."""
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(
        df[list(op_setting_cols)].to_numpy()
    )


def assign_condition(
    df: pd.DataFrame, op_setting_cols: list[str], clusterer: KMeans
) -> pd.DataFrame:
    df = df.copy()
    df["condition"] = clusterer.predict(df[list(op_setting_cols)].to_numpy())
    return df


def fit_condition_stats(
    df: pd.DataFrame, feature_cols: list[str], condition_col: str = "condition"
) -> dict[int, dict[str, tuple[float, float]]]:
    """Per-condition (mean, std) for each feature, computed on train only."""
    stats: dict[int, dict[str, tuple[float, float]]] = {}
    for condition, group in df.groupby(condition_col):
        stats[int(condition)] = {
            col: (float(group[col].mean()), float(group[col].std(ddof=0)) or 1.0)
            for col in feature_cols
        }
    return stats


def apply_condition_normalisation(
    df: pd.DataFrame,
    feature_cols: list[str],
    condition_stats: dict[int, dict[str, tuple[float, float]]],
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Z-score each feature within its assigned operating condition, using train-fitted stats.

    Raw C-MAPSS sensor values cluster by operating condition for FD002/FD004 (six distinct
    conditions); without this, a model trained on raw values partly learns to recognise the
    operating condition rather than the fault, since condition swamps the fault signal.
    """
    df = df.copy()
    for col in feature_cols:
        means = df[condition_col].map(lambda c, col=col: condition_stats[int(c)][col][0])
        stds = df[condition_col].map(lambda c, col=col: condition_stats[int(c)][col][1])
        df[col] = (df[col] - means) / stds
    return df


def drop_ai4i_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the five AI4I failure-mode flags. ``machine_failure`` is defined as their logical
    OR, so keeping any of them as a model feature is textbook label leakage."""
    return df.drop(columns=[c for c in AI4I_LEAKAGE_COLUMNS if c in df.columns])
