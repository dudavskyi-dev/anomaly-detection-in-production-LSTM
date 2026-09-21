"""End-to-end C-MAPSS preprocessing: labelling, feature selection, scaling, and windowing for
one subset, in one place — so P03/P04/P05/P06 all consume identically-prepared data instead of
each re-deriving the sequence themselves.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.config import settings
from pdm.preprocessing.features import (
    apply_condition_normalisation,
    assign_condition,
    drop_constant_columns,
    fit_condition_stats,
    fit_operating_condition_clusters,
    load_constant_columns,
)
from pdm.preprocessing.labelling import add_failure_label, add_rul, add_rul_test, cap_rul
from pdm.preprocessing.scaling import StandardScaler
from pdm.preprocessing.splits import split_by_unit
from pdm.preprocessing.windowing import FeatureSpec, make_windows

MULTI_CONDITION_SUBSETS = ("FD002", "FD004")
OP_SETTING_COLUMNS = ("op_setting_1", "op_setting_2", "op_setting_3")
TARGET_COLUMNS = ("rul", "will_fail")


@dataclass
class PreparedSplit:
    windows: np.ndarray
    targets: dict[str, np.ndarray]
    index_info: list[tuple[int, int]]


@dataclass
class PreparedDataset:
    subset: str
    train: PreparedSplit
    val: PreparedSplit
    test: PreparedSplit
    feature_spec: FeatureSpec
    scaler: StandardScaler


def _label(df: pd.DataFrame, *, is_test: bool, rul_cap: int) -> pd.DataFrame:
    df = add_rul_test(df) if is_test else add_rul(df)
    df = add_failure_label(df, settings.data.failure_horizon_w)
    df = cap_rul(df, rul_cap)
    return df


def _apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]
) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols]).to_numpy()
    return df


def prepare_cmapss_subset(
    subset: str,
    processed_dir: Path,
    *,
    val_fraction: float | None = None,
    seed: int | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    pad_short_units: bool = True,
    rul_cap: int | None = None,
) -> PreparedDataset:
    """Load one C-MAPSS subset's train/test parquet, label, select features, scale, and window.

    Every fitted parameter — constant columns (from P01), KMeans operating-condition clusters,
    per-condition statistics, and the global scaler — is fit on the **train** split only and
    reused unchanged for validation and test.

    ``rul_cap`` overrides ``config.data.rul_cap`` for one call — used by P04's capped-vs-uncapped
    ablation (pass a very large cap, e.g. ``100_000``, to effectively disable capping) without
    mutating global config.
    """
    val_fraction = settings.data.val_fraction if val_fraction is None else val_fraction
    seed = settings.seed if seed is None else seed
    window_size = settings.data.window_size if window_size is None else window_size
    stride = settings.data.stride if stride is None else stride
    rul_cap = settings.data.rul_cap if rul_cap is None else rul_cap

    cmapss_dir = processed_dir / "cmapss"
    train_full = pd.read_parquet(cmapss_dir / f"train_{subset}.parquet")
    test_full = pd.read_parquet(cmapss_dir / f"test_{subset}.parquet")

    train_full = _label(train_full, is_test=False, rul_cap=rul_cap)
    test_df = _label(test_full, is_test=True, rul_cap=rul_cap)

    train_df, val_df = split_by_unit(train_full, val_fraction, seed)

    constant_columns = load_constant_columns(cmapss_dir / "sensor_profile.json", subset)
    train_df = drop_constant_columns(train_df, constant_columns)
    val_df = drop_constant_columns(val_df, constant_columns)
    test_df = drop_constant_columns(test_df, constant_columns)

    feature_cols = [c for c in train_df.columns if c.startswith(("op_setting_", "sensor_"))]
    per_condition_normalization = subset in MULTI_CONDITION_SUBSETS

    if per_condition_normalization:
        op_cols = [c for c in OP_SETTING_COLUMNS if c in train_df.columns]
        clusterer = fit_operating_condition_clusters(
            train_df, op_cols, settings.data.n_operating_conditions, seed
        )
        train_df = assign_condition(train_df, op_cols, clusterer)
        val_df = assign_condition(val_df, op_cols, clusterer)
        test_df = assign_condition(test_df, op_cols, clusterer)

        condition_stats = fit_condition_stats(train_df, feature_cols)
        train_df = apply_condition_normalisation(train_df, feature_cols, condition_stats)
        val_df = apply_condition_normalisation(val_df, feature_cols, condition_stats)
        test_df = apply_condition_normalisation(test_df, feature_cols, condition_stats)

    scaler = StandardScaler().fit(train_df, feature_cols)
    train_df = _apply_scaler(train_df, scaler, feature_cols)
    val_df = _apply_scaler(val_df, scaler, feature_cols)
    test_df = _apply_scaler(test_df, scaler, feature_cols)

    target_cols = list(TARGET_COLUMNS)
    train_windows, train_targets, train_idx = make_windows(
        train_df, feature_cols, target_cols, window_size, stride, pad_short_units=pad_short_units
    )
    val_windows, val_targets, val_idx = make_windows(
        val_df, feature_cols, target_cols, window_size, stride, pad_short_units=pad_short_units
    )
    test_windows, test_targets, test_idx = make_windows(
        test_df, feature_cols, target_cols, window_size, stride, pad_short_units=pad_short_units
    )

    feature_spec = FeatureSpec(
        feature_names=feature_cols,
        window_size=window_size,
        stride=stride,
        pad_short_units=pad_short_units,
        per_condition_normalization=per_condition_normalization,
    )

    return PreparedDataset(
        subset=subset,
        train=PreparedSplit(train_windows, train_targets, train_idx),
        val=PreparedSplit(val_windows, val_targets, val_idx),
        test=PreparedSplit(test_windows, test_targets, test_idx),
        feature_spec=feature_spec,
        scaler=scaler,
    )


def _read_via_replay(
    path: Path, *, noise_std: float, dropout_prob: float, seed: int
) -> pd.DataFrame:
    """Reads ``path`` by actually consuming ``pdm.ingestion.replay``'s row-by-row generator
    (``rate_hz=0`` disables the inter-record sleep — same code path a live replay would run,
    just without the wait) rather than a bare ``pd.read_parquet``. This is what makes P09's
    "production traffic" genuinely flow through the telemetry replay simulator (the P09 spec's
    own phrasing), not just something that could conceptually reuse it — including its
    ``noise_std``/``dropout_prob`` knobs for simulating sensor jitter/dropped readings, which
    :func:`load_raw_cmapss_windows` exposes all the way through to ``pdm drift check``.
    """
    from pdm.ingestion.replay import replay

    records = list(
        replay(path, rate_hz=0.0, noise_std=noise_std, dropout_prob=dropout_prob, seed=seed)
    )
    return pd.DataFrame.from_records(records)


def load_raw_cmapss_windows(
    subset: str,
    processed_dir: Path,
    *,
    feature_names: list[str],
    window_size: int,
    stride: int,
    split: str = "test",
    val_fraction: float | None = None,
    seed: int | None = None,
    pad_short_units: bool = True,
    rul_cap: int | None = None,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    replay_seed: int = 0,
) -> PreparedSplit:
    """Label and window one C-MAPSS subset's ``split`` **without** fitting any scaler or
    operating-condition normalisation of its own — unlike :func:`prepare_cmapss_subset`, which
    always fits and applies a scaler specific to whatever subset it loads.

    Exists for P09's drift checks (``pdm.monitoring``): "replay ``subset`` as production
    traffic" must feed raw sensor values through the model's **already-trained** bundle scaler
    (fit once, at training time, on the training subset alone) — never a scaler re-fit on the
    traffic being checked, which a real deployment could never do either, since it doesn't get
    to look at production data before deciding how to normalise it. Selects exactly
    ``feature_names`` (the serving bundle's own feature list) rather than ``subset``'s own
    constant-column-filtered list, since those two lists are not guaranteed to match across
    subsets — FD002/FD004's multi-condition sensors vary where FD001/FD003's are constant.

    Only the raw file backing the *requested* ``split`` is read via the replay simulator
    (``_read_via_replay``) — that file is the only one actually standing in for "production
    traffic" here; a request for ``split="train"``/``"val"`` never touches ``test_*.parquet``
    at all, and vice versa.
    """
    val_fraction = settings.data.val_fraction if val_fraction is None else val_fraction
    seed = settings.seed if seed is None else seed
    rul_cap = settings.data.rul_cap if rul_cap is None else rul_cap

    cmapss_dir = processed_dir / "cmapss"

    if split == "test":
        raw = _read_via_replay(
            cmapss_dir / f"test_{subset}.parquet",
            noise_std=noise_std,
            dropout_prob=dropout_prob,
            seed=replay_seed,
        )
        df = _label(raw, is_test=True, rul_cap=rul_cap)
    elif split in ("train", "val"):
        raw = _read_via_replay(
            cmapss_dir / f"train_{subset}.parquet",
            noise_std=noise_std,
            dropout_prob=dropout_prob,
            seed=replay_seed,
        )
        labelled = _label(raw, is_test=False, rul_cap=rul_cap)
        train_df, val_df = split_by_unit(labelled, val_fraction, seed)
        df = train_df if split == "train" else val_df
    else:
        raise ValueError(f"split must be one of ('train', 'val', 'test'), got {split!r}")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"{subset} is missing feature columns the bundle expects: {missing}")

    windows, targets, index_info = make_windows(
        df,
        feature_names,
        list(TARGET_COLUMNS),
        window_size,
        stride,
        pad_short_units=pad_short_units,
    )
    return PreparedSplit(windows, targets, index_info)
