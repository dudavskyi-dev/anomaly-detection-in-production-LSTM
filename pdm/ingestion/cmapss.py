"""NASA C-MAPSS turbofan degradation loader: download, parse, profile, and convert to parquet.

The distributed archive is a zip containing a single nested zip (``CMAPSSData.zip``) holding
flat whitespace-separated text files with no header row and a trailing-whitespace-terminated
line (an extra delimiter at end of line that a naive fixed-width or single-space split would
turn into a phantom 27th column). ``pandas.read_csv`` with a ``\\s+`` regex separator handles
this correctly and is used here rather than a hand-rolled splitter.
"""

import io
import json
import zipfile
from io import StringIO
from pathlib import Path

import pandas as pd

from pdm.config import settings
from pdm.ingestion.manifest import download_file
from pdm.ingestion.schema import validate_cmapss

SOURCE_PAGE = (
    "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/"
    "pcoe-data-set-repository/"
)
DOWNLOAD_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
N_SENSORS = 21
COLUMNS = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
    f"sensor_{i}" for i in range(1, N_SENSORS + 1)
]


def download_cmapss(raw_dir: Path, manifest_path: Path) -> Path:
    """Download the outer archive (cached via the manifest) and return its local path."""
    zip_path = raw_dir / "cmapss.zip"
    download_file(DOWNLOAD_URL, zip_path, source_page=SOURCE_PAGE, manifest_path=manifest_path)
    return zip_path


def _read_inner_zip(outer_zip_path: Path) -> zipfile.ZipFile:
    with zipfile.ZipFile(outer_zip_path) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        inner_bytes = outer.read(inner_name)
    return zipfile.ZipFile(io.BytesIO(inner_bytes))


def extract_subset_texts(outer_zip_path: Path, subset: str) -> dict[str, str]:
    """Return the raw text of the train/test/RUL files for one subset (e.g. ``"FD001"``)."""
    with _read_inner_zip(outer_zip_path) as inner:
        return {
            "train": inner.read(f"train_{subset}.txt").decode("utf-8"),
            "test": inner.read(f"test_{subset}.txt").decode("utf-8"),
            "rul": inner.read(f"RUL_{subset}.txt").decode("utf-8"),
        }


def parse_run_text(text: str) -> pd.DataFrame:
    """Parse a whitespace-separated C-MAPSS train/test text blob into a named dataframe."""
    df = pd.read_csv(StringIO(text), sep=r"\s+", header=None, names=COLUMNS, engine="python")
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def parse_rul_text(text: str) -> list[int]:
    """Parse a ``RUL_FDxxx.txt`` blob: one integer per test engine, in unit order."""
    return [int(line) for line in text.split()]


def attach_true_test_rul(test_df: pd.DataFrame, rul_values: list[int]) -> pd.DataFrame:
    """Broadcast each test engine's true RUL (at its truncation point) across all of its rows.

    Test trajectories are truncated before failure; ``RUL_FDxxx.txt`` gives the RUL at that
    truncation point, in ascending unit-id order. Getting this offset wrong is the single most
    common C-MAPSS bug (P02 uses this column to compute the correct piecewise-linear RUL).
    """
    unit_ids = sorted(test_df["unit"].unique())
    if len(unit_ids) != len(rul_values):
        raise ValueError(
            f"RUL file has {len(rul_values)} values but test set has {len(unit_ids)} units"
        )
    rul_map = dict(zip(unit_ids, rul_values, strict=True))
    df = test_df.copy()
    df["true_rul"] = df["unit"].map(rul_map)
    return df


def profile_subset(train_df: pd.DataFrame, test_df: pd.DataFrame, subset: str) -> dict:
    """Row/unit counts, cycles-per-unit stats, and constant-sensor detection for one subset."""
    threshold = settings.data.constant_sensor_std_threshold
    feature_cols = [c for c in train_df.columns if c.startswith(("op_setting_", "sensor_"))]
    variances = {col: float(train_df[col].std()) for col in feature_cols}
    constant_columns = [col for col, std in variances.items() if std < threshold]

    cycles_per_unit = train_df.groupby("unit").size()
    return {
        "subset": subset,
        "train_rows": int(len(train_df)),
        "train_units": int(train_df["unit"].nunique()),
        "test_rows": int(len(test_df)),
        "test_units": int(test_df["unit"].nunique()),
        "cycles_per_unit_min": int(cycles_per_unit.min()),
        "cycles_per_unit_median": float(cycles_per_unit.median()),
        "cycles_per_unit_max": int(cycles_per_unit.max()),
        "feature_std": variances,
        "constant_columns": constant_columns,
        "constant_column_threshold": threshold,
    }


def load_and_convert_cmapss(
    raw_dir: Path, processed_dir: Path, manifest_path: Path, subsets: tuple[str, ...] = SUBSETS
) -> dict[str, dict]:
    """Download (if needed), parse, validate, and write parquet for every requested subset.

    Returns a per-subset profile dict, also written to
    ``<processed_dir>/cmapss/sensor_profile.json`` for P02 to consume without re-deriving it.
    """
    zip_path = download_cmapss(raw_dir, manifest_path)
    out_dir = processed_dir / "cmapss"
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles: dict[str, dict] = {}
    for subset in subsets:
        texts = extract_subset_texts(zip_path, subset)
        train_df = parse_run_text(texts["train"])
        test_df = parse_run_text(texts["test"])
        rul_values = parse_rul_text(texts["rul"])

        train_df = validate_cmapss(train_df, subset=subset, split="train")
        test_df = validate_cmapss(test_df, subset=subset, split="test")
        test_df = attach_true_test_rul(test_df, rul_values)

        train_df.to_parquet(out_dir / f"train_{subset}.parquet", index=False)
        test_df.to_parquet(out_dir / f"test_{subset}.parquet", index=False)

        profiles[subset] = profile_subset(train_df, test_df, subset)

    (out_dir / "sensor_profile.json").write_text(
        json.dumps(profiles, indent=2, sort_keys=True), encoding="utf-8"
    )
    return profiles
