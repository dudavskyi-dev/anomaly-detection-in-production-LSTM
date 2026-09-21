"""AI4I 2020 predictive maintenance dataset loader.

The five failure-mode flags (TWF/HDF/PWF/OSF/RNF) are kept here, deliberately. The dataset's
``machine_failure`` target is defined as their logical OR, so leaving them in the feature matrix
would be textbook label leakage — P02 drops them as an explicit, tested step, not silently here.
"""

import zipfile
from pathlib import Path

import pandas as pd

from pdm.ingestion.manifest import download_file, is_cached, record_manifest_entry
from pdm.ingestion.schema import validate_ai4i

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
FALLBACK_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/601/" "ai4i+2020+predictive+maintenance+dataset.zip"
)

COLUMN_RENAME = {
    "Type": "type",
    "Air temperature": "air_temperature",
    "Process temperature": "process_temperature",
    "Rotational speed": "rotational_speed",
    "Torque": "torque",
    "Tool wear": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}
ID_COLUMNS = ("UDI", "Product ID")


def _fetch_via_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=601)
    return pd.concat([dataset.data.features, dataset.data.targets], axis=1)


def _fetch_via_csv_fallback(raw_dir: Path, manifest_path: Path) -> pd.DataFrame:
    zip_path = raw_dir / "ai4i2020.zip"
    download_file(FALLBACK_ZIP_URL, zip_path, source_page=SOURCE_PAGE, manifest_path=manifest_path)
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            df = pd.read_csv(fh)
    return df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])


def fetch_ai4i(raw_dir: Path, manifest_path: Path) -> pd.DataFrame:
    """Fetch AI4I, preferring the ``ucimlrepo`` API and falling back to a direct CSV download.

    Both paths are real data sources — this is not a synthetic-data fallback. Only raises if
    both fail. If a manifest-verified copy is already cached locally, no network call is made
    at all (``ucimlrepo`` has no ETag/hash of its own to check against, so the cache is keyed on
    our own re-exported CSV instead).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "ai4i2020.csv"

    if is_cached(raw_csv, manifest_path=manifest_path):
        return pd.read_csv(raw_csv)

    try:
        df = _fetch_via_ucimlrepo()
        source = "ucimlrepo:601"
    except Exception as primary_exc:
        try:
            df = _fetch_via_csv_fallback(raw_dir, manifest_path)
            source = FALLBACK_ZIP_URL
        except Exception as fallback_exc:
            raise RuntimeError(
                f"AI4I fetch failed via ucimlrepo ({primary_exc!r}) and via direct CSV download "
                f"({fallback_exc!r}). No synthetic fallback is generated. See {SOURCE_PAGE} "
                "for a manual download."
            ) from fallback_exc

    df.to_csv(raw_csv, index=False)
    record_manifest_entry(raw_csv, source=source, manifest_path=manifest_path)
    return df


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_RENAME)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


def profile_ai4i(df: pd.DataFrame) -> dict:
    n = len(df)
    n_failures = int(df["machine_failure"].sum())
    return {
        "rows": n,
        "failures": n_failures,
        "failure_rate": n_failures / n if n else 0.0,
        "columns": df.columns.tolist(),
    }


def load_and_convert_ai4i(raw_dir: Path, processed_dir: Path, manifest_path: Path) -> dict:
    df = fetch_ai4i(raw_dir, manifest_path)
    df = normalise_columns(df)
    df = validate_ai4i(df)

    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / "ai4i.parquet", index=False)

    return profile_ai4i(df)
