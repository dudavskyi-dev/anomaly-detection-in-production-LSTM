"""Numenta Anomaly Benchmark loader: the ``realKnownCause/machine_temperature_system_failure``
series plus its labelled anomaly windows, used as external, real-world anomaly-detection
validation (C-MAPSS is simulated; this machine really failed).
"""

import json
from pathlib import Path

import pandas as pd

from pdm.ingestion.manifest import download_file
from pdm.ingestion.schema import validate_nab

SOURCE_PAGE = "https://github.com/numenta/NAB"
SERIES_KEY = "realKnownCause/machine_temperature_system_failure.csv"
CSV_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/data/"
    "realKnownCause/machine_temperature_system_failure.csv"
)
WINDOWS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"


def fetch_nab(raw_dir: Path, manifest_path: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "machine_temperature_system_failure.csv"
    windows_path = raw_dir / "combined_windows.json"
    download_file(CSV_URL, csv_path, source_page=SOURCE_PAGE, manifest_path=manifest_path)
    download_file(WINDOWS_URL, windows_path, source_page=SOURCE_PAGE, manifest_path=manifest_path)
    return csv_path, windows_path


def load_windows(windows_path: Path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    all_windows = json.loads(windows_path.read_text(encoding="utf-8"))
    series_windows = all_windows.get(SERIES_KEY, [])
    return [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in series_windows]


def parse_nab_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def label_anomalies(
    df: pd.DataFrame, windows: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> pd.DataFrame:
    is_anomaly = pd.Series(False, index=df.index)
    for start, end in windows:
        is_anomaly |= (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df = df.copy()
    df["is_anomaly"] = is_anomaly
    return df


def profile_nab(df: pd.DataFrame, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> dict:
    return {
        "rows": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "n_windows": len(windows),
        "n_anomalous_rows": int(df["is_anomaly"].sum()),
    }


def load_and_convert_nab(raw_dir: Path, processed_dir: Path, manifest_path: Path) -> dict:
    csv_path, windows_path = fetch_nab(raw_dir, manifest_path)
    df = parse_nab_csv(csv_path)
    windows = load_windows(windows_path)
    df = label_anomalies(df, windows)
    df = validate_nab(df)

    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_dir / "machine_temperature_system_failure.parquet", index=False)
    (processed_dir / "machine_temperature_windows.json").write_text(
        json.dumps([[str(s), str(e)] for s, e in windows], indent=2), encoding="utf-8"
    )

    return profile_nab(df, windows)
