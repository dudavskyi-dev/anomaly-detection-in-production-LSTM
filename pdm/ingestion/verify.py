"""Re-validate already-converted parquet files against their schemas without re-downloading.

Backs ``pdm data verify`` — a way to confirm the local `data/processed/` tree is intact and
schema-valid after, say, a partial `make data` run or manual tampering.
"""

from pathlib import Path

import pandas as pd

from pdm.ingestion.cmapss import SUBSETS
from pdm.ingestion.schema import validate_ai4i, validate_cmapss, validate_nab


def verify_cmapss(processed_dir: Path) -> dict:
    cmapss_dir = processed_dir / "cmapss"
    report: dict[str, dict] = {}
    for subset in SUBSETS:
        train_path = cmapss_dir / f"train_{subset}.parquet"
        test_path = cmapss_dir / f"test_{subset}.parquet"
        if not train_path.exists() or not test_path.exists():
            report[subset] = {"status": "missing"}
            continue
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path).drop(columns=["true_rul"])
        validate_cmapss(train_df, subset=subset, split="train")
        validate_cmapss(test_df, subset=subset, split="test")
        report[subset] = {"status": "ok", "train_rows": len(train_df), "test_rows": len(test_df)}
    return report


def verify_ai4i(processed_dir: Path) -> dict:
    path = processed_dir / "ai4i.parquet"
    if not path.exists():
        return {"status": "missing"}
    df = pd.read_parquet(path)
    validate_ai4i(df)
    return {"status": "ok", "rows": len(df)}


def verify_nab(processed_dir: Path) -> dict:
    path = processed_dir / "machine_temperature_system_failure.parquet"
    if not path.exists():
        return {"status": "missing"}
    df = pd.read_parquet(path)
    validate_nab(df)
    return {"status": "ok", "rows": len(df)}


def verify_all(processed_dir: Path) -> dict:
    report = {
        "cmapss": verify_cmapss(processed_dir),
        "ai4i": verify_ai4i(processed_dir),
        "nab": verify_nab(processed_dir),
    }
    missing = [
        name
        for name, status in {
            **{f"cmapss/{k}": v for k, v in report["cmapss"].items()},
            "ai4i": report["ai4i"],
            "nab": report["nab"],
        }.items()
        if status.get("status") == "missing"
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing processed datasets, run `pdm data download` first: {missing}"
        )
    return report
