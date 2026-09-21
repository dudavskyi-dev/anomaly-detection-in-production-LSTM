"""Pandera schemas for every raw dataset. Validation runs on every load and fails loudly,
listing the offending columns and rows rather than silently coercing or dropping data.
"""

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

N_CMAPSS_SENSORS = 21


def cmapss_schema() -> DataFrameSchema:
    """Schema for a parsed C-MAPSS train/test dataframe: unit, cycle, 3 settings, 21 sensors."""
    columns = {
        "unit": Column(int, Check.gt(0)),
        "cycle": Column(int, Check.gt(0)),
        "op_setting_1": Column(float, nullable=False),
        "op_setting_2": Column(float, nullable=False),
        "op_setting_3": Column(float, nullable=False),
    }
    for i in range(1, N_CMAPSS_SENSORS + 1):
        columns[f"sensor_{i}"] = Column(float, nullable=False)
    return DataFrameSchema(columns, strict=True, coerce=True)


def _check_cycles_start_at_one_and_are_contiguous(df: pd.DataFrame) -> bool:
    for _, group in df.groupby("unit"):
        cycles = group["cycle"].to_numpy()
        expected = range(1, len(cycles) + 1)
        if not (cycles == list(expected)).all():
            return False
    return True


def validate_cmapss(df: pd.DataFrame, *, subset: str, split: str) -> pd.DataFrame:
    """Validate a parsed C-MAPSS dataframe, raising a readable error naming subset/split."""
    try:
        validated = cmapss_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(
            f"C-MAPSS {subset} {split} failed schema validation:\n{exc.failure_cases}"
        ) from exc
    if not _check_cycles_start_at_one_and_are_contiguous(validated):
        raise ValueError(
            f"C-MAPSS {subset} {split}: at least one unit has non-contiguous cycle numbers "
            "(expected 1..N per unit with no gaps)."
        )
    return validated


def ai4i_schema() -> DataFrameSchema:
    """Schema for the AI4I dataset. Failure-mode flags are validated but not yet dropped —
    P02 removes them as a deliberate, tested step (the target is defined as their OR)."""
    return DataFrameSchema(
        {
            "type": Column(str, Check.isin(["L", "M", "H"])),
            "air_temperature": Column(float),
            "process_temperature": Column(float),
            "rotational_speed": Column(float),
            "torque": Column(float),
            "tool_wear": Column(float),
            "machine_failure": Column(int, Check.isin([0, 1])),
            "twf": Column(int, Check.isin([0, 1])),
            "hdf": Column(int, Check.isin([0, 1])),
            "pwf": Column(int, Check.isin([0, 1])),
            "osf": Column(int, Check.isin([0, 1])),
            "rnf": Column(int, Check.isin([0, 1])),
        },
        strict=True,
        coerce=True,
    )


def validate_ai4i(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return ai4i_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(f"AI4I failed schema validation:\n{exc.failure_cases}") from exc


def nab_schema() -> DataFrameSchema:
    """Schema for the NAB univariate series: timestamp, value, and the derived anomaly label."""
    return DataFrameSchema(
        {
            "timestamp": Column("datetime64[ns]"),
            "value": Column(float),
            "is_anomaly": Column(bool),
        },
        checks=Check(
            lambda df: df["timestamp"].is_monotonic_increasing,
            error="timestamps must be sorted ascending",
        ),
        strict=True,
        coerce=True,
    )


def validate_nab(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return nab_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(f"NAB failed schema validation:\n{exc.failure_cases}") from exc
