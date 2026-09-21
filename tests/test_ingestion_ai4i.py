"""AI4I column normalisation and schema validation. Fetching (ucimlrepo / CSV fallback) is
network-bound and deliberately not exercised here — these tests work purely on a small
committed CSV fixture shaped like ucimlrepo's combined features+targets frame."""

from pathlib import Path

import pandas as pd
import pytest

from pdm.ingestion.ai4i import normalise_columns, profile_ai4i
from pdm.ingestion.schema import validate_ai4i

pytestmark = pytest.mark.fast

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "ai4i_sample.csv")


def test_normalise_columns_snake_cases_everything():
    df = normalise_columns(_load_fixture())
    assert list(df.columns) == [
        "type",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        "machine_failure",
        "twf",
        "hdf",
        "pwf",
        "osf",
        "rnf",
    ]


def test_failure_mode_flags_are_not_dropped_here():
    # P01 keeps them; P02 drops them as a deliberate, tested step.
    df = normalise_columns(_load_fixture())
    assert {"twf", "hdf", "pwf", "osf", "rnf"}.issubset(df.columns)


def test_validate_ai4i_accepts_well_formed_fixture():
    df = normalise_columns(_load_fixture())
    validated = validate_ai4i(df)
    assert len(validated) == 12


def test_validate_ai4i_rejects_unknown_type_category():
    df = normalise_columns(_load_fixture())
    df.loc[0, "type"] = "X"
    with pytest.raises(ValueError, match="schema validation"):
        validate_ai4i(df)


def test_validate_ai4i_rejects_non_binary_failure_flag():
    df = normalise_columns(_load_fixture())
    df.loc[0, "machine_failure"] = 2
    with pytest.raises(ValueError, match="schema validation"):
        validate_ai4i(df)


def test_profile_ai4i_counts_failures_from_fixture():
    df = normalise_columns(_load_fixture())
    profile = profile_ai4i(df)
    assert profile["rows"] == 12
    assert profile["failures"] == 2
    assert profile["failure_rate"] == pytest.approx(2 / 12)
