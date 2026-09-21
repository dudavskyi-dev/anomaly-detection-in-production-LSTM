"""C-MAPSS text parsing, schema validation, and the test-set RUL offset — the single most
common bug in C-MAPSS code, per the project spec."""

from pathlib import Path

import pytest

from pdm.ingestion.cmapss import (
    COLUMNS,
    N_SENSORS,
    attach_true_test_rul,
    parse_rul_text,
    parse_run_text,
)
from pdm.ingestion.schema import validate_cmapss

pytestmark = pytest.mark.fast

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_run_text_shape_and_columns():
    text = (FIXTURES / "cmapss_train_sample.txt").read_text()
    df = parse_run_text(text)

    assert list(df.columns) == COLUMNS
    assert len(df.columns) == 5 + N_SENSORS
    assert len(df) == 12
    assert set(df["unit"].unique()) == {1, 2}
    assert df.groupby("unit").size().tolist() == [6, 6]


def test_parse_run_text_handles_trailing_whitespace():
    # every real C-MAPSS line ends with trailing whitespace; a naive splitter turns that into
    # a phantom 27th column of NaNs.
    text = (FIXTURES / "cmapss_train_sample.txt").read_text()
    assert text.splitlines()[0].endswith("  ")
    df = parse_run_text(text)
    assert df.shape[1] == len(COLUMNS)


def test_validate_cmapss_accepts_well_formed_fixture():
    df = parse_run_text((FIXTURES / "cmapss_train_sample.txt").read_text())
    validated = validate_cmapss(df, subset="FIXTURE", split="train")
    assert len(validated) == len(df)


def test_validate_cmapss_rejects_gapped_cycles():
    df = parse_run_text((FIXTURES / "cmapss_train_sample.txt").read_text())
    df = df.drop(df.index[2])  # removes unit 1's cycle 3, leaving a gap: 1, 2, 4, 5, 6
    with pytest.raises(ValueError, match="non-contiguous"):
        validate_cmapss(df, subset="FIXTURE", split="train")


def test_validate_cmapss_rejects_non_numeric_sensor():
    df = parse_run_text((FIXTURES / "cmapss_train_sample.txt").read_text())
    df["sensor_1"] = "not-a-number"
    with pytest.raises(ValueError, match="schema validation"):
        validate_cmapss(df, subset="FIXTURE", split="train")


def test_validate_cmapss_rejects_non_positive_cycle():
    df = parse_run_text((FIXTURES / "cmapss_train_sample.txt").read_text())
    df.loc[0, "cycle"] = 0
    with pytest.raises(ValueError, match="schema validation"):
        validate_cmapss(df, subset="FIXTURE", split="train")


def test_parse_rul_text():
    rul_values = parse_rul_text((FIXTURES / "cmapss_rul_sample.txt").read_text())
    assert rul_values == [15, 21]


def test_attach_true_test_rul_broadcasts_per_unit():
    test_df = parse_run_text((FIXTURES / "cmapss_test_sample.txt").read_text())
    rul_values = parse_rul_text((FIXTURES / "cmapss_rul_sample.txt").read_text())

    out = attach_true_test_rul(test_df, rul_values)

    assert out.loc[out["unit"] == 1, "true_rul"].unique().tolist() == [15]
    assert out.loc[out["unit"] == 2, "true_rul"].unique().tolist() == [21]


def test_attach_true_test_rul_rejects_length_mismatch():
    test_df = parse_run_text((FIXTURES / "cmapss_test_sample.txt").read_text())
    with pytest.raises(ValueError, match="RUL file has"):
        attach_true_test_rul(test_df, [1, 2, 3])
