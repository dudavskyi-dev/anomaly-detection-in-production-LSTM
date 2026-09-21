"""NAB CSV parsing, window-based anomaly labelling, and schema validation on a tiny fixture
series (12 five-minute readings with one labelled anomaly window)."""

from pathlib import Path

import pytest

from pdm.ingestion.nab import label_anomalies, load_windows, parse_nab_csv, profile_nab
from pdm.ingestion.schema import validate_nab

pytestmark = pytest.mark.fast

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_sorts_by_timestamp_and_keeps_all_rows():
    df = parse_nab_csv(FIXTURES / "nab_sample.csv")
    assert len(df) == 12
    assert df["timestamp"].is_monotonic_increasing


def test_label_anomalies_marks_exactly_the_windowed_rows():
    df = parse_nab_csv(FIXTURES / "nab_sample.csv")
    windows = load_windows(FIXTURES / "nab_windows_sample.json")
    labelled = label_anomalies(df, windows)

    # fixture window is 21:50:00 .. 22:05:00 inclusive, at 5-minute spacing -> 4 rows
    assert labelled["is_anomaly"].sum() == 4
    anomalous_times = (
        labelled.loc[labelled["is_anomaly"], "timestamp"].dt.strftime("%H:%M").tolist()
    )
    assert anomalous_times == ["21:50", "21:55", "22:00", "22:05"]


def test_validate_nab_accepts_well_formed_fixture():
    df = parse_nab_csv(FIXTURES / "nab_sample.csv")
    windows = load_windows(FIXTURES / "nab_windows_sample.json")
    validated = validate_nab(label_anomalies(df, windows))
    assert len(validated) == 12


def test_validate_nab_rejects_unsorted_timestamps():
    df = parse_nab_csv(FIXTURES / "nab_sample.csv")
    windows = load_windows(FIXTURES / "nab_windows_sample.json")
    labelled = label_anomalies(df, windows)
    shuffled = labelled.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="schema validation"):
        validate_nab(shuffled)


def test_profile_nab_reports_window_and_row_counts():
    df = parse_nab_csv(FIXTURES / "nab_sample.csv")
    windows = load_windows(FIXTURES / "nab_windows_sample.json")
    labelled = label_anomalies(df, windows)
    profile = profile_nab(labelled, windows)
    assert profile["rows"] == 12
    assert profile["n_windows"] == 1
    assert profile["n_anomalous_rows"] == 4
