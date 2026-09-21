"""RUL labelling: capping, failure-within-W, and the C-MAPSS test-set RUL offset."""

import pandas as pd
import pytest

from pdm.preprocessing.labelling import add_failure_label, add_rul, add_rul_test, cap_rul

pytestmark = pytest.mark.fast


def _make_unit(unit: int, n_cycles: int) -> pd.DataFrame:
    return pd.DataFrame({"unit": unit, "cycle": range(1, n_cycles + 1)})


def test_add_rul_counts_down_to_zero_at_last_cycle():
    df = pd.concat([_make_unit(1, 5), _make_unit(2, 3)], ignore_index=True)
    out = add_rul(df)
    assert out.loc[out["unit"] == 1, "rul"].tolist() == [4, 3, 2, 1, 0]
    assert out.loc[out["unit"] == 2, "rul"].tolist() == [2, 1, 0]


def test_add_rul_test_offsets_by_true_rul():
    df = _make_unit(1, 5)
    df["true_rul"] = 20
    out = add_rul_test(df)
    assert out["rul"].tolist() == [24, 23, 22, 21, 20]


def test_add_rul_test_requires_true_rul_column():
    df = _make_unit(1, 5)
    with pytest.raises(ValueError, match="true_rul"):
        add_rul_test(df)


def test_cap_rul_clips_high_values_only():
    df = pd.DataFrame({"rul": [200, 150, 125, 50, 0]})
    out = cap_rul(df, cap=125)
    assert out["rul"].tolist() == [125, 125, 125, 50, 0]


def test_add_failure_label_uses_horizon():
    df = pd.DataFrame({"rul": [40, 30, 29, 0]})
    out = add_failure_label(df, horizon_w=30)
    assert out["will_fail"].tolist() == [0, 1, 1, 1]


def test_capping_does_not_change_the_failure_label():
    # cap (125) exceeds horizon (30), so the label must be identical regardless of cap order.
    df = _make_unit(1, 200)
    before_cap = add_failure_label(add_rul(df.copy()), horizon_w=30)
    after_cap = add_failure_label(cap_rul(add_rul(df.copy()), cap=125), horizon_w=30)
    assert before_cap["will_fail"].tolist() == after_cap["will_fail"].tolist()
