"""Per-unit RUL labelling: raw RUL, piecewise-linear capping, and the failure-within-W label.

Call order matters: :func:`add_rul` (or :func:`add_rul_test` for C-MAPSS test data) must run
before :func:`add_failure_label`, which in turn must run before :func:`cap_rul` — the failure
label is defined against the *true* remaining life, not a capped one, even though in practice
``rul_cap`` (125) comfortably exceeds ``failure_horizon_w`` (30) so the cap never actually
changes which rows are labelled as about to fail.
"""

import pandas as pd


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL = last observed cycle for the unit − current cycle.

    Correct for **train** data, where every unit runs to failure so its last row *is* the
    failure point. Using this on C-MAPSS test data (truncated before failure) silently produces
    RUL values that are too low — see :func:`add_rul_test`.
    """
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = last_cycle - df["cycle"]
    return df


def add_rul_test(df: pd.DataFrame) -> pd.DataFrame:
    """RUL for C-MAPSS **test** data, offset by the true RUL at the truncation point.

    Test trajectories are cut off before failure; ``true_rul`` (attached in P01 from
    ``RUL_FDxxx.txt``) is how much life remained when the recording stopped. Reusing
    :func:`add_rul` here — computing RUL as if the last recorded cycle *were* the failure —
    is the single most common C-MAPSS bug: it silently understates every test RUL by exactly
    ``true_rul``. ``tests/test_leakage.py`` asserts the two differ by exactly that amount.
    """
    if "true_rul" not in df.columns:
        raise ValueError(
            "add_rul_test requires a `true_rul` column (attached by P01's C-MAPSS test loader)"
        )
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (last_cycle - df["cycle"]) + df["true_rul"]
    return df


def cap_rul(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Piecewise-linear RUL: clip at ``cap`` cycles.

    Degradation isn't observable while an engine is still healthy, so an uncapped linear target
    asks the model to predict a precise countdown from cycle 1, when the sensors carry no signal
    that would let it. Capping says "healthy" once, instead of pretending to count down through
    it — see ``docs/decisions/P02-preprocessing.md`` for the measured effect on RMSE.
    """
    df = df.copy()
    df["rul"] = df["rul"].clip(upper=cap)
    return df


def add_failure_label(df: pd.DataFrame, horizon_w: int) -> pd.DataFrame:
    """Binary label: will this unit fail within ``horizon_w`` cycles, from the current ``rul``.

    Must be called with the *uncapped* ``rul`` already present (i.e. before :func:`cap_rul`).
    """
    df = df.copy()
    df["will_fail"] = (df["rul"] <= horizon_w).astype(int)
    return df
