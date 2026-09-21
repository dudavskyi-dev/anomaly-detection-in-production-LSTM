# P02 — Preprocessing, labelling and windowing decision log

## Why RUL is capped, and the measured effect

RUL is capped at `config.data.rul_cap = 125` cycles (spec default, standard for C-MAPSS).
Degradation isn't observable while an engine is still healthy — early-life sensor readings look
statistically identical whether an engine will fail at cycle 200 or cycle 350 — so an uncapped
linear target asks a model to predict a precise countdown from data that carries no signal for
it during the healthy plateau. Capping says "healthy" once, rather than demanding a doomed
precise prediction through the whole flat region.

**Measured, both ways** (Ridge regression, `alpha=1.0`, raw per-row sensor features after
dropping FD001's constant columns — no windowing, deliberately a *quick* baseline for this
ablation, not the real P03 baseline suite — trained on a `split_by_unit` train/val split,
`val_fraction=0.2, seed=42`, 16,779 train rows / 3,852 val rows, 17 features):

| Target | Val RMSE |
|---|---|
| Capped (`min(rul, 125)`) | **21.836** |
| Uncapped (raw `rul`, max 361 in train) | **38.876** |

Capping nearly halves the error. This matches the C-MAPSS literature's consistent finding and
is the reason every subsequent milestone reports RMSE against the capped target, not the raw one.

## Sensors dropped per subset

Reuses the exact lists P01 derived and recorded in `data/processed/cmapss/sensor_profile.json`
(`docs/decisions/P01-data.md` has the full table with measured variance). `features.py`'s
`load_constant_columns` reads that file rather than hardcoding the list — FD001 drops 7 columns
(`op_setting_3`, `sensor_1/5/10/16/18/19`), FD003 drops 6 (same minus `sensor_10`), FD002/FD004
drop none via this path (see next section for why that's correct, not a gap).

## Per-condition normalisation: verified needed for FD002/FD004, verified *not* needed for FD001

**FD001/FD003** run under a single operating condition — P01 already measured `op_setting_3`'s
std as exactly `0.0` in both, meaning there is, by construction, only one condition to
normalise within. Running the clustering machinery here would either collapse to k=1 (a no-op)
or manufacture spurious clusters from measurement noise. `pipeline.py` only applies clustering
and per-condition normalisation when `subset in ("FD002", "FD004")`.

**FD002/FD004** genuinely need it, verified empirically rather than assumed from the spec text:
silhouette score across k=2..10 (KMeans on the three `op_setting` columns, `n_init=10`,
`random_state=42`, silhouette computed on a fixed-seed 5,000-row subsample for tractability)
peaks unambiguously at k=6 for both subsets:

| k | FD002 silhouette | FD004 silhouette |
|---|---|---|
| 5 | 0.9252 | 0.9269 |
| **6** | **0.9997** | **0.9997** |
| 7 | 0.8962 | 0.8976 |

k=6 isn't just the best candidate, it's a near-perfect clustering (silhouette ≈ 1) — the six
operating conditions are extremely well separated in `(op_setting_1, op_setting_2,
op_setting_3)` space, which is exactly what "six discrete, simulated operating conditions"
predicts. `fit_operating_condition_clusters` uses `N_OPERATING_CONDITIONS = 6` in
`pipeline.py`, backed by this measurement, not copied from the spec.

**Leakage-safe fitting order**: the KMeans clusterer and the per-condition (mean, std) used for
normalisation are both fit on the **train split only** (after `split_by_unit`), then applied to
validation and test via `clusterer.predict(...)` and the stored train statistics — never
re-fit per split. Getting this backwards (e.g. clustering the whole dataset before splitting)
would leak validation/test operating-condition structure into the clusters used to preprocess
the training data.

**End-to-end pipeline run, real data** (`prepare_cmapss_subset`, default config):

| Subset | Features | Train windows | Val windows | Test windows | Per-condition norm | Wall time |
|---|---|---|---|---|---|---|
| FD001 | 17 | 14,459 | 3,272 | 10,196 | No | 0.36s |
| FD002 | 24 | 37,051 | 9,168 | 26,511 | Yes | 2.29s |

No NaNs in any split for either subset (checked directly). FD002 has more features than FD001
(24 vs 17) because none of the "constant" columns are dropped there — correctly, since with six
operating conditions those columns aren't actually constant, they're condition-dependent.

## The naive-split demonstration, with numbers

`tests/test_leakage.py::test_naive_row_split_scores_implausibly_better_than_grouped_split` uses
a small synthetic dataset (6 units × 20 cycles, each unit's sensor value offset by `unit * 10`
so units are well-separated) and a 1-nearest-neighbour regressor — deliberately a model that can
only do well by literally finding a near-duplicate row, so any leakage shows up starkly:

| Split | Val RMSE |
|---|---|
| Naive row-wise (`naive_row_split`) | **3.20** |
| Grouped by unit (`split_by_unit`) | **11.11** |

The naive split scores **3.5× better** — not because the model generalises better, but because
the same units appear on both sides, so 1-NN partly "predicts" a held-out row by matching it to
a near-identical row of the *same engine* from training. `split_by_unit` never lets that happen.
This test is kept permanently as a concrete answer to "why not just do a random split" rather
than an abstract argument.

## Bug hunting in the test-set RUL offset

No bug was actually hit implementing `add_rul_test` — P01's decision log had already flagged the
exact trap (test engines are truncated, `true_rul` from `RUL_FDxxx.txt` gives the life remaining
at truncation) before this milestone started, so the offset was implemented correctly on the
first pass and verified against the real FD001 test data (`true_rul` column already validated
in P01). What *is* new here is a permanent regression test
(`tests/test_leakage.py::test_test_set_rul_offset_is_applied_correctly`) that computes RUL both
the correct way (`add_rul_test`, using `true_rul`) and the classic-wrong way (`add_rul`, as if
the test set were train data) on the same tiny fixture, and asserts they differ by exactly the
dropped offset (50, in the fixture) — so if a future refactor ever reintroduces this bug, it
fails loudly rather than silently shipping RUL predictions that are systematically low.

## Windowing and padding

- Windows are built per unit (`groupby("unit")`), sorted by cycle, with `stride` applied within
  each unit independently — `tests/test_leakage.py::test_no_window_crosses_a_unit_boundary` and
  `tests/test_preprocessing_windowing.py::test_no_window_spans_two_units` both check the
  produced windows' feature values are a contiguous run from a single unit.
- Units shorter than `window_size` are **left-padded by repeating the unit's first observed
  row**, not zero-padded. A zero-padded window would inject a value no real sensor produces
  (0.0 is not "healthy", it's out of the sensors' physical range once scaled); repeating the
  first row treats the padded prefix as "as healthy as the earliest reading we have", which is
  the closest honest approximation available. This choice is recorded in `feature_spec.json` via
  `pad_short_units`, and no C-MAPSS unit in FD001/FD002 is actually shorter than
  `window_size=30` in practice (minimum cycles/unit was 128, per P01's profile) — this path
  matters for robustness and for other datasets more than for C-MAPSS itself.
- Targets are read from the window's **last** timestep, for every `target_col` requested in one
  pass (`rul` and `will_fail` share the same windows — no reason to window the data twice for
  two targets that are read from identical rows).

## Scaler contract

`StandardScaler.transform`/`.inverse_transform` require the input dataframe's columns to equal
the fitted `feature_names` list **exactly**, position for position — not just the same set. A
caller that passes columns in the wrong order, or with an extra/missing column, gets a
`FeatureColumnMismatchError` naming both the expected and actual column lists, rather than a
silent reindex. This is deliberately stricter than sklearn's own `StandardScaler`, which would
silently accept a reordered numpy array; the point is to make a feature-order drift between
training and (eventually) serving impossible to introduce silently.

## Ambiguities / questions flagged, not guessed

None blocked this milestone. One open item carried forward to P03: the failure-classification
target (`will_fail`) and the RUL regression target (`rul`) now share one windowing pass, which
means P03's classical baselines and P04/P05's deep models can draw both targets from the same
`PreparedDataset` without re-deriving windows — worth confirming in P03 that this doesn't
constrain anything about class-weighting or threshold selection.
