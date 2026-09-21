# P07 — MLflow tracking and registry decision log

## Params vs metrics vs tags

- **Params** (`pdm.tracking.mlflow_client.log_params`): every leaf value of the *config* that
  produced this run — the whole `settings.model_dump()` tree, flattened to dotted keys
  (`model.dropout`, `training.learning_rate`, ...), plus a few call-site values that aren't in
  `settings` at all but are still fixed inputs to the run (`subset`, `n_features`). The test for
  "is this a param": **would you need this value to run the exact same training again?** If
  yes, it's a param, logged once at the start and never overwritten.
- **Metrics** (`log_metrics`/`log_epoch_history`): every *measured* number — per-epoch
  train/val loss and score (logged once, after training finishes, from the already-recorded
  `EpochStats` history — no epoch-loop instrumentation needed) and the final `val_*`/`test_*`
  numbers. The test: **is this an output, and could a future run legitimately produce a
  different value for it?** If yes, it's a metric, not a param — this is also why "seed" is a
  **param** (an input you chose) while "best_epoch" is folded into the metrics history rather
  than treated as a separate param, since it's discovered by training, not chosen before it.
- **Tags** (`start_run`'s fixed tag set: `git_sha`, `git_dirty`, `dataset_hash`, `framework`,
  `seed`, plus per-call-site tags like `task`/`subset`/`model`): **identity and provenance
  metadata that answers "what code, what data, which variant" without being part of the
  model's own configuration.** `seed` appears as *both* a tag and (implicitly, via the params
  block) reproducible input — tagged specifically because "show me every run at seed 3" is a
  query someone will actually run, and MLflow's UI/search filters on tags far more naturally
  than on a param buried in a flattened dict.

## Tracking server vs artifact store vs registry, in my own words

- **The tracking server** is the thing you talk to when you call `start_run`/`log_params`/
  `log_metrics` — it's a database of *runs*: each run's params, metrics (with full history, not
  just the final value), tags, and start/end time. It answers "what happened."
- **The artifact store** is where the *files* a run produced live — the bundle directory,
  training-curve plots, anything passed to `log_artifact(s)`. In this project's default local
  setup the tracking server's metadata and the artifact store are both just directories under
  `./mlruns` (`file:./mlruns`), which makes the distinction easy to miss — but they're
  conceptually separate, and a real deployment typically splits them (Postgres for tracking
  metadata, S3/GCS for artifacts) because they have completely different access patterns (many
  small structured writes vs occasional large blob writes).
- **The registry** is neither of those — it's a layer *on top* of the tracking server that
  gives specific runs' artifacts a stable name and version number (`pdm-sentinel-rul` version 3)
  independent of which run produced them, plus a stage (`None`/`Staging`/`Production`/
  `Archived`) that changes over time without the underlying run or artifact ever changing. A
  run is immutable once logged; a registered model version's *stage* is the one thing about
  this whole system that's designed to be mutated after the fact, and only through the gate.

## The promotion gate

`pdm.tracking.promotion.evaluate_gate` (pure, no MLflow calls — `tests/test_promotion.py`
exercises it directly) compares a candidate's `primary_metric` against the current Production
version's value for the same metric, requiring a **relative margin** (`settings.tracking.
promotion_margin`, default `0.02` = 2%) in the correct direction for whichever metric is being
gated (lower-is-better for `rmse`/`mae`/`nasa_score`/`brier`, higher-is-better for everything
else — `is_higher_better` strips a leading `val_`/`test_`/`train_` split prefix first, since
every metric this project actually logs carries one). A first-ever registration (no Production
version exists yet) always passes — there's nothing to beat, and refusing it would make the
registry permanently empty.

`pdm.tracking.promotion.promote_to_production` is the **only** function anywhere in this
codebase that ever passes `stage="Production"` to MLflow — `pdm.tracking.registry` exposes
`transition_to_staging`/`archive` for every other stage, deliberately not a raw
"transition to Production" helper. `tests/test_registry.py::
test_no_function_other_than_promote_to_production_sets_the_production_stage` checks this
structurally (greps the registry module's own source for the string), not just by convention.

**A real floating-point bug this design caught immediately**: the first version of
`evaluate_gate` compared `candidate_value >= production_value * (1 + margin)` with no
tolerance. `0.80 * 1.02` evaluates to `0.8160000000000001` in IEEE-754 double precision, not the
mathematically exact `0.816` — so a candidate whose metric is computed to land *exactly* on the
required threshold (a real possibility, not a contrived one: two runs can produce identical
metrics from identical data under identical determinism) would be wrongly refused depending on
floating-point summation order. Fixed with a `1e-9` relative tolerance on the comparison itself
(`tests/test_promotion.py::test_higher_is_better_metric_requires_at_least_the_margin_improvement`
pins the exact boundary value, `0.816` against a `0.80` incumbent, to prevent a regression).

**The default primary metric had to change to match the logging convention, not the other way
round.** `settings.tracking.promotion_primary_metric` originally defaulted to `"rmse"`, but
every training path logs `val_rmse`/`test_rmse` — never a bare `rmse` — so the out-of-the-box
gate would silently find no metric on any real run and refuse every promotion with a confusing
"candidate run has no logged metric" error. Changed the default to `"test_rmse"` rather than
also logging an unprefixed alias, since the split prefix is genuinely informative (which split
produced this number matters) and duplicating every metric under two names would be worse.

**A second real bug, caught by actually running `pdm registry compare` against the real
baseline runs from this session, not just by unit tests with hand-picked metric names**:
`is_higher_better("test_rmse")` fell through to the default "assume higher is better" branch
before the prefix-stripping fix above existed, because the lookup table only contained the bare
name `"rmse"`. The practical effect: `pdm registry compare --metric test_rmse` ranked the
**worst** RUL models first (`dummy_mean`, RMSE ~37-38) instead of the best
(`random_forest`, RMSE ~15). Caught by literally running the command against real data and
noticing the ranking looked wrong — the exact kind of thing a unit test with synthetic values
wouldn't have surfaced, since I'd only tested `is_higher_better("rmse")`, never
`is_higher_better("test_rmse")`, until the CLI's actual output looked backwards.

## Worked example: two runs, why one promoted and the other didn't

Both runs below are real `pdm train` invocations against FD001 logged in this session's MLflow
store, not constructed numbers.

**Run 1** — `e1bd32d9a5b549fe98568f2fa1349e58`, default full training budget (`max_epochs=100`,
`patience=10`), seed 42: converged at epoch 16 of 26, **test RMSE 16.373**.

```
pdm registry promote --run-id e1bd32d9a5b549fe98568f2fa1349e58
# -> Registered pdm-sentinel-rul version 1 ... Promoted ... "no existing Production model to beat"
```

Registered as `pdm-sentinel-rul` version 1 and promoted immediately — there was no existing
Production version, so the gate's "nothing to beat" branch passes it automatically. Version 1
is now Production.

**Run 2** — `d8311b8ece574747a7d55d9c34c92705`, deliberately crippled budget
(`PDM__TRAINING__MAX_EPOCHS=3 PDM__TRAINING__EARLY_STOPPING_PATIENCE=3`, seed 43): only 3
epochs, val RMSE still falling every epoch (never converged) — **test RMSE 74.291**, more than
4x worse.

```
pdm registry promote --run-id d8311b8ece574747a7d55d9c34c92705
# -> Registered pdm-sentinel-rul version 2 ...
# -> Promotion REFUSED: candidate test_rmse=74.2909 <= required 16.0457
#    (production test_rmse=16.3732, margin=2.0%, higher_is_better=False) -> REFUSED
# -> exit code 1
```

Version 2 registered successfully (registration itself doesn't gate anything — only the
Production *transition* does) and was moved to Staging, but `promote_to_production` computed
the required threshold as `16.3732 * (1 - 0.02) = 16.0457` and refused because `74.2909` is
nowhere near `<= 16.0457`. Checked directly afterward: `pdm-sentinel-rul`'s Production stage
still points at version 1 / run `e1bd...`, and version 2 sits in Staging, exactly where a
refused-but-registered candidate should land — visible for inspection, not silently discarded,
but nowhere near serving traffic.

## The reproducibility test

`tests/test_reproducibility.py::test_retraining_from_logged_params_reproduces_the_logged_metric`
logs a small training run's params, reloads them **from the MLflow run itself** (not from the
Python variables already in scope — the whole point is to check nothing needed for
reproduction was left unlogged), retrains from the reloaded values, and asserts the metric
matches within `rel=1e-6`. It passed on the first real attempt with the current logging setup —
which is itself informative: it means `pdm.models.torch.train.set_full_determinism` (called
before model construction, per the lesson `docs/decisions/P04-pytorch.md` recorded) plus the
five params logged (`seed`, `hidden`, `max_epochs`, `patience`, `learning_rate`) are jointly
sufficient to reproduce this training path's result exactly. Nothing had to be added to the
logging to make it pass — a genuinely different outcome from P04's and P05's determinism
bugs, where the first version of the relevant test *did* fail and caught something real. Worth
recording as a real (negative) result rather than implying every milestone finds a bug: this
one didn't, because the determinism discipline P04 already established was already sufficient.

## A gap a later audit caught: the anomaly table had no run-id column

The spec's own constraint is explicit: "every number in `docs/RESULTS.md` must be traceable to
an MLflow run id." `pdm/evaluation/results.py`'s baseline/classification/AI4I tables satisfied
this from the start via `_mlflow_runs_cell`. The anomaly-detection table
(`_anomaly_table`, sourced from `artifacts/anomaly_experiments/summary.json` rather than the
per-run `metrics.json` glob the other tables use) did not — it rendered F1/precision/recall/
PR-AUC with no run-id column at all, even though `final_cmapss_evaluation`/
`final_nab_evaluation` (`pdm/models/torch/anomaly_experiments.py`) already wrapped every seed in
`start_run(...)` and logged real metrics there; the run ids were simply never captured out of
that `with` block or threaded into the returned dict. This was a genuine, unnoticed violation of
this milestone's own explicit constraint, caught by a later cross-milestone audit, not by this
milestone's own tests (none of which asserted anything about the anomaly table's columns).
Fixed by capturing `run.info.run_id` per seed into a `mlflow_run_ids: dict[seed, run_id]` on both
functions' return values (the same shape the baseline runner already uses), threading it through
`summary.json`, and adding the column to `_anomaly_table` — every row within a dataset block
shares the same run-id cell because it genuinely is the same set of runs (AE-alone,
Isolation-Forest-alone, and every fused threshold style are all logged inside one
`start_run` per seed, not one run per row).

## Scope decisions

- **Only PyTorch/TensorFlow RUL and anomaly training runs are registrable models.** Classical
  baselines (`pdm/models/baseline/runner.py`) log params/metrics/tags per (model, seed) — fully
  satisfying "no training path may write metrics only to stdout" — but never call
  `mlflow.*.log_model`/`log_artifact` for a model file, so there is nothing at a `runs:/.../
  bundle` path to register. This is intentional, not an oversight: `pdm-sentinel-rul` and
  `pdm-sentinel-anomaly` name the two deep-learning production paths the spec's registry
  deliverable is about; a `dummy_mean` baseline is a measuring stick, never a promotion
  candidate, and forcing it into the registry would just add noise to the model list.
- **The bottleneck ablation, contamination sweep, and fusion-weight sweep (P06) are not logged
  to MLflow individually** — only the final 5-seed C-MAPSS/NAB evaluation is (one run per seed
  per dataset). These sweeps are hyperparameter search, not candidate models; logging every
  point of a sweep as its own "run" would make the registry/experiment view mostly noise from
  configurations nobody will ever promote. The full sweep results remain fully traceable through
  their existing JSON artifacts (`artifacts/anomaly_experiments/*.json`) and
  `docs/decisions/P06-anomaly.md` — MLflow tracks the candidates that matter for promotion, the
  JSON artifacts track the exploration that produced them.
- **The framework benchmark (P05)'s per-seed regressor/classifier runs are logged**
  (`framework_benchmark__regression__<framework>__seed<n>` etc.) since those are genuine
  candidate trainings, one per framework; the benchmark's supplementary measurements (input
  pipeline timing, SavedModel export latency) are not runs in their own right and stay in the
  existing JSON artifact only, matching how P05's own decision log already presents them.
- **Docker Compose for the MLflow UI (`docker/docker-compose.yml`) is written and validated**
  (`docker compose config` parses it cleanly) with its backing store on a named Docker volume
  (`pdm-sentinel-mlflow-data`) per the deliverable — but that volume is separate from the local
  `./mlruns` directory the `pdm` CLI actually writes to, so it starts empty. For inspecting
  *this session's* real runs, `mlflow ui --backend-store-uri file:./mlruns --port 5000` (no
  Docker) was used instead, pointed straight at the same file store every `pdm train`/`pdm
  evaluate`/`pdm registry` invocation this session used. Unifying the two (bind-mounting
  `./mlruns` into the container, or pointing local runs at a shared server) is P08's job once
  the serving container exists and the whole stack's storage story needs deciding together.

