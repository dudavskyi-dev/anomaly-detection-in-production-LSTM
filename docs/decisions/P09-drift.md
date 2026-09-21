# P09 — Data drift detection and automated retraining decision log

## Three gaps a self-review caught before calling this done

A first pass at this milestone was reported as "done" and was not: re-reading the spec's
deliverable text literally against the actual code found three real gaps.

1. **The report was missing "the verdict."** Deliverable #5 asks for the PSI/KS table *and the
   verdict*. The trigger decision (fired or not, why, cooldown state, candidate run id) was
   computed and written to `drift_result.json` and stdout, but never rendered into `report.md`
   itself. Fixed: `render_markdown_report` now takes a `trigger` dict and renders a
   `## Verdict: RETRAIN TRIGGERED` / `## Verdict: no action` section up top, with the reason and
   (when applicable) the exact `pdm registry promote` command to run next.
2. **"Through the replay simulator" wasn't literal.** Deliverable #3 explicitly names
   `pdm.ingestion.replay` — the module built in P01 to simulate a live telemetry feed. The first
   version of `load_raw_cmapss_windows` bypassed it entirely with a direct `pd.read_parquet`
   call; the numbers were correct but the specific module the spec named was never actually
   exercised. Fixed: the file backing whichever split is being checked as "production traffic"
   is now read via `replay(path, rate_hz=0.0, noise_std=..., dropout_prob=...)` — same code path
   a live replay would run, sleep disabled. `pdm drift check` now exposes `--noise-std`/
   `--dropout-prob` so the simulator's own jitter/dropout knobs are genuinely reachable, not
   inert passthrough parameters. Verified the default (both zero) still reproduces the exact same
   numbers as before (FD001-val mean PSI 0.0168, RMSE 13.387 — identical to the pre-fix run), and
   that a nonzero `noise_std`/`dropout_prob` demonstrably changes the windows
   (`tests/test_preprocessing_pipeline.py::test_load_raw_cmapss_windows_genuinely_flows_through_the_replay_simulator`).
3. **No scheduling existed.** Deliverable #4 wants this "running on a schedule (a compose-managed
   job or a simple scheduler; no Airflow needed)." Only an on-demand CLI command and `make drift`
   existed — nothing fired it periodically on its own. Fixed: added a `drift-monitor` service to
   `docker/docker-compose.yml`, a `while true; ...; sleep $INTERVAL; done` loop around
   `pdm drift check` for FD002 and FD003 — deliberately not a cron container or workflow engine,
   since a shell loop is all a single periodic CLI job needs. It points its MLflow client at the
   compose-managed `mlflow` service (`PDM__TRACKING__URI=http://mlflow:5000`) rather than the
   host's local `./mlruns`, so a triggered candidate shows up in the same registry the rest of
   the stack uses. Reuses `docker/Dockerfile` via a new `EXTRAS` build arg
   (`torch,serving,tracking,monitoring` instead of the `api` service's `torch,serving`) rather
   than maintaining a near-duplicate Dockerfile. **Built and actually run** (not just
   `docker compose config`-validated): `docker compose build drift-monitor` succeeded, and
   `docker compose run --rm --entrypoint "python -m pdm.cli drift check --production-subset FD002
   --split val --no-retrain" drift-monitor` genuinely executed inside the container against the
   real mounted data/artifacts volumes and the compose-managed MLflow service (deliberately
   `--split val`, not the default `test`, purely so this validation run couldn't be confused with
   — or accidentally re-trigger cooldown state shared with — the real FD002/test check earlier in
   this log), producing a real drift report (mean PSI 3.3488, RUL RMSE 35.278 — both a similar
   order of magnitude to the FD002/test numbers above, as expected for the same drifted subset)
   and writing it back to the host-mounted `reports/` directory.

All numbers below come from `pdm drift check`, run for real against the actual
`artifacts/torch_rul_regressor__FD001__seed42` (RUL) and `artifacts/torch_lstm_ae__FD001__seed42`
(anomaly) bundles, replaying real C-MAPSS FD001/FD002/FD003 windows as "production" traffic
through the replay-style pipeline described below. Nothing here is invented or backfilled from
intuition about what the numbers "should" look like — several of them (see the two genuine
findings below) surprised the implementation and changed it.

## Methodology: what "production traffic" actually means here

A served bundle only ever sees raw sensor windows and must scale them with **its own** scaler —
the one fit once, at training time, on FD001 alone (`pdm.preprocessing.pipeline.
load_raw_cmapss_windows` loads FD002/FD003 labelled-but-unscaled, selects exactly the FD001
bundle's feature list, and `RulBundle.scale()` applies FD001's frozen `mean_`/`std_`). This is
deliberate and is where the drift actually comes from: FD002 has six operating conditions and
FD003 has a second fault mode, neither of which FD001's scaler (or the model) was ever fit to
expect.

**FD001's own train/test split is not a fair "no-drift" baseline** — a real, non-obvious finding
caught while building the false-positive check. C-MAPSS's train sequences run every engine to
failure; its test sequences are truncated at an unlabelled, non-uniform cutoff before failure.
Comparing FD001-train (the reference) against FD001-*test* therefore measures a real but
uninteresting artifact of the benchmark's own construction (test units are systematically
under-represented at the high-degradation end), not genuine production drift: it produced a
misleading mean PSI of 0.18 and a prediction-drift PSI of 0.46 on data from the *same subset the
model was trained on*. The false-positive check below instead uses FD001's **val** split — held
out from the same full run-to-failure trajectories as train, via the same unit-disjoint split
training itself uses — which is the correct apples-to-apples "no real distributional shift, only
sampling noise" comparison. (The RMSE this reproduces, 13.387, matches the bundle's own logged
`val_rmse` to five significant figures — a good independent sanity check that the raw-loading
path is numerically consistent with the original training pipeline.)

## Drift scores: FD001-holdout vs. FD002 vs. FD003

| Traffic | Mean PSI (aggregate) | Max feature PSI | Features PSI-significant (of 17) | KS-reject after BH (of 17) | Prediction-drift PSI |
|---|---|---|---|---|---|
| FD001 val (false-positive check) | **0.017** | 0.123 (sensor_14) | 0 | 2 | 0.027 |
| FD002 test | **3.411** | 5.374 (op_setting_2) | 17 | 17 | 5.465 |
| FD003 test | **0.772** | 2.498 (sensor_6) | 14 | 15 | 0.746 |

## Model performance on each, and the correlation with drift score

| Traffic | RUL RMSE | RUL MAE | Anomaly F1 (precision / recall) |
|---|---|---|---|
| FD001 val (baseline) | **13.39** | 9.91 | 0.294 (0.964 / 0.174) |
| FD002 test | **31.46** (2.35x baseline) | 28.49 | 0.659 (0.492 / 1.000) |
| FD003 test | **81.89** (6.12x baseline) | 63.04 | 0.384 (0.305 / 0.518) |

The drift signal is not a nuisance alarm: both drifted subsets show real, large RMSE degradation
over the clean baseline, in the direction the drift score predicts (higher drift, worse
performance, in aggregate). But **the relationship is not simply "more PSI = proportionally worse
RMSE,"** and the reason why is itself informative:

- **FD002's enormous PSI (3.4, values into the thousands of "effective std deviations" for
  `op_setting_1`/`op_setting_2`) is dominated by a scaler-mismatch artifact, not just "the
  sensors moved."** FD001 is single-condition: its own `op_setting_1` has a training std of
  `0.00219` (see `scaler.json`) because the value barely varies within FD001 at all. Dividing
  FD002's genuinely multi-condition operating settings (which span a wide operational envelope)
  by that near-zero denominator inflates the z-scores enormously — mathematically correct given
  the scaler that's actually deployed, but a reminder that **PSI magnitude alone doesn't
  calibrate to "how bad is this,"** especially against a reference distribution with degenerate
  variance in some features. FD002's degradation (2.35x RMSE) is real and substantial, but far
  smaller than the PSI number alone would suggest.
- **FD003's more moderate PSI (0.77) reflects a second fault mode FD001 never saw, and correlates
  with *worse* degradation than FD002's** (6.12x vs. 2.35x baseline RMSE) **despite a much smaller
  drift score.** FD003 is single-condition like FD001 — its `op_setting_1`/`op_setting_2` PSI are
  both ~0 (0.001, 0.003), correctly showing "same operating envelope" — but its sensor
  *degradation trajectories* differ enough that the model's RUL predictions are catastrophically
  off. This is exactly the case the P09 spec's "optional" prediction-drift check exists for:
  input drift (per-feature marginals) can under-state a problem that only shows up once you look
  at what the model actually does with the input — FD003's prediction-drift PSI (0.746) tracks
  its real degradation far more proportionately than its input PSI does.

The practical lesson: aggregate input-PSI is a reasonable trigger signal (it fired correctly on
both genuinely-degraded subsets and stayed quiet on the clean baseline), but **it should never be
read as a severity estimate on its own** — that's what pairing it with prediction drift and, where
labels exist, measured degradation is for.

## Thresholds chosen and the false-positive rate on the no-drift holdout

`settings.monitoring.psi_significant_threshold = 0.25` is the retrain-trigger threshold (deliberately
reusing the same 0.25 "significant" PSI band the literature already treats as the significant-shift
cutoff, rather than inventing a separate number). Against the honest FD001-val false-positive
baseline (mean PSI 0.017, max feature PSI 0.123), this threshold is not close to firing —
**0 of 17 features exceed the significant PSI band**, so the aggregate score sits roughly 15x
below the trigger point. `psi_n_bins=10` (quantile bins on the training distribution) and
`ks_reference_sample_size=2000` are both chosen for the same reason: enough resolution to detect
a real shift without every bin being sparse enough to need the epsilon floor on the reference
side too.

**A quieter false positive that the aggregate score correctly stays below threshold for:**
Benjamini-Hochberg still rejects 2 of 17 features (`sensor_14`, `sensor_9`) on the clean FD001-val
baseline at `alpha=0.05` — see the next section for why, and why that's expected and fine.

## Why two detection methods, not one — and where they disagreed

PSI and KS answer different questions and disagree in both directions in this data:

- **PSI is a practical-significance measure with built-in thresholds** (<0.1 / 0.1-0.25 / >0.25)
  independent of sample size — the same PSI value means the same thing whether it came from 100
  samples or 100,000.
- **KS is a statistical-significance test whose power scales with sample size.** This project's
  windows flatten every timestep into a sample (thousands of points per feature, even for a
  modest number of windows), so KS has enormous power to detect *any* real difference, however
  practically tiny.

**The disagreement this produced, concretely:** on the FD001-val false-positive baseline,
`sensor_14` has PSI 0.123 (`"moderate"`, and per-feature, not even at the *significant* band) but
its KS test rejects at `p=4.79e-13` — a p-value small enough to survive Benjamini-Hochberg
correction across all 17 features. Taken alone, KS says "these are not the same distribution,
reject the null" on data this project has independently confirmed (via `val_rmse` matching
training's own number) is a legitimate, non-drifted holdout. PSI's threshold correctly stays
quiet. **Skipping the multiple-comparison correction entirely would make this worse, not
better:** at an uncorrected `alpha=0.05` across 17 features, the expected number of false
positives from chance alone is already ~0.85 per check even if nothing has drifted; this project
sees 2 (using BH, the less conservative of the two implemented methods), which is in the
expected range for a small, real, non-drift effect plus ordinary multiple-comparisons risk.
Skip correction on a real deployment with many more monitored features, and PSI would end up
the *only* signal worth trusting, defeating the purpose of running KS at all.

**Bonferroni vs. Benjamini-Hochberg** (`tests/test_monitoring_drift.py::
test_benjamini_hochberg_is_less_conservative_than_bonferroni_on_correlated_shift` pins this down
with a synthetic case): this project's sensors are correlated by construction (several C-MAPSS
sensors move together with the same degradation trend), so a real shift tends to drop many
p-values together rather than producing one isolated tiny p-value. Bonferroni's flat
`alpha / n_features` line is tuned for the latter case and would under-detect the former;
Benjamini-Hochberg's rank-dependent line is deliberately chosen here (`settings.monitoring.
ks_correction = "benjamini_hochberg"`) because it controls the *false discovery rate* rather than
the probability of any single false positive, trading a slightly higher false-positive tolerance
for much better sensitivity to exactly the kind of correlated, multi-sensor shift this system
needs to catch.

## The retraining-on-drift trap, in my own words

Retraining automatically on whatever data triggered the drift alarm is not a safe default — it
can be exactly backwards. If the "drift" is a failing sensor, or the genuine onset of the fault
mode this whole system exists to catch, folding that data into a new training set doesn't fix
anything; it teaches the model that the anomaly *is* the new normal, quietly erasing the signal
the system was built to raise. A retrain that "succeeds" by this measure — the model now fits the
drifted data well — would be the worst possible outcome: confident, unalarmed, wrong.

This project's trigger (`pdm.monitoring.trigger`) is built so that path isn't available even by
accident, in two separate ways:

1. **Retraining never touches the raw drifted traffic.** `run_retraining_candidate` retrains on a
   fixed, already-labelled, already-trusted C-MAPSS subset (by default, whatever subset the
   currently-served bundle itself was trained on) via the exact same `pdm train` entry point a
   human would use — never on the FD002/FD003 windows that triggered the check. A real production
   system would need a human to curate and relabel drifted traffic before it's trusted as
   training data at all; this project doesn't attempt to automate that judgment call, and treats
   attempting to as the trap itself in a different shape.
2. **No auto-promotion.** `pdm.monitoring.trigger` never imports
   `pdm.tracking.promotion.promote_to_production` and never transitions anything to the
   `Production` stage. A triggered retrain produces a new registered model version left in
   `Staging` — visible, comparable, but inert until a human runs `pdm registry promote`, which
   applies the exact same P07 promotion gate (a real relative-improvement margin against the
   current Production model's own metric) as every other candidate. If the "retrained" model is
   actually worse (which a model trained on the same trusted data as before, at a different seed,
   usually will be — see the real run below), the gate refuses it, exactly as designed.

Together, these mean "drift fired" can only ever produce a *proposal*, evaluated by the same
gate as any other, never a fait accompli.

## Covariate drift vs. concept drift vs. label drift — what's actually detectable here, without labels

- **Covariate (input) drift** — the distribution of `P(X)` (sensor readings) has shifted,
  independent of whether `P(Y|X)` (the true relationship between readings and RUL) has changed.
  **Fully detectable without labels** — this is exactly what `compute_input_drift`'s PSI/KS
  measure, comparing raw incoming windows against the frozen training histogram. No ground truth
  RUL is required to compute it, which is the point: it's the only drift signal a real deployment
  can compute continuously, since production RUL/failure labels don't exist until an engine
  actually fails.
- **Concept drift** — `P(Y|X)` itself has changed: the same sensor readings now imply a different
  true RUL than they used to (a different fault mechanism, a different degradation rate). **Not
  directly detectable without labels**, by definition — you cannot measure a change in a
  relationship you cannot observe one side of in real time. What this project can and does do
  instead is use **prediction drift** (`compute_prediction_drift`) as an *indirect, partial*
  proxy: if the model's own output distribution shifts even where input marginals look only
  moderately different (FD003, above), that's consistent with — though not proof of — a concept
  shift the model is struggling with. It is a symptom check, not a diagnosis: a model that's
  simply become miscalibrated on in-range inputs would look the same way.
- **Label drift** — the distribution of `P(Y)` (true RUL/failure outcomes) has shifted on its own
  (e.g. a fleet that's aging on average, or one that just had a batch of engines replaced).
  **Not detectable at all without labels**, and this project does not claim to detect it: doing so
  requires the ground truth RUL/failure outcomes themselves, which is exactly the data a live
  deployment doesn't have until an engine reaches end of life. The honest degradation numbers in
  this log (RMSE/F1 on FD002/FD003) are only possible here because C-MAPSS ships labels for what
  is, narratively, "production" data — a real deployment would need a delayed, retrospective
  evaluation pipeline (scoring predictions against outcomes once they're finally known) to ever
  measure this at all, which is out of this project's scope.

## A real triggered retrain, end to end

Running `pdm drift check --production-subset FD002` (not `--no-retrain`) against the real FD002
traffic above genuinely fired: aggregate PSI 3.41 cleared the 0.25 threshold with cooldown clear,
which launched a real `pdm train --subset FD001 --seed 1042` run (seed offset by
`settings.monitoring.retrain_seed_offset`, so it wrote to
`artifacts/torch_rul_regressor__FD001__seed1042` rather than overwriting the currently-served
`seed42` bundle — no collision, confirmed by both bundle directories existing side by side
afterward), logged MLflow run `2b220c8df00746df978e4ca86dc230a9`, and registered it as
`pdm-sentinel-rul` **version 3 in stage Staging** — the trigger stopped there, exactly as
designed; it never touched the `Production` stage.

Running the promotion step by hand afterward, as a human operator would (`pdm registry promote
--run-id 2b220c8df00746df978e4ca86dc230a9`), the gate genuinely passed this one:

```
Promoted pdm-sentinel-rul version 3 to Production: candidate test_rmse=15.4534 <= required
16.0457 (production test_rmse=16.3732, margin=2.0%, higher_is_better=False) -> PASSED
```

The candidate's test RMSE (15.453) beat the previously-Production version's (16.373) by 5.6% —
comfortably past the 2% required margin — and the old Production version (trained at seed 42) was
archived automatically as part of the same promotion. **The honest caveat, stated plainly rather
than presented as more meaningful than it is:** this candidate was retrained on exactly the same
FD001 data and config as the model it replaced, at a different seed — it knows nothing whatsoever
about FD002. Its improvement reflects ordinary seed-to-seed variance in this project's own 5-seed
stability numbers (`docs/decisions/P04-pytorch.md`), not anything learned about the drifted
traffic that triggered the check. That's not a flaw in this run — it's the entire point of the
design in the previous section: the trigger's job is only to produce a fresh, honestly-evaluated
candidate through the same trusted pipeline and let the existing gate decide, never to imply the
retrain specifically "fixed" the drift. A retrain that actually addressed FD002's operating-
condition shift would need curated, relabelled FD002 training data and a human decision to use
it — outside this milestone's scope, and exactly the judgment call the trap section above argues
against automating.
