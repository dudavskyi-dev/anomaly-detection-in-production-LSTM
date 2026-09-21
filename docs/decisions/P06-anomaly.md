# P06 — Anomaly detection decision log

Everything below is measured from a real run of `pdm anomaly` on FD001 (17 features, same
subset/features as P04/P05), full training budget, all 5 configured seeds for every headline
number, traceable to `artifacts/anomaly_experiments/*.json` and `docs/RESULTS.md`'s "Anomaly
detection" section (generated from `artifacts/anomaly_experiments/summary.json` — see
`pdm.evaluation.results._anomaly_table`).

## Why healthy-only training makes this semi-supervised, not unsupervised

The LSTM autoencoder and Isolation Forest are both fit **only** on windows where the (already
capped, from P02) RUL target equals `settings.data.rul_cap` — i.e. windows from early enough in
an engine's life that P02's own labelling assumes no observable degradation yet. That label —
"this window is healthy" — comes from the same run-to-failure ground truth C-MAPSS provides for
RUL regression, not from the anomaly detector inventing or discovering it. A genuinely
unsupervised setup would have no such label at all: every window would be fair game for
training, and the model would have to somehow infer on its own which portions of each engine's
life were normal. What's actually happening here is standard semi-supervised anomaly detection —
a small amount of label information (which examples are "normal") shapes what the model learns
to reconstruct well, and everything scored later (including windows the label says are
degrading) is unlabelled at *scoring* time. That's a meaningfully weaker assumption than
supervised failure classification (P04's classifier, which needs a label on every window,
including the anomalous ones), but a meaningfully stronger one than true unsupervised detection.

**What I'd do with no labels at all** (not just no test labels — no way to say which *training*
windows were healthy): the practical answer used across real predictive-maintenance deployments
is to substitute a domain heuristic for the missing label — e.g. train on the **first N cycles**
of every unit's life (a fixed, small fraction, on the assumption that catastrophic early failure
is rare and most units start out healthy) rather than on a RUL-derived cutoff, and accept that
this heuristic is now the thing that needs validating (e.g. by checking that the resulting score
distribution on the same early cycles is unimodal and tight, which would be weak evidence the
heuristic assumption held). The score itself is still trained the same way afterward; only the
mechanism for choosing which windows count as "healthy" changes.

## The benchmark table

Chosen configuration: **bottleneck (latent) dimension 2**, **fusion weight 0.8** (LSTM-AE
share), both picked on validation. Full 5-seed, full-training-budget test results (from
`artifacts/anomaly_experiments/final_cmapss_evaluation.json` /
`final_nab_evaluation.json`, also in `docs/RESULTS.md`):

**C-MAPSS** (healthy vs. late-life windows, test):

| Detector | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| LSTM autoencoder (percentile-99) | 0.050 ± 0.012 | 0.695 ± 0.044 | 0.026 ± 0.007 | 0.492 ± 0.015 |
| Isolation Forest (percentile-99) | 0.315 ± 0.022 | 0.931 ± 0.018 | 0.190 ± 0.016 | 0.692 ± 0.005 |
| Fused, max-F1 threshold | **0.648 ± 0.007** | 0.495 ± 0.010 | 0.942 ± 0.060 | 0.550 ± 0.011 |
| Fused, 90%-precision threshold | 0.224 ± 0.078 | 0.562 ± 0.023 | 0.147 ± 0.064 | 0.550 ± 0.011 |
| Fused, percentile-99 threshold | 0.077 ± 0.015 | 0.753 ± 0.042 | 0.041 ± 0.008 | 0.550 ± 0.011 |

**NAB** `machine_temperature_system_failure` (point-wise, test — one threshold style only, see
below):

| Detector | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| LSTM autoencoder (percentile-99) | 0.293 ± 0.039 | 0.268 ± 0.072 | 0.351 ± 0.045 | 0.350 ± 0.097 |
| Isolation Forest (percentile-99) | **0.548 ± 0.004** | 0.771 ± 0.008 | 0.425 ± 0.004 | 0.509 ± 0.010 |
| Fused, percentile-99 threshold | 0.294 ± 0.038 | 0.267 ± 0.073 | 0.358 ± 0.046 | 0.352 ± 0.098 |

**The number that goes on the CV**: C-MAPSS anomaly-detection **F1 = 0.648 ± 0.007** (fused
score, max-F1-on-validation threshold, 5 seeds) — this is the best defensible C-MAPSS number,
and it clears the classical Isolation Forest baseline (0.315) by a wide margin, but only because
of *threshold* quality, not detector quality alone (see below). NAB's best number is **F1 = 0.548
± 0.004**, and it belongs to the plain Isolation Forest, not the LSTM-AE and not the fused score
— reported honestly even though it means the "sophisticated" detector and the fusion step both
lose to the classical baseline on the external validation dataset.

## Bottleneck ablation

Validation F1 (percentile-99 threshold, 2 seeds, `artifacts/anomaly_experiments/bottleneck_ablation.json`):

| Latent dim | F1 | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| 2 | **0.418 ± 0.018** | 1.000 | 0.264 | 0.858 |
| 4 | 0.378 ± 0.050 | 1.000 | 0.234 | 0.838 |
| 8 | 0.360 ± 0.007 | 0.999 | 0.219 | 0.827 |
| 16 | 0.320 ± 0.027 | 1.000 | 0.190 | 0.811 |
| 32 | 0.322 ± 0.003 | 0.996 | 0.192 | 0.812 |

**F1 falls monotonically as the bottleneck widens** (with 16→32 flat within noise, not a real
reversal) — precision stays pinned near 1.0 across every bottleneck size (expected: the
percentile-99 threshold is defined to keep the false-positive rate on *healthy* data at ~1%
regardless of bottleneck size), so the whole F1 decline is a **recall** collapse: 0.264 → 0.192.
This is the identity-function collapse the ablation was designed to find, and it starts
immediately — there's no "safe" wide bottleneck in this range, only degrees of it. The smallest
tested bottleneck (2) won outright, which was not the expected shape (a classic collapse curve
usually holds roughly flat before dropping); the honest reading is that *any* headroom beyond
what's needed to represent "healthy" starts being spent on reconstructing anomalies too, on this
subset, at this window size.

A genuine, orthogonal piece of evidence that the bottleneck really constrains capacity the way
the ablation assumes it does: `tests/test_torch_autoencoder.py`'s overfit-tiny-batch test
originally used `latent_dim=4` on 8 windows of shape `(6, 3)` (18 values each) and **could not**
drive reconstruction error below ~0.6 no matter how long it trained (300 epochs, several
learning rates tried) — 4 numbers structurally cannot encode 18 independent random values.
`latent_dim=16` on the same data converges to ~0 within 500 epochs. This is the *other* end of
the bottleneck-ablation story: too narrow, and the model can't even memorise a training batch it
has already seen; too wide, and (per the ablation above) it reconstructs everything, healthy or
anomalous, well enough that detection collapses. The useful operating range is the middle —
except empirically here, the "middle" of the tested grid was already past the point where
narrower kept winning, meaning latent dims below 2 might be worth trying in a follow-up if the
goal is squeezing out more F1, at the cost of moving closer to the tiny-batch capacity floor the
overfit test found.

## Isolation Forest contamination sensitivity

Validation F1 via IF's own `contamination`-implied operating point (`model.predict`, 2 seeds,
`artifacts/anomaly_experiments/isolation_forest_contamination_sweep.json`):

| Contamination | F1 | Precision | Recall |
|---|---|---|---|
| 0.02 | 0.675 | 0.965 | 0.524 |
| 0.05 | 0.728 | 0.935 | 0.600 |
| 0.10 | 0.757 | 0.896 | 0.661 |
| 0.15 | 0.772 | 0.872 | 0.700 |
| 0.20 | 0.782 | 0.850 | 0.732 |

Smooth, monotonic precision/recall tradeoff across the whole tested grid — no cliff, no
instability, F1 still climbing at the top of the range tried (0.20). Isolation Forest is
noticeably **less sensitive** to this hyperparameter than the autoencoder is to bottleneck size:
every contamination value here gives a reasonable, usable operating point, whereas the
bottleneck ablation above shows real degradation start immediately outside the smallest tested
value. `contamination` behaves like a normal precision/recall dial; `latent_dim` behaves like a
capability switch.

## Fusion weight selection

Raw LSTM-AE reconstruction error (an unbounded, right-skewed MSE) and Isolation Forest's score
(roughly zero-centred, an average isolation path length) are not on the same scale — averaging
them directly would let whichever one happens to produce numerically larger raw values dominate
the fusion regardless of which is actually more discriminative. Both are standardised
(z-scored) using **their own training-score mean/std** (`pdm.evaluation.fusion.ScoreStandardizer`,
fit once on healthy training windows, frozen and reused — never refit on validation or test)
before being combined, the same train-only-statistics discipline the feature scaler uses.

Validation F1 at the max-F1-on-validation threshold, swept over the fusion weight (LSTM-AE
share; `1.0` = AE alone, `0.0` = Isolation Forest alone; 2 seeds,
`artifacts/anomaly_experiments/fusion_weight_sweep.json`):

| Weight | F1 | Precision | Recall |
|---|---|---|---|
| 0.0 (IF alone) | 0.816 ± 0.011 | 0.710 | 0.963 |
| 0.5 | 0.819 ± 0.014 | 0.721 | 0.955 |
| **0.8** | **0.824 ± 0.017** | 0.706 | 0.989 |
| 0.9 | 0.824 ± 0.015 | 0.714 | 0.974 |
| 1.0 (AE alone) | 0.821 ± 0.008 | 0.717 | 0.962 |

The curve is nearly flat (0.816 to 0.824 across the whole sweep) — fusion helps, but only
marginally, over either detector alone **at this threshold style**. `0.8` and `0.9` tie for the
best mean F1; `0.8` was picked because it appears first in the swept grid. Given how flat this
curve is, the specific weight chosen matters far less than the choice to fuse at all (the
worst point on the curve, 0.816, still isn't far off the best, 0.824) — a useful thing to say
out loud rather than imply the weight sweep found a sharp, well-defined optimum it didn't.

## Threshold selection (deliverable #5)

Three threshold styles were computed on the fused score and reported on test:

- **Max-F1** (validation): the operating point that would look best in a single-number
  headline, but is expensive to defend — it implicitly assumes the validation split's exact
  class balance and cost tradeoff transfer unchanged to production traffic.
- **90%-precision** (validation): the recall achieved while holding false-alarm rate low enough
  that an operator wouldn't start ignoring the system.
- **Percentile-99-of-healthy-training-scores**: the only style that needs **no labels at all** —
  computable the moment training finishes, before any validation data is even scored.

Applied to the fused score and scored on **test** (all three thresholds were chosen on
validation or train, per their definitions — never on test), 5 seeds:

| Threshold style | Test F1 | Test Precision | Test Recall |
|---|---|---|---|
| Max-F1 (validation) | **0.648 ± 0.007** | 0.495 ± 0.010 | 0.942 ± 0.060 |
| 90%-precision (validation) | 0.224 ± 0.078 | 0.562 ± 0.023 | 0.147 ± 0.064 |
| Percentile-99 (train) | 0.077 ± 0.015 | 0.753 ± 0.042 | 0.041 ± 0.008 |

Two things stand out, both consistent across all 5 seeds (checked the per-seed arrays, not just
the means):

1. **The max-F1 threshold's validation performance (0.824, from the fusion sweep above) does not
   fully transfer to test (0.648)** — a real, measured generalisation gap of about 18 points of
   F1. Still by far the best of the three styles on test, but a reminder that "best on
   validation" is optimistic for what production will actually see.
2. **The 90%-precision threshold badly overshoots its own promise**: tuned to guarantee ≥90%
   precision on validation, it delivers **56%** precision on test — every single one of the 5
   seeds, not an outlier (`precision` values: 0.588, 0.550, 0.590, 0.538, 0.544). A threshold
   that silently fails to deliver the precision guarantee it was chosen for is a genuinely
   dangerous thing to ship without knowing this — an operator told "90% precision" who actually
   gets 56% will lose trust in the system fast, for reasons that look like a broken promise
   rather than an expected tradeoff.

**What I'd ship: the max-F1 threshold, but re-validated on a held-out slice before going live,
not the number reported here.** It's the only one of the three that's actually usable (F1 0.65
vs 0.08–0.22 for the other two), and its failure mode (dropping from 0.824 to 0.648) is a
"less good than hoped," not a broken guarantee — unlike the precision-target threshold, which
promised something concrete and didn't deliver it. Between recall-heavy and precision-heavy
framing: the spec's own reasoning holds here (a missed failure costs more than a false alarm),
and the max-F1 operating point already lands at 94% recall with 50% precision, which is roughly
the right side of that tradeoff for a maintenance context — half of the flagged windows would be
false alarms, which is a real, quantified alert-fatigue cost to weigh against catching 94% of
real degradation, not a number to hide.

## The C-MAPSS-to-NAB gap

**The measured result is the opposite of the spec's stated expectation ("expect the NAB numbers
to be worse"), and the reason why is itself the most useful finding in this section.** At the
*same* threshold style (percentile-99, the only one computable honestly on both datasets), NAB's
numbers are clearly **better** than C-MAPSS's, for both detectors:

| Detector | C-MAPSS F1 (percentile-99) | NAB F1 (percentile-99) |
|---|---|---|
| LSTM autoencoder | 0.050 | 0.293 |
| Isolation Forest | 0.315 | 0.548 |
| Fused | 0.077 | 0.294 |

But the *best achievable* number on each dataset flips the comparison back the way the spec
expected: C-MAPSS's best (fused, max-F1-on-validation threshold) reaches **0.648**, comfortably
above NAB's best (Isolation Forest alone, percentile-99 — the only style available) at **0.548**.

Both halves of this are real and explain each other:

1. **Threshold asymmetry (methodological, not about data difficulty).** C-MAPSS has a real
   validation split with labelled anomalies, so it gets to use the max-F1 threshold — the style
   that turned out to help the most (0.077 → 0.648, a 8.4x improvement over the label-free
   threshold on the same fused score). NAB has no such split (see below) and is stuck with the
   conservative percentile-99 style for every detector. Comparing "C-MAPSS's best" against
   "NAB's only option" is not comparing the datasets, it's comparing threshold-tuning budgets.
2. **Class balance (a genuine, data-driven reason the *label-free* comparison runs backwards).**
   C-MAPSS's "anomalous" class (`rul < healthy_rul_threshold`) is the **majority** of a typical
   engine's recorded life — most of a unit's operating history is below the RUL cap, only its
   early cycles count as "healthy." A percentile-99-of-healthy threshold is calibrated to flag
   only the most extreme 1% of the *small* healthy reference distribution, which — when the
   thing you're trying to catch is actually the majority class — catches very little of it
   (recall 0.026–0.190). NAB's anomalous rows are the traditional minority (~11% of test rows),
   which is exactly the regime a percentile-of-normal threshold is designed for, so it performs
   far closer to as intended. The label-free comparison isn't "NAB is easier," it's "C-MAPSS's
   class balance defeats this specific threshold style structurally," independent of anything
   about sensor realism.

Net: the spec's underlying intuition (real-world external validation should be harder than the
simulator it wasn't trained on) is still directionally right once threshold-tuning budget is
held equal — but the raw, same-threshold numbers say the opposite, and reporting only the
"expected" comparison would have hidden a real, useful methodological lesson about how heavily a
percentile-based threshold's apparent performance depends on the base rate it's applied to.

**A second, real finding: the fusion weight chosen on C-MAPSS does not transfer to NAB.** NAB's
fused score (weight 0.8, i.e. mostly AE) scores F1 0.294 — barely different from AE alone
(0.293), and far below what Isolation Forest alone achieves on NAB (0.548). Had the fusion
weight been re-tuned for NAB specifically (impossible here — no NAB validation split — but
telling nonetheless), a weight favouring IF far more heavily would clearly have won. Shipping
one fusion weight across both a simulator and a real deployment target is exactly the kind of
choice that looks reasonable in isolation and turns out to silently cost performance on the
target that matters more.

There is also a real, un-threshold-related reason to expect NAB to be intrinsically harder that
the numbers above don't fully capture: C-MAPSS's "anomalous" windows are *known*-degrading
engine cycles from the same simulator that generated the healthy windows, sharing the same
sensor model and noise characteristics, while NAB's series is a real physical system with real
sensor noise and only four labelled events total to learn anything from. That gap is real; it
just isn't the dominant effect in the numbers actually measured here.

### Why NAB gets only one threshold style

`pdm/preprocessing/nab_pipeline.py` splits the series into **train** (everything strictly before
the first labelled anomaly window) and **test** (everything from the training cutoff onward,
including the labelled anomalies) — there is no NAB validation split at all. Tuning a max-F1 or
precision-target threshold needs labelled anomalies that are disjoint from whatever gets scored
as test; with only four labelled events in the whole series, carving out a validation slice would
mean either starving training of its only known-healthy stretch or leaking the very labels test
is scored against. The percentile-99-of-healthy-training-scores threshold needs no labels at all
and is the only one that can be computed honestly here — which is itself the answer to "what
would you do with no labels" from a different angle: on NAB, this project effectively already
had no threshold-tuning labels, and the percentile method is what it fell back on.

## NAB's own scoring vs. point-wise scoring

Every number in this log is **point-wise**: each windowed timestamp is an independent
prediction, scored as its own true/false positive/negative. NAB's own published scoring method
does not work this way — it credits at most one true positive per labelled anomaly *window*
(rewarding early detection within the window more than late detection, via a sigmoid weighting
function keyed to how early within the window the first detection lands) and penalises multiple
detections inside the same labelled window as redundant rather than as extra true positives. The
practical effect: point-wise scoring here likely *undercounts* precision relative to NAB's own
metric whenever a detector correctly flags a whole anomalous stretch (every flagged timestamp in
a genuinely-anomalous window counts as a separate true positive under point-wise scoring, which
inflates recall somewhat comparably, but a detector that fires many times within one truly
anomalous window is not really "9 correct detections," it's one). Full NAB scoring (with its
per-application reward profiles) was not reimplemented here — out of scope for what P06 asked
("note how NAB's own window-based scoring differs"), not attempted and then hidden.

## TensorFlow production-mirror bundle

A single-seed (seed 42, full training budget) TensorFlow LSTM autoencoder was trained on the
same healthy windows and bundled (`artifacts/tf_lstm_ae__FD001__seed42/`) as the "production
path" mirror the spec calls for — not benchmarked across seeds or frameworks the way P05 did for
the RUL regressor, since P06's deliverables ask for a working production artifact, not a second
framework bake-off. Its own test metrics (percentile-99 threshold, this one seed only): F1
0.071, precision 0.762, recall 0.037 — in the same range as the PyTorch AE's own single-seed
number at this threshold style (consistent with the two being structurally similar, unlike P05's
architecture where a custom Keras cell was needed for exact parity; no such reconciliation was
attempted here).

## Plots (deliverable #7)

All under `reports/anomaly/`, generated by `pdm anomaly` from the seed-42 full-budget run:
`f1_vs_bottleneck.png` (the ablation curve above), `score_distribution_cmapss.png` (healthy vs.
late-life reconstruction-error histograms with the percentile-99 threshold marked),
`pr_curve_ae.png` / `pr_curve_isolation_forest.png` (precision-recall curves, test), and
`per_sensor_heatmap.png` (per-sensor reconstruction error for the 15 highest-scoring true
anomalies in test — the explainability view: which sensors actually drove those detections).

## Environment / implementation notes

- `classification_metrics` (`pdm/evaluation/metrics.py`) gained an `include_brier: bool = True`
  parameter this milestone: `brier_score_loss` requires a probability in `[0, 1]`, and every
  anomaly score here (raw reconstruction MSE, standardised fused z-scores, IF's negated
  `score_samples`) is emphatically not one — the first version of this milestone's code crashed
  immediately with `ValueError: y_proba contains values greater than 1` the first time a
  fused score was scored. Every anomaly-detection call site now passes `include_brier=False`;
  every pre-existing caller (P03/P04's classifiers, which do pass calibrated probabilities)
  keeps the old default and is unaffected.
- The LSTM autoencoder's training loop reuses `pdm.models.torch.train.train_model` completely
  unmodified: `WindowDataset` already puts the input windows under the key `"windows"` in every
  batch, so passing `target_col="windows"` makes the shared loop compare the model's
  reconstruction against the same tensor it was given as input — no new dataset or training-loop
  code was needed to make an existing supervised-training harness work for a reconstruction task.
