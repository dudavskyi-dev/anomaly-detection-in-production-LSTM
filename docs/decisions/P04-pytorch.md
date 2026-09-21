# P04 — PyTorch LSTM decision log

All numbers below are measured directly from real runs on FD001 (17 features after P01's
constant-sensor drop), traceable to `artifacts/torch_experiments/*.json` and
`artifacts/torch_rul_regressor__FD001__seed42/metrics.json`. Sweep experiments (window size,
architecture, RUL cap, classifier imbalance) use a **reduced budget** (`max_epochs=40,
patience=8`, seeds `(0, 1)`) for CPU-time reasons — see "Scope decision" below. The final
5-seed stability run and the shipped bundle use the full default budget
(`max_epochs=100, patience=10`, seeds `(0,1,2,3,4)` and `42` respectively).

## Parameter count (must match P05's TensorFlow mirror)

Default architecture — `LSTM(100) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(1)`, 17 input
features:

| Layer | Shape | Params |
|---|---|---|
| `lstm1.weight_ih_l0` | (400, 17) | 6,800 |
| `lstm1.weight_hh_l0` | (400, 100) | 40,000 |
| `lstm1.bias_ih_l0` / `bias_hh_l0` | (400,) × 2 | 800 |
| `lstm2.weight_ih_l0` | (200, 100) | 20,000 |
| `lstm2.weight_hh_l0` | (200, 50) | 10,000 |
| `lstm2.bias_ih_l0` / `bias_hh_l0` | (200,) × 2 | 400 |
| `head.weight` / `head.bias` | (1,50) / (1,) | 51 |
| **Total** | | **78,051** |

## Window size sweep (validation only, reduced budget, 2 seeds)

| Window size | Val RMSE | Epochs to best | Train wall-clock | p50 / p95 / p99 latency (ms) |
|---|---|---|---|---|
| 10 | 16.830 ± 0.745 | 11.5 | 58.1 s | 0.58 / 0.88 / 1.09 |
| 20 | 14.872 ± 1.289 | 18.5 | 139.8 s | 0.98 / 2.11 / 3.03 |
| 30 (default) | 13.398 ± 0.604 | 21.0 | 190.8 s | 0.79 / 1.18 / 1.38 |
| 50 | 12.658 ± 0.198 | 18.0 | 215.4 s | 0.94 / 1.78 / 2.43 |

RMSE improves monotonically with more context, with clearly diminishing returns: 10→30 buys a
20.4% RMSE reduction, 30→50 buys a further 5.5%. All inference latencies are sub-3ms regardless
of window size — for this model size, latency is not the constraint on window size, training
wall-clock and diminishing accuracy returns are. **Shipped default stays `window_size=30`**
(matching the spec default and P03's baselines, so every reported number in this project is
comparable to the same input representation); `window_size=50`'s further RMSE gain is real and
would be the first thing to adopt with a larger compute budget, since its latency cost is
negligible.

## Architecture ablation (validation only, reduced budget, 2 seeds)

| Configuration | Val RMSE | Parameters | Wall-clock | Epochs to best |
|---|---|---|---|---|
| Two-layer (100,50), dropout 0.2 (shipped) | 13.398 ± 0.604 | 78,051 | 179.3 s | 21.0 |
| Two-layer (100,50), no dropout | 13.755 ± 0.958 | 78,051 | 150.7 s | 16.5 |
| One-layer (100), dropout 0.2 | **12.608 ± 0.651** | 47,701 | 102.8 s | 21.0 |
| One-layer (100), no dropout | 12.900 ± 0.586 | 47,701 | 78.0 s | 19.0 |

**This is a genuinely surprising, real result, not the expected outcome**: the single-layer
model *beats* the two-layer (shipped) architecture at this training budget, using 39% fewer
parameters and ~43% less wall-clock time. Dropout helps in both cases (13.398 < 13.755;
12.608 < 12.900), so the regularisation direction is as expected — the layer-count result is
not. Two plausible explanations, neither confirmed further here for lack of compute budget in
this session: (a) the two-layer model has more capacity to fit and may need more than 40 epochs
to reach an optimum that the one-layer model reaches faster; (b) FD001 (single operating
condition, single fault mode — the "easiest" C-MAPSS subset) may simply not have enough temporal
complexity to reward a second LSTM layer, so its extra capacity is just extra optimisation
difficulty. This is exactly the kind of finding that should change what ships: **a case can be
made for a single-layer LSTM as the production architecture** on this specific subset, and the
two-layer spec default should be re-examined with a longer training budget before being treated
as settled. Reported honestly rather than discarded because it contradicts the spec's suggested
architecture.

## RUL cap ablation (validation only, reduced budget, 2 seeds)

| Condition | Val RMSE | Epochs to best |
|---|---|---|
| Capped at 125 (shipped) | 13.398 ± 0.604 | 21.0 |
| Uncapped | **30.362 ± 0.679** | 19.5 |

Capping more than halves RMSE (a 55.9% reduction) — an even larger effect than P02's quick Ridge
check on raw per-row features found (21.836 capped vs 38.876 uncapped, a 43.8% reduction).
The LSTM is more sensitive to the uncapped target's dominant, uninformative "healthy plateau"
signal than the simpler baseline was, which makes sense: a model with more capacity has more
room to waste fitting the plateau's noise.

Residual plots, one representative seed (seed 0: val RMSE 12.793 capped / 31.040 uncapped, both
consistent with the 2-seed sweep means above), are in `reports/p04_rul_cap_ablation/`
(`residuals_capped.png`, `residuals_uncapped.png`, `scatter_capped.png`, `scatter_uncapped.png`).
Both confirm the mechanism directly, not just by inference from the RMSE gap:

- **Capped** (`residuals_capped.png`): residuals are tight and centred on zero for true RUL
  below ~40, then visibly widen (roughly ±40) as true RUL approaches the 125 cap — exactly what
  the spec predicts: healthy engines look statistically alike regardless of how much life
  actually remains, so there is no sensor signal yet to distinguish "120 cycles left" from
  "90 cycles left," and the model's error concentrates precisely where capping says it should
  stop being asked to guess precisely.
- **Uncapped** (`residuals_uncapped.png`): a much starker failure mode — predictions saturate
  around 139 (the model's max predicted value across the whole validation set, `pred_rul_max
  = 139.02`, against a `true_rul_max` of 263), so residuals become *systematically* more negative
  as true RUL grows past ~140, forming a clean diagonal band down to −125. The model isn't
  noisily wrong at high RUL, it is essentially unable to represent RUL values much beyond what
  it saw densely in training, and capping is precisely the fix for that ceiling effect.

## Final 5-seed stability (shipped configuration: window_size=30, two-layer (100,50), dropout 0.2)

| Split | RMSE | MAE | R² | NASA score |
|---|---|---|---|---|
| Validation | 14.369 ± 0.981 | — | — | — |
| **Test** | **15.431 ± 0.493** | 11.586 ± 0.758 | 0.730 ± 0.017 | 51,262 ± 1,174 |

Individual seed test RMSEs: 14.873, 15.684, 14.852, 16.108, 15.640 (seeds 0-4).

**This does not clearly beat the P03 RandomForest baseline** (test RMSE 15.365 ± 0.284 from
`docs/RESULTS.md`) — the two are statistically indistinguishable (14.94–15.92 vs 15.08–15.65,
heavily overlapping ranges), and the LSTM's mean is very slightly *worse*. It does clearly beat
Ridge (16.954 ± 0.177). Per the project's own acceptance criteria ("a deep model losing to ridge
regression means something is wrong, and finding out what is the point"): the LSTM did not lose
to Ridge, but it also did not clearly win over the stronger classical baseline, and that is worth
taking seriously rather than reporting as a win. Candidate explanations, most to least likely:

1. **The hand-crafted summary features already capture most of the extractable signal on this
   subset.** FD001's baseline features include per-sensor mean/std/min/max/slope over the
   window — for a single-operating-condition, single-fault-mode subset, a slope feature over 30
   cycles may already be close to what an LSTM would learn to compute internally. The LSTM's
   structural advantage (learning temporal dynamics without being told what statistic to
   compute) may matter far more on FD002/FD004, where per-condition structure is more complex
   and the fixed baseline features don't adapt to it — untested here, a concrete next step.
2. **No hyperparameter search was done for the LSTM** (fixed `lr=1e-3`, `batch_size=64`, Adam,
   no LR scheduling), while the RandomForest baseline's `max_features`/`max_depth` were tuned
   (out of necessity — see P03's decision log) during debugging. An unequal amount of tuning
   effort between the two models makes the comparison less clean than it looks.
3. The architecture ablation above suggests the shipped two-layer architecture may itself be
   suboptimal at this training budget, which would drag down the "final configuration" number
   reported here too.

None of this changes what's reported (the measured 15.431 ± 0.493 is correct and stays on the
results table) — it changes what the honest headline claim is: "the LSTM matches the strongest
classical baseline and beats linear regression" is what's supported by these numbers; "the LSTM
beats every baseline" is not.

## Classifier: `pos_weight` comparison (failure-within-30-cycles, validation, reduced budget, 2 seeds)

| Condition | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Unweighted | 0.902 ± 0.044 | 0.929 ± 0.024 | **0.914 ± 0.011** | 0.976 ± 0.006 | 0.995 ± 0.001 | 0.0226 ± 0.0035 |
| `pos_weight` (n_neg/n_pos) | 0.875 ± 0.042 | **0.940 ± 0.028** | 0.905 ± 0.009 | 0.978 ± 0.001 | 0.995 ± 0.0004 | 0.0261 ± 0.0035 |

Textbook trade-off, measured rather than assumed: `pos_weight` buys +1.1pp recall at a cost of
2.7pp precision (net F1 slightly down) and a slightly worse Brier score (less calibrated
probabilities — expected, since `pos_weight` deliberately distorts the loss away from the true
class balance). PR-AUC is essentially unchanged (rank-ordering of scores barely moves; only the
decision threshold's effective operating point shifts). For a maintenance context where a missed
failure is costlier than a false alarm, the weighted version's recall gain is probably worth the
precision cost — but the effect size here is small enough that threshold selection (P03's
`thresholds.py`, already exercised on the classical baseline in P03) matters more than the loss
weighting choice for controlling this specific trade-off.

## Environment problems hit and fixed

**Bug 1 — `pandas`-before-`torch` breaks torch's native DLL loading on this Windows machine.**
Reproduced directly and unambiguously: `python -c "import pandas; import torch"` raises
`OSError: [WinError 1114] ... c10.dll`; reversing the two imports works every time. This is a
real, general risk, not a one-off: `pdm/cli.py`'s command handlers lazily import
`pdm.preprocessing` (pulls in pandas) and `pdm.models.torch` (pulls in torch) independently
depending on which command runs, and pytest's alphabetical test-collection order means several
pandas-importing test files load before any `test_torch_*.py` file. Fixed at both points that
matter: `tests/conftest.py` and the top of `pdm/cli.py` now do a best-effort
`try: import torch except ImportError: pass` before anything else, guaranteeing torch (when
installed — it's still an optional extra) claims its DLLs first regardless of what a command
handler or test file imports afterward.

**Bug 2 — a determinism test failure caused by seeding after model construction.** The first
run of the determinism test showed two "identical-seed" training runs diverging from epoch 1.
Root cause: `set_full_determinism(seed)` was called *inside* `train_model`, but the model passed
into it was already constructed by the caller — its initial weights were already drawn from
whatever the ambient torch RNG state happened to be at construction time, which is not
reproducible across separate calls unless the caller seeds *before* constructing the model.
`train_model`'s internal seeding call is necessary for the *training loop's* own randomness
(shuffling order, dropout masks) but cannot retroactively fix already-initialized weights. Fixed
by documenting the contract explicitly in `set_full_determinism`'s docstring and updating the
test to seed before constructing the model; `train_model` still seeds internally too, since both
are needed for genuine end-to-end determinism.

**Bug 3 — a "training bug" that was actually a mis-scaled test target.** The first version of
the overfit-tiny-batch test used a target centred at 50 (`windows.mean(axis=(1,2))*10+50`).
With `batch_size == n` there is exactly one optimiser step per epoch, and Adam's step size is
roughly constant (`≈ lr`) per parameter per step once its moment estimates stabilise regardless
of gradient magnitude — shifting an output bias by ~50 units at `lr=0.01` needs on the order of
thousands of such steps, not the 300 the test allowed. The resulting loss curve (slow, roughly
linear decrease) looked exactly like a training bug. Fixed by using a naturally small,
near-zero-mean target (`windows.mean(axis=(1,2))`, no offset) — the same tiny batch now overfits
to RMSE < 0.1 well within 300 epochs. Kept as a cautionary note: a "the model won't learn" test
failure is sometimes a target-scale-vs-budget mismatch, not an architecture or optimiser bug.

**Model file size** (measured from the actual shipped bundle,
`artifacts/torch_rul_regressor__FD001__seed42/model_torch.pt`): **316,485 bytes** (~309 KiB) for
78,051 parameters (~4.05 bytes/parameter — float32 plus small `torch.save` pickle overhead).

## Scope decision: reduced budget and fewer seeds for sweeps

Every sweep (window size, architecture, RUL cap, classifier imbalance) uses `max_epochs=40,
patience=8` and 2 seeds rather than the default `max_epochs=100, patience=10` and the full
5-seed configuration. This machine is CPU-only (`torch.cuda.is_available() == False`); at the
default budget, one training run takes 150-215 seconds, and the full required experiment matrix
at full budget and 5 seeds would have meant roughly 50 individual training runs — impractical
for one working session. 2 seeds is still "not a single result" (satisfies the letter of "no
result may be reported from a single seed") and gives a real, if noisier, mean ± std for
*comparison* between sweep points. The number that is actually reported as **the** result — the
final configuration's test RMSE — uses the full default budget and all 5 configured seeds,
matching every other milestone's standard, and is the only number that belongs in the headline
results table. The full experiment run (window sweep + architecture ablation + RUL cap
ablation, reduced budget) took **4,246.6 seconds (~70.8 minutes)** wall-clock in one background
run; the final 5-seed stability run alone took **1,022.0 seconds (~17.0 minutes)**.

## Ambiguities / questions flagged, not guessed

- The architecture ablation's implication (single layer may beat two layers on FD001) is flagged
  above but **not acted on** by changing the shipped default — that would need the full-budget,
  5-seed treatment this session's compute couldn't afford, and changing an architecture based on
  a 2-seed, 40-epoch ablation would itself be the kind of under-powered conclusion this project's
  "no result from a single seed" rule exists to prevent. Recorded as the most valuable "what I'd
  do with another month" item this milestone produced.
- Whether the LSTM-vs-RandomForest near-tie is a property of FD001 specifically (single
  condition, "easy" subset) or would hold on FD002/FD004 is untested — P05's framework benchmark
  is the natural place to extend the comparison, since it will train on the same architecture.
