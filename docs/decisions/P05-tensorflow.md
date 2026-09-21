# P05 — TensorFlow implementation and framework benchmark decision log

Everything below is measured from a real run of `pdm benchmark` on FD001 (17 features, the same
subset and feature set P04 used), 5 seeds per framework (`settings.training.seeds = (0,1,2,3,4)`),
full training budget (`max_epochs=100, patience=10`), traceable to
`artifacts/framework_benchmark/FD001.json`. Nothing in the tables below is hand-typed —
`docs/RESULTS.md`'s own generator doesn't cover this milestone (P04 didn't wire its numbers into
it either), so this log quotes the artifact JSON directly instead.

## Parameter-count reconciliation — no, not identical on the first try

Building `pdm/models/tf/architecture.py` with a plain `keras.layers.LSTM(100)` gave **77,451**
parameters against PyTorch's **78,051** for the shipped `(100, 50)` architecture at 17 features —
a 600-parameter gap, and *not* a rounding or off-by-one difference; it reproduced exactly on
every hidden-size configuration tried.

Root cause, found by comparing per-tensor shapes side by side: `torch.nn.LSTM` keeps **two**
additive bias vectors per layer, `bias_ih` and `bias_hh`, each `(4*hidden,)`, always used as
`... + b_ih + ... + b_hh`. Keras's `LSTM` layer keeps **one** bias vector of the same shape. The
two parameterisations are mathematically equivalent in what they *can* represent — only the sum
`b_ih + b_hh` ever affects the output — but PyTorch's version has exactly `4*hidden` more trainable
scalars per layer: `4*100 = 400` for the first layer, `4*50 = 200` for the second, `600` total,
which is exactly the observed gap.

Two ways to close it were considered:
1. **Pad the Keras side** with an extra, functionally-dead bias vector just to make the parameter
   *count* match. Rejected: it would satisfy the letter of "parameter counts match" while leaving
   the architectures behaviourally different in a way that's invisible from the count alone —
   exactly the kind of number-gaming this project's decision logs exist to avoid.
2. **Reimplement the LSTM cell** with PyTorch's split-bias parameterisation on the Keras side, so
   both frameworks are parameter-count-identical *because* they compute the same function, not
   despite computing different ones.

Went with (2): `pdm/models/tf/architecture.py`'s `SplitBiasLSTMCell` is a custom
`keras.layers.Layer` (wrapped in `keras.layers.RNN`) with `kernel`, `recurrent_kernel`, `bias_ih`,
and `bias_hh`, combined exactly the way `torch.nn.LSTM` combines them, gate order (input, forget,
cell, output) matching PyTorch's convention too. Result: **78,051 parameters on both sides**,
verified for `(100, 50)` and three other hidden-size configurations in
`tests/test_tf_architecture.py::test_parameter_count_matches_pytorch_mirror_exactly`. This was a
more interesting outcome than a plain `keras.layers.LSTM` would have given: writing a custom RNN
cell from scratch to match another framework's internal parameterisation is a concrete, specific
example of "which framework was more annoying to work with" (see below), not just a passing
mention of it.

## The benchmark table

Regression (RUL, test split, mean ± std over 5 seeds), from
`artifacts/framework_benchmark/FD001.json`'s `regression` block:

| Framework | Test RMSE | Test MAE | Test NASA score | Train wall-clock (s) | Epochs to best | Peak RSS (MB) | Model size (bytes)† | Latency p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|---|---|
| PyTorch | 15.431 ± 0.493 | 11.586 ± 0.758 | 51,262 ± 1,174 | 217.8 ± 40.7 | 19.6 ± 1.5 | 1,388.7 ± 271.2 | 316,037 | 1.32 / 3.71 / 16.29 |
| TensorFlow | 15.738 ± 0.359 | 12.249 ± 0.470 | 52,583 ± 4,822 | 233.1 ± 27.4 | 17.4 ± 1.0 | 1,673.0 ± 221.2 | 314,646 | 198.5 / 238.1 / 286.6 |

†TensorFlow's model-size number is **not** what `pdm.evaluation.framework_benchmark`'s first
version measured (974,552 bytes via `model.save_weights()`) — see "Environment problems" below
for why that number was wrong and how it was fixed; 314,646 is the corrected, apples-to-apples
figure.

The two frameworks agree on test RMSE within **1.99% relative** (15.431 vs 15.738), comfortably
inside the 5% documented tolerance (`tests/test_framework_parity.py`,
`test_frameworks_agree_on_test_rmse_within_documented_tolerance`, passing). PyTorch's number
here (15.431 ± 0.493) also reproduces P04's own final-stability figure exactly
(`docs/decisions/P04-pytorch.md`: 15.431 ± 0.493) — the same seeds, same architecture, same data,
run through a completely separate code path, land on the identical number, which is itself a
useful determinism sanity check on the whole pipeline.

The one genuinely large, striking gap is **eager inference latency**: TensorFlow's p50 is
**~151x** PyTorch's (198.5ms vs 1.32ms). See "production export path" below — this gap is a
property of eager-mode TensorFlow specifically, not of the model or the framework's trained
weights, and mostly disappears once the model is exported.

Classification (failure-within-30-cycles, test split, mean ± std over 5 seeds, unweighted loss),
from the `classification` block:

| Framework | Test PR-AUC | Test ROC-AUC | Test F1 | Train wall-clock (s) | Epochs to best |
|---|---|---|---|---|---|
| PyTorch | 0.892 ± 0.021 | 0.9949 ± 0.0018 | 0.782 ± 0.018 | 95.5 ± 26.5 | 2.6 ± 0.8 |
| TensorFlow | 0.883 ± 0.016 | 0.9945 ± 0.0014 | 0.760 ± 0.029 | 118.5 ± 24.8 | 3.8 ± 1.6 |

Close on every accuracy metric (PR-AUC within 1%, F1 within 2.8%), PyTorch modestly faster to
train here too, consistent with the regression table.

## Required analysis

### Which framework trained faster, and is it the framework or the data pipeline?

TensorFlow trained ~7% slower overall (233.1s vs 217.8s mean wall-clock, regression; 118.5s vs
95.5s, classification) despite converging in *fewer* epochs on the regression task (17.4 vs
19.6) — so per-epoch, TensorFlow was doing more work, not the data pipeline waiting on it.
`compare_input_pipeline_timing` (`pdm/evaluation/framework_benchmark.py`) confirms this directly:
iterating each framework's train loader/dataset for 3 full epochs with **no model involved**, per
seed (`input_pipeline_timing` in the artifact):

| Seed | PyTorch `DataLoader` (s) | TensorFlow `tf.data` (s) |
|---|---|---|
| 0 | 0.574 | 0.193 |
| 1 | 0.566 | 0.195 |
| 2 | 0.889 | 0.397 |
| 3 | 0.486 | 0.221 |
| 4 | 0.887 | 0.370 |

`tf.data` is **2.3–2.7x faster than PyTorch's `DataLoader`** at pure iteration, every single
seed. This rules out the data pipeline as the source of TensorFlow's slightly slower *overall*
training time — if anything, the input pipeline points the other way. The actual slowdown must
come from TensorFlow's own per-step computation or graph-tracing/dispatch overhead (this project's
custom `train_step`, `tf.function` retracing across the 20 distinct model instances built across
seeds/frameworks/tasks, or eager op dispatch cost), not from feeding the model too slowly.

### Did the two reach the same accuracy? If not, what differed?

Three candidate sources of divergence were checked directly rather than assumed:

- **Initialisation.** Both frameworks use approximately the same default initialisation family
  (PyTorch's `nn.LSTM` default is a uniform `U(-1/sqrt(hidden), 1/sqrt(hidden))`; Keras's default
  is Glorot-uniform for the input kernel and orthogonal for the recurrent kernel) — these are
  *not* identical distributions, and this was not reconciled further; see "ambiguities" below.
- **Bias handling.** Reconciled exactly — see the parameter-count section above. This eliminates
  one entire axis of possible divergence between the two implementations.
- **Gradient-clipping semantics.** These genuinely differ and the difference is real, not a
  guess: Keras's built-in `optimizer(clipnorm=...)` clips **each gradient tensor independently**
  by its own norm; `torch.nn.utils.clip_grad_norm_` clips **all parameters together** by one
  global norm across the whole model. For a two-layer LSTM where per-tensor gradient norms can
  differ by an order of magnitude between layers, these are not the same operation — the
  per-tensor version can leave one layer's gradients unclipped while aggressively clipping
  another's. `pdm/models/tf/architecture.py`'s `_GlobalNormClipMixin` overrides `train_step` to
  use `tf.clip_by_global_norm` instead, matching PyTorch's semantics exactly, specifically so
  this stops being a confound in the comparison above.
- **Early-stopping criterion.** Both monitor the same quantity (validation RMSE for the
  regressor, validation BCE for the classifier) with the same patience, `min_delta`, and
  best-weight restoration — see `pdm/models/tf/train.py`'s `EarlyStopping(monitor=..., mode="min",
  patience=..., min_delta=1e-6, restore_best_weights=True)` against
  `pdm/models/torch/train.py`'s hand-rolled equivalent.

### Which was less pleasant to work with, and why (specific)

- **`model.save_weights()` silently serialises the optimizer's state too, once the model has
  been compiled and trained.** The first version of `_tf_model_size_bytes` used
  `model.save_weights()` and reported TensorFlow's model as **974,552 bytes** against PyTorch's
  316,037 — a 3.08x gap that looked like a real, reportable framework difference, until
  inspecting the saved `.weights.h5` file's internal HDF5 groups directly
  (`h5py.File(...).visititems(...)`) showed why: alongside the real `layers/...` weights sat an
  `optimizer/vars/...` group containing **Adam's two moment estimates per trainable parameter** —
  roughly 2x the model's own parameter count, serialised into the same file with no separate flag
  or warning. `torch.save(model.state_dict())` never includes optimizer state (you'd have to
  explicitly save `optimizer.state_dict()` separately to get the same information) — so the
  original comparison was quietly measuring "trained TF model + its optimizer" against "PyTorch
  model alone," not model against model. Fixed by switching to `model.get_weights()` (public API,
  guaranteed model-only) serialised through `numpy.savez` instead of `model.save_weights()` — the
  corrected number, 314,646 bytes, lines up with PyTorch's 316,037 almost exactly, as it should
  for two architectures with an identical, verified parameter count. This is exactly the kind of
  "looked like a framework difference, was actually a measurement bug" discrepancy the spec asks
  to chase down rather than report at face value, and the most interesting individual finding
  this milestone produced.
- **Getting an equal parameter count required writing a custom RNN cell in Keras.** PyTorch's
  `nn.LSTM` needed zero customisation to get the "obvious" architecture; matching it in Keras
  meant reimplementing the LSTM recurrence by hand (`SplitBiasLSTMCell`). This is squarely a
  "TensorFlow was more annoying" point, and a genuinely useful one for an interview: it forced an
  actual, from-scratch understanding of the LSTM gate equations rather than trusting a library
  default.
- **The training-loop tracker bug.** Overriding `keras.Model.train_step` for global-norm clipping
  silently broke the "loss" history: the first working version called `self.compute_loss(...)`
  but never called `self._loss_tracker.update_state(...)` — the public `compute_loss()` computes
  and validates the loss but, unlike Keras's own internal `train_step`
  (`keras.src.backend.tensorflow.trainer.TensorFlowTrainer.train_step`), does **not** update the
  tracked "loss" metric as a side effect. Every epoch of `history.history["loss"]` silently read
  back as `0.0` — a real, hour-costing bug caught only because
  `tests/test_tf_train.py::test_diverging_loss_is_detected_and_stops_training` expected the NaN
  loss to show up in history and it didn't. PyTorch's explicit training loop has no equivalent
  failure mode: there is no hidden bookkeeping step to forget, because there is no bookkeeping
  step at all — you write down what you want tracked. This is a concrete "PyTorch's explicitness
  vs Keras's convenience" story with a real bug behind it, not a stylistic preference.
- **`Model.export()`'s traced shape is not what it looks like it should be.** Calling
  `model.export(dir)` on a subclassed Keras model traces a *fixed* serving input shape from
  whatever shape the model was **last called with** — not the dynamic `(None, None, n_features)`
  shape the model was built to accept and happily runs on in eager mode. The first attempt
  exported a model whose serving signature was pinned to a `(None, 2, n_features)` window,
  because that's the throwaway shape used internally just to force variable creation
  (`self(tf.zeros((1, 2, n_features)))` in `LSTMRegressor.__init__`, so `count_parameters()` works
  before any real data is seen) — completely unrelated to the real serving shape. The eager `call`
  silently accepted any window length as always; the exported `SavedModel` silently rejected
  every window length except 2. Nothing errors at export time; it only surfaces the first time you
  feed the exported model real data, in `export_and_measure_savedmodel`, which explicitly re-calls
  the model with the real `window_size` immediately before exporting to force a correct retrace.
  This class of "worked in eager, silently wrong after export" bug has no PyTorch equivalent in
  this project, since eager PyTorch inference (measured here) is the only inference path exercised
  — TorchScript/ONNX export was in scope as optional and wasn't attempted this milestone.
- **Windows DLL loading.** Both frameworks share the exact same failure mode on this machine
  (see below) — this one is a tie, not a point for either side.

### Production export path (deliverable #6)

`export_and_measure_savedmodel` (`pdm/evaluation/framework_benchmark.py`) exports the TensorFlow
model via Keras 3's `Model.export()`, reloads it with `tf.saved_model.load`, and measures
single-sample latency through the reloaded `serving_default` signature exactly the way
`measure_inference_latency` measures eager latency — same warmup count (50), same call count
(1000), same process. From `artifacts/framework_benchmark/FD001.json`'s `savedmodel_export`:

| Path | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| TensorFlow, eager | 133.0 | 203.5 | 289.9 |
| TensorFlow, exported `SavedModel` | 3.54 | 5.90 | 8.64 |
| *(for reference)* PyTorch, eager | 1.32 | 3.71 | 16.29 |

Exporting is a **~38x speedup over eager TensorFlow** (133.0ms → 3.54ms p50) — confirming the
spec's framing of TF as the production-deployment track is measured, not marketing, *for this
specific comparison* (eager vs its own export, not vs PyTorch's best option — see "ambiguities").
It does **not**, however, close the gap with PyTorch's eager latency entirely: exported TF is
still roughly 2.7x slower than eager PyTorch (3.54ms vs 1.32ms p50). The honest, measured
statement is: "TensorFlow's SavedModel export makes TensorFlow dramatically faster than eager
TensorFlow, but is still not as fast as eager PyTorch on this architecture and hardware" — not
"TensorFlow is faster for production," which the numbers don't support without also exporting
the PyTorch side (not attempted this milestone; see ambiguities).

The exported `SavedModel` directory is 771,054 bytes on disk — 2.45x the raw weights-only size
(314,646 bytes), the expected cost of a full SavedModel bundle (computation graph, signatures,
variable checkpoints) versus a bare weights file.

## Environment problems hit and fixed

**Bug — `pandas`-before-`tensorflow` breaks TensorFlow's native DLL loading on this Windows
machine, the exact same class of bug P04 found for `pandas`-before-`torch`.** Reproduced
directly: `python -c "import pandas; import tensorflow"` raises
`ImportError: DLL load failed while importing _pywrap_tensorflow_internal: ... [Error]
Failed to load _pywrap_tensorflow_common.dll: INITIALIZATION FAILED (0x45A)` — the same
underlying Windows error code (`0x45A` = `1114`, `ERROR_DLL_INIT_FAILED`) P04 hit for
`c10.dll`. Confirmed the fix generalises: importing `tensorflow` (or `torch`) before `pandas`
avoids it regardless of the other import's position, but importing `pandas` first breaks
whichever of `torch`/`tensorflow` is imported afterwards. Fixed the same way P04 fixed it, at
the same two points: `tests/conftest.py` and the top of `pdm/cli.py` now also do a best-effort
`try: import tensorflow except ImportError: pass`, right after the existing `torch` import
attempt and before anything else in the module loads.

**Environment note — TensorFlow's pip package hits Windows' `MAX_PATH` (260 characters) during
install.** `pip install tensorflow` into the global (Windows Store) Python's `site-packages`
failed with `OSError: [Errno 2] No such file or directory: '...gcp_service_account_identity_
credentials.h'` — not a missing file, but a 146-character *relative* path inside the wheel
landing on top of an already-139-character site-packages prefix, totalling 285 characters. Fixed
without touching the registry (`LongPathsEnabled`, which needs admin rights and a
system-wide change) by installing into the project's existing short-path `.venv`
(`D:\pythonProjects\Anomaly Detection\.venv\Lib\site-packages\`, 60 characters) instead — total
206 characters, comfortably under the limit. `torch` was already installed in that same `.venv`
from P04; TensorFlow now lives there too.

## Verdict: which for research, which for production

**Research/iteration: PyTorch.** Every piece of training-loop bookkeeping (loss tracking,
gradient clipping, divergence handling) is a line you wrote yourself and can read back —
`pdm/models/torch/train.py`'s `train_model` is ~90 lines a newcomer can read start to finish.
The Keras equivalent needed a custom `train_step`, and doing that wrong produced a real, silent
bug (the loss-tracker miss, above) with no error message pointing at the cause — you only notice
because a downstream number looks wrong. Two separate real bugs were caused by *implicit*
framework behaviour this milestone (the loss-tracker miss, and the optimizer-state-in-weights-file
mismeasurement); zero were caused by PyTorch's explicitness, because there was nothing implicit
to get wrong. That asymmetry, not a stylistic preference, is the basis for this verdict.

**Production: TensorFlow, with a specific and now-measured reason, not the general "TF is for
production" folklore.** The concrete, measured argument is the export path: `Model.export()` →
`tf.saved_model.load()` → `infer.signatures["serving_default"]` turns a 133ms eager call into a
3.54ms one, a real and substantial win with a two-line reload story
(`export_and_measure_savedmodel`). PyTorch has an equivalent path (TorchScript/ONNX) that was in
scope as optional and **not exercised here** — so the honest framing is "TensorFlow's own export
path is worth using if you deploy with TensorFlow," not "TensorFlow beats PyTorch in production":
on the numbers actually measured, exported TF (3.54ms) is still ~2.7x slower than eager PyTorch
(1.32ms), so the fair production comparison this milestone can make is incomplete without also
measuring PyTorch's own export path — flagged below, not glossed over.

## Ambiguities / questions flagged, not guessed

- Weight initialisation distributions were not reconciled between frameworks (PyTorch's default
  `nn.LSTM` uniform init vs Keras's Glorot-uniform/orthogonal split). Both are standard,
  reasonable defaults, and reconciling them would mean *also* writing a custom initializer to
  reproduce PyTorch's exact scheme in Keras — judged out of scope for what the parity test needs
  (a 5%-relative-RMSE agreement, not bit-identical training trajectories), but flagged rather than
  silently assumed identical.
- TorchScript/ONNX export (mentioned as optional in the spec) was not attempted, so "is
  TensorFlow's production path faster than PyTorch's" is only ever compared against **eager**
  PyTorch here, not PyTorch's own optimised export path. The honest claim this milestone supports
  is narrower: "TensorFlow's SavedModel export is/isn't faster than eager PyTorch," not "than
  PyTorch's best production option."
- The exact mechanism behind TensorFlow's ~7% slower wall-clock training time (despite a *faster*
  standalone data pipeline, per the input-pipeline table above) was narrowed to "TensorFlow's own
  per-step computation or graph-tracing overhead" but not root-caused further — distinguishing
  "the custom `train_step`'s Python-level overhead," "`tf.function` retracing cost across the 20
  distinct model instances built in this run," and "eager TF op dispatch being inherently slower
  than PyTorch's for this op mix" would need per-step profiling (e.g. `tf.profiler`) not done this
  milestone. The measurement (TF is slower, the input pipeline is not why) is solid; the
  attribution among these three candidates is not.
- PyTorch's eager-latency **p99** has unusually high variance (16.29ms ± 26.72ms — one seed's run
  alone hit 69.7ms while the other four stayed under 5ms; see the `latency_p99_ms.values` array in
  the artifact). This looks like an intermittent scheduling/OS-noise effect on this Windows
  machine (background process contention, a GC pause, or CPU frequency scaling) rather than a
  property of the model or framework, since it appears in exactly one of five otherwise-identical
  runs — flagged rather than smoothed over by only reporting the mean, but not chased down further
  since it doesn't change the ~150x eager-latency gap's headline direction either way.
