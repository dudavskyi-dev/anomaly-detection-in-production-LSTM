# P08 — FastAPI serving and containerisation decision log

## The bundle-extension problem: architecture wasn't self-describing before this milestone

The spec's bundle contract (§5) never previously needed to answer "what were `hidden_sizes` and
`dropout` for *this* run" — every training path (P04-P07) reconstructed a model with whatever
`settings.model.lstm_hidden_sizes`/`dropout` the *current* config said, and that always happened
to match because nothing had changed those defaults between training and re-use. Serving is the
first consumer that must reconstruct a model **in a different process, possibly much later,
possibly after the config's defaults have moved on** — exactly the scenario where "assume
current config still matches" quietly breaks. `metadata.json` gained a new `architecture` field
(`hidden_sizes`/`dropout` for the regressor, `latent_dim` for the autoencoder) this milestone,
written by every bundle-saving call site (`pdm.models.bundle.write_metadata`,
`pdm.models.torch/tf.bundle.save_*_bundle`, and every CLI command that calls them). This was not
a hypothetical concern: P06's own `torch_lstm_ae__FD001__seed42`/`tf_lstm_ae__FD001__seed42`
bundles were trained with `latent_dim=2` (the ablation's empirical winner), but
`settings.model.ae_bottleneck_dim` defaults to `8` — reconstructing either bundle from current
config defaults alone would have built the *wrong-shaped* model and failed to load its weights.
Caught before it ever reached a real request, by `pdm.serving.bundle`'s own parameter-count
cross-check against `metadata.json` (see below) — and fixed at the source by regenerating both
P06 bundles with the new `architecture` field once the fix landed, rather than special-casing
serving to "know" latent_dim=2 for that one bundle.

## Why the training/serving split is enforced at the package level

`pdm.serving` never imports a training-*orchestration* module
(`pdm.preprocessing.pipeline`, `pdm.models.baseline.runner`,
`pdm.models.torch.experiments`/`anomaly_experiments`, `pdm.evaluation.framework_benchmark`, or
anything under `pdm.ingestion`) — checked structurally by
`tests/test_serving_import_graph.py`, which imports `pdm.serving.app` in a fresh subprocess and
asserts none of those modules ever entered `sys.modules`. The reasoning: those modules assume a
training *environment* — real datasets on disk, non-deterministic hyperparameter sweeps, a
willingness to take minutes to run — none of which a request handler can assume or afford.
Coupling serving to them would mean a change to, say, the C-MAPSS ingestion schema could
somehow break the inference API, for no reason connected to what the API actually does (load a
bundle, run a forward pass). Serving *does* transitively import each framework's
architecture/training-loop module (e.g. `pdm.models.torch.train`, reused for the LSTM
autoencoder's inference-only scoring functions that happen to live in the same file as the
training loop) — a deliberate, narrower line than "nothing training-related at all," chosen
because those functions are pure inference math with no orchestration assumptions, not because
drawing the line there was the path of least resistance.

**What training/serving skew would look like if the scaler weren't shipped in the bundle:**
concretely, if serving recomputed a `StandardScaler` from *its own* view of "current" data
(or, worse, skipped scaling because someone assumed the bundle's model was trained on raw
values) instead of loading the exact `mean_`/`std_` arrays `scaler.json` records, every
prediction would be silently wrong in a way that's easy to miss in testing and expensive to
diagnose in production: the model would receive inputs shifted and scaled differently than
whatever it was actually trained on, degrading accuracy without throwing any error at all — no
NaN, no crash, no obviously-wrong shape, just quietly worse predictions that look plausible in
isolation. This is exactly the failure mode `pdm.serving.bundle` is built to make structurally
impossible: `RulBundle.scale()`/`AnomalyBundle.scale()` are the *only* place a raw window gets
transformed before hitting the model, and they always read `scaler.mean_`/`scaler.std_` from the
loaded `scaler.json` — there is no code path that recomputes or approximates it.

## Measured latency: p50/p95/p99, batch vs. single

Measured against the actual `docker/Dockerfile` image (`pdm-sentinel:latest`), run as a real
container (`docker run`, port-mapped, real bundles mounted read-only from `../artifacts`) and
hit over the network from the host via `urllib` — not `TestClient`, which shares a process with
the server and skips the socket/HTTP stack entirely. 20 warmup requests then 200 measured calls
per endpoint, single-threaded (this is per-request latency, not throughput under concurrency —
see `tests/test_serving_concurrency.py` for the concurrent-load contract test):

| Endpoint | p50 | p95 | p99 | min | max |
|---|---|---|---|---|---|
| `/predict/rul` (single window) | 14.5ms | 16.1ms | 17.5ms | 5.1ms | 26.2ms |
| `/detect/anomaly` (single window) | 15.4ms | 17.0ms | 18.3ms | 5.4ms | 25.6ms |
| `/predict/batch` (16 windows) | 18.2ms | 20.3ms | 27.8ms | 8.1ms | 28.1ms |

Batching 16 windows into one request costs roughly +3.7ms at p50 over a single window — the
per-window marginal cost is small because the dominant fixed cost per request is Python/FastAPI
request handling and JSON (de)serialisation, not the forward pass itself (a 1-layer LSTM over a
short window is cheap on CPU). This means batching is worth doing when a caller has many windows
to score at once purely to amortise HTTP overhead, not because the model itself is a bottleneck.
All three endpoints comfortably clear the 200ms regression-ceiling budget
`tests/test_serving_latency.py` checks in-process — the real, containerized numbers are roughly
10x tighter than that budget, confirming the test's ceiling was deliberately generous rather than
a number this milestone was quietly failing to hit. Container memory at idle after warmup:
~218MiB RSS (`docker stats`), all CPU-only PyTorch — no GPU memory involved.

## Failure mode → behaviour

| Failure mode | Behaviour | Verified by |
|---|---|---|
| Corrupt/missing RUL bundle file | Service refuses to start (`BundleValidationError` at startup) | `tests/test_serving_startup.py` |
| RUL model weights file corrupt | Service refuses to start | `tests/test_serving_startup.py` |
| `metadata.json` parameter count disagrees with the reconstructed model | Service refuses to start | `tests/test_serving_bundle.py` |
| Anomaly bundle missing/unconfigured | Service starts; `/detect/anomaly` returns 503 with a reason; `/predict/rul` unaffected | `tests/test_serving_app.py`, `tests/test_serving_startup.py` |
| Anomaly bundle present but broken | Same as above — never blocks startup | `tests/test_serving_startup.py` |
| Wrong feature order | `422`, names the field, shows expected vs. received order | `tests/test_serving_app.py` |
| Wrong window length / feature count | `422`, names the field and the expected shape | `tests/test_serving_app.py` |
| NaN/Inf in the window | `422`, names the exact offending cell | `tests/test_serving_app.py` |
| Empty window / empty request body | `422` | `tests/test_serving_app.py` |
| Batch larger than `PDM__SERVING__MAX_BATCH_SIZE` | `422` | `tests/test_serving_app.py` |

## Image size: before and after multi-staging

Both images built from the real project, real `pyproject.toml` extras, on the same machine
(`docker images` / `docker system df -v` for the breakdown):

| Image | Total size | Unique (non-base) layers |
|---|---|---|
| `docker/Dockerfile.naive-single-stage` (single `FROM`, no `--prefix` split) | 2.26GB | 2.115GB |
| `docker/Dockerfile` (multi-stage: builder discarded, only `/install` copied forward) | 2.25GB | 2.096GB |

Multi-staging saved only ~19MB (~1%) here — smaller than the "multi-stage cuts image size in
half" story usually told about it, and worth being honest about rather than overselling. The
reason: that story's savings normally come from discarding a C/C++ build toolchain
(`build-essential`, `gcc`, headers) needed to compile native extensions, plus pip's download
cache. Neither applies much here — every dependency in `.[torch,serving]` (torch, numpy,
fastapi, uvicorn, pydantic, prometheus-client) ships prebuilt `manylinux` wheels, so
`python:3.11-slim` never needs a compiler at all, and both Dockerfiles already pass
`--no-cache-dir` so neither accumulates a pip cache. What's actually being saved by the
multi-stage split here is smaller and more structural: pip/setuptools/wheel themselves and any
transient build metadata never reach the final image, shrinking the attack surface (no package
manager tooling available inside the running container) rather than the disk footprint. The
real weight in both images — roughly 2GB of the ~2.25GB total — is `torch` itself and its
(non-CUDA, since the Dockerfile now installs the CPU-only wheel from
`download.pytorch.org/whl/cpu`) transitive dependencies; that is the actual lever for a smaller
image, not the build-stage split. Confirmed the CPU-only wheel took effect inside the built
image: `docker run pdm-sentinel:latest python -c "import torch; print(torch.__version__,
torch.cuda.is_available())"` → `2.14.0+cpu False`.

**A genuine finding this produced:** the first build attempts (before this fix) failed outright,
not just "were large" — `pip install ".[torch,serving]"` resolves the default PyPI `torch` wheel,
which pulls in ~2GB of NVIDIA CUDA runtime wheels (`nvidia_cusolver`, `nvidia_cusparse`, etc.) as
transitive dependencies even though this container has no GPU and never touches CUDA. Those huge
downloads were what triggered intermittent DNS/connection failures during the build
(`Failed to establish a new connection: [Errno -2] Name or service not known` partway through a
~200MB wheel). Fixed by installing `torch~=2.14.0` from `https://download.pytorch.org/whl/cpu`
*before* the extras install, in both Dockerfiles — pip sees the version constraint already
satisfied and doesn't reinstall a CUDA-bundled build on top of it. This is a correctness fix
(the build was failing, not just slow) that happened to also be the real lever on image size,
which is why both Dockerfiles carry the same fix rather than only the one being compared against.

## A genuine finding: strict JSON encoders can't send NaN at all

Testing "NaN rejection" turned out to be less straightforward than expected: `httpx`'s (and
most spec-compliant JSON libraries') encoder refuses to serialise `float('nan')` in the first
place — `ValueError: Out of range float values are not JSON compliant` — raised **client-side**,
before a request is even sent. A real client using a standards-compliant JSON encoder cannot
construct the malformed request this validator exists to catch. Both the contract test
(`tests/test_serving_app.py::test_predict_rul_nan_in_window_returns_422`) and the manual smoke
test that first surfaced this had to send the request body as raw bytes with a literal `NaN`
token (which Python's own `json.dumps`/`json.loads` — and FastAPI's parser, built on the same
stdlib — accept as a non-standard extension by default) to exercise the validator at all. The
practical implication: this validator's real value isn't "a well-behaved client sent NaN," it's
guarding against **permissive clients and internal producers** — a sensor gateway using a
looser encoder, a numeric field silently becoming NaN upstream (a division by zero, an empty
reading defaulting to float conversion of `""`), or a service replaying previously-logged,
already-malformed payloads. Worth stating in an interview: the validator earns its keep from the
*second* category of caller, not the first.

## A genuine finding: `.dockerignore` excluding `README.md` breaks the build

The first version of `.dockerignore` excluded `README.md` under "docs aren't needed at
runtime" — true, but irrelevant: `pyproject.toml` declares `readme = "README.md"`, and
`pip install .` (run *during the build*, not at runtime) reads that file to build the package's
metadata. The build failed immediately: `failed to compute cache key: ... "/README.md": not
found`. Fixed by keeping `README.md` un-ignored and documenting why directly in
`.dockerignore` itself, so the next person tempted to "clean up" the ignore list sees the
reason before repeating the mistake. A small, concrete example of the difference between "not
needed while the container is running" and "not needed to build the container" — they're not
the same list.

## Scope decisions

- **The served RUL model's "failure probability within W" is derived, not classifier-output.**
  No training path in this project bundles a companion failure classifier alongside the
  regressor (P04's `LSTMClassifier` exists and was trained during P04's own ablations, but was
  never saved into a shippable bundle). Rather than invent a heuristic or add a whole second
  model-training path this late in the project, `/predict/rul` computes
  `P(true RUL <= W)` from a **documented statistical approximation**: a Gaussian residual model
  using the bundle's own measured test RMSE (`metrics.json`, a real, measured number) as the
  standard deviation, `Phi((W - predicted_rul) / rmse)`. This is honestly weaker than a
  calibrated classifier — P04's own residual plots showed residuals widen and skew near the RUL
  cap, violating the constant-Gaussian-noise assumption — and that limitation is stated here
  rather than presented as more rigorous than it is. `alert` is `predicted_rul <= W` directly
  (equivalent to `failure_probability >= 0.5` by construction, not a separate independent rule).
- **The Docker image serves PyTorch bundles only** (`.[torch,serving]`, not `.[tf,serving]`) to
  keep the shipped image's size and build time down — TensorFlow support exists in the codebase
  and `pdm.serving.bundle` dispatches on `metadata.json`'s `framework` field, so serving a
  TensorFlow bundle only needs rebuilding the image with the other extras, not a code change.
- **Docker Compose's `api` service was actually run this session, not just config-validated**:
  `docker compose -f docker/docker-compose.yml up -d api` built the real image, started the
  container against real bundles mounted from `../artifacts`, reached `healthy` on its own
  healthcheck, and answered `GET /health` over the mapped port before being torn down — the
  latency and image-size numbers above come from that same image, not a separate paper exercise.
