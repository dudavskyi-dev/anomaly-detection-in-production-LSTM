# PdM-Sentinel — Technical Specification

This document is the source of truth for the project's design. Every decision log in
`docs/decisions/` is written against it; where an implementation detail and this spec
disagree, the disagreement is called out explicitly rather than silently resolved.

## 1. Problem statement

Industrial rotating equipment (aircraft turbofan engines, in our data) emits multivariate
sensor telemetry. We want to:

- **Forecast**: given the recent telemetry window for a unit, predict its Remaining Useful
  Life (RUL) in operating cycles, and flag units likely to fail within a horizon `W` cycles.
- **Detect**: flag anomalous telemetry windows in real time, without relying on failure labels.
- **Operate**: serve both over an API, monitor input distributions, and retrain when the
  incoming data drifts away from the training distribution.

## 2. Non-goals

- No real streaming broker (Kafka). A file-based / HTTP ingestion simulator is enough.
- No cloud deployment. Everything runs under `docker compose` locally.
- No hyperparameter search beyond small, documented sweeps.
- No claims about business impact (downtime reduction) anywhere in the repo. Only measured
  model metrics.

## 3. Architecture

```
                    ┌──────────────────────────────────────────┐
   C-MAPSS  ───────▶│ ingestion/  raw → parquet, schema-checked │
   AI4I     ───────▶│             + telemetry replay simulator  │
   NAB      ───────▶└───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │ preprocessing/                            │
                    │  per-unit RUL labelling, RUL capping      │
                    │  sensor selection, standardisation        │
                    │  sliding windows (stride, no leakage)     │
                    │  scaler fitted on TRAIN ONLY → artifact   │
                    └───────────────────┬──────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
  ┌───────▼────────┐          ┌─────────▼─────────┐         ┌─────────▼────────┐
  │ models/torch/  │          │ models/tf/        │         │ models/baseline/ │
  │  LSTM regressor│          │  LSTM regressor   │         │  Ridge, RF,      │
  │  LSTM autoenc. │          │  (identical arch) │         │  IsolationForest │
  └───────┬────────┘          └─────────┬─────────┘         └─────────┬────────┘
          └─────────────────────────────┼─────────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │ evaluation/  RMSE, MAE, NASA score,       │
                    │  PR-AUC, F1, precision/recall, ROC,       │
                    │  seed stability (5 seeds, mean ± std)     │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │ tracking/  MLflow params/metrics/artifacts│
                    │            + model registry, stages       │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │ serving/  FastAPI                         │
                    │   GET  /health     GET /metrics (Prom)    │
                    │   POST /predict/rul                       │
                    │   POST /detect/anomaly                    │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │ monitoring/  drift job (PSI + KS),        │
                    │   Prometheus scrape, Grafana dashboards,  │
                    │   retrain trigger on drift threshold      │
                    └──────────────────────────────────────────┘
```

## 4. Repository layout

```
pdm-sentinel/
├── pdm/
│   ├── config/          pydantic-settings, env-var driven, nested
│   ├── ingestion/       loaders per dataset + replay simulator
│   ├── preprocessing/   labelling, scaling, windowing
│   ├── models/
│   │   ├── torch/       PyTorch implementations
│   │   ├── tf/          TensorFlow/Keras implementations
│   │   └── baseline/    sklearn
│   ├── evaluation/      metrics, threshold selection, plots
│   ├── tracking/        MLflow wrapper
│   ├── serving/         FastAPI app
│   ├── monitoring/      drift detection, retrain trigger
│   └── cli.py           single entry point (typer)
├── artifacts/           versioned model bundles (gitignored)
├── data/                raw + processed (gitignored)
├── docs/
│   ├── SPEC.md
│   ├── DATASETS.md
│   ├── decisions/       one file per milestone, recording decisions and measured results
│   └── RESULTS.md       generated benchmark tables
├── tests/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   └── grafana/
├── .github/workflows/ci.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## 5. The model-bundle contract

Training emits a self-contained, versioned bundle. Serving loads exactly one bundle and
never recomputes anything from training data.

```
artifacts/<run_id>/
├── model_torch.pt          (or model_tf.keras)
├── scaler.json             fitted on train split ONLY
├── feature_spec.json       ordered sensor column names, window_size, stride
├── thresholds.json         RUL alert horizon W, anomaly score threshold + how derived
├── metrics.json            every measured number for this run
├── reference_stats.json    train-split feature distributions, for drift comparison
└── metadata.json           git sha, timestamp, dataset version, seed, framework, MLflow run id
```

Serving fails loudly at startup if the bundle is incomplete or `feature_spec.json` does not
match the model's expected input shape. Never silently coerce.

## 6. Modelling requirements

### 6.1 RUL regression (the "forecast failures" claim)

- Target: piecewise-linear RUL with cap `RUL_CAP = 125` cycles (standard for C-MAPSS;
  document why capping is used — degradation is not observable while the engine is healthy).
- Input: sliding window of `window_size = 30` cycles over selected sensors, stride 1.
- Architecture (identical in both frameworks): 2 stacked LSTM layers (100, 50 units),
  dropout 0.2, dense head → 1 output. Loss MSE, optimiser Adam, early stopping on val RMSE.
- **Split by engine unit, never by row.** No engine appears in both train and validation.
- Metrics: RMSE, MAE, and the asymmetric NASA scoring function (late predictions penalised
  more than early ones — implement and explain it).

### 6.2 Failure classification (the "F1 = 0.85" claim)

- Binary label: will this unit fail within `W = 30` cycles?
- Same windows, sigmoid head, BCE loss, class weighting for imbalance.
- Metrics: **PR-AUC as the headline**, plus F1, precision, recall at the chosen operating
  point, and the full precision-recall curve. Report ROC-AUC too but say in the docs why
  PR-AUC is the more honest metric here.
- Threshold selection: pick the operating point on the validation set, not the test set.
  Record both the max-F1 threshold and a recall-at-fixed-precision threshold, and explain
  which one is appropriate to ship for a maintenance use case and why.

### 6.3 Anomaly detection

- LSTM autoencoder trained on healthy windows only (early-life cycles), reconstruction-error
  scoring. Isolation Forest on the same windows as the classical baseline.
- Threshold from the training-score distribution (e.g. 99th percentile), tuned on validation.
- Evaluate on (a) held-out C-MAPSS late-life windows, and (b) NAB
  `realKnownCause/machine_temperature_system_failure.csv` as external real-world validation.
- Expect the NAB numbers to be worse. Report them anyway; document the gap.

### 6.4 Framework benchmark (the "PyTorch and TensorFlow" claim)

Both implementations must be genuinely comparable:

- Same architecture, same hyperparameters, same seeds, same splits, same preprocessed tensors.
- Report per framework: train wall-clock time, epochs to converge, peak memory, test RMSE,
  test PR-AUC, p50/p95/p99 single-sample inference latency, and model file size.
- Assert in a test that the two implementations' metrics agree within a documented tolerance.
  If they don't, that's a bug to find, and finding it is a genuinely good interview story.

## 7. Drift detection and retraining

- Drift job compares a window of recent production features against `reference_stats.json`.
- Two methods: **PSI** per feature (thresholds 0.1 / 0.25) and **two-sample KS test** with
  Bonferroni correction across features. Document why two methods rather than one.
- Drift is simulated honestly: train on C-MAPSS FD001 (one operating condition, one fault
  mode) and feed FD002/FD003 windows as "production" traffic. These genuinely have different
  distributions, so the drift signal is real, not injected synthetic noise.
- On drift above threshold: emit a Prometheus metric, write a drift report, and trigger a
  retraining run that registers a new model version in MLflow **as a candidate, not as
  production**. Promotion requires the new model to beat the incumbent on a held-out set.
  Implement that gate. Auto-promotion without a gate is a well-known footgun; say so in the docs.

## 8. Serving requirements

- FastAPI, pydantic request/response models, explicit input validation with useful errors.
- `POST /predict/rul` — body: sensor window; returns predicted RUL, failure probability,
  alert boolean, model version, latency.
- `POST /detect/anomaly` — returns anomaly score, threshold, boolean, per-sensor
  reconstruction error contributions (the explainability answer for a flagged window).
- `GET /metrics` — Prometheus format: request count, latency histogram, prediction
  distribution, anomaly rate, current drift score, loaded model version.
- Structured JSON logging with request ids.
- Graceful degradation: if the anomaly model is unavailable, RUL prediction still serves.

## 9. Quality bar

- Python 3.11+, `pyproject.toml`, ruff + black, mypy on `pdm/` (non-strict is fine).
- pytest with a fast subset (`make test-fast`) and a full suite. Minimum: preprocessing
  leakage test, windowing shape test, scaler-round-trip test, metric correctness test
  against hand-computed values, API contract tests, bundle-validation test.
- Every random seed set and recorded. Two runs with the same seed must produce the same metrics.
- `make` targets: `install`, `data`, `train`, `train-tf`, `benchmark`, `eval`, `serve`,
  `drift`, `test`, `test-fast`, `docker-up`, `lint`.

## 10. Documentation requirements

- `README.md` — what it is, architecture diagram, quickstart, results table with **real
  numbers only**, honest limitations section.
- `docs/RESULTS.md` — generated from `metrics.json` files, not hand-written.
- `docs/decisions/PXX-<name>.md` — per milestone: decisions made, alternatives rejected and
  why, problems hit, how they were fixed, measured numbers, open questions.

The decision logs are a hard requirement, not a nice-to-have — they are the primary record of
what was actually built, measured, and rejected, and the first place to look to understand why
the system looks the way it does.
