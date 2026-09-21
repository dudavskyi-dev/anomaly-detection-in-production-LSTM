# PdM-Sentinel

[![CI](https://github.com/dudavskyi-dev/anomaly-detection-in-production-LSTM/actions/workflows/ci.yml/badge.svg)](https://github.com/dudavskyi-dev/anomaly-detection-in-production-LSTM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end predictive maintenance and anomaly detection system for industrial rotating
equipment. It forecasts remaining useful life (RUL) and imminent failure from multivariate
sensor telemetry, flags anomalous behaviour without relying on failure labels, and serves both
over an API with drift monitoring and gated automated retraining.

Built against NASA C-MAPSS turbofan degradation data, the AI4I 2020 predictive-maintenance
dataset, and the Numenta Anomaly Benchmark. See [`docs/DATASETS.md`](docs/DATASETS.md) for
details and licensing, and [`docs/SPEC.md`](docs/SPEC.md) for the full technical specification.

## Architecture

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

## Install

Requires Python 3.11+.

```bash
make install        # core + dev tooling
make test-fast       # quick tests
make lint            # ruff + black --check + mypy
```

Heavier optional dependency groups (`torch`, `tf`, `serving`, `tracking`) are installed by the
milestone that first needs them — see `pyproject.toml`'s `[project.optional-dependencies]` — to
keep the base install fast. `pip install -e ".[all]"` installs everything at once.

## Status

- [x] P00 — Bootstrap
- [x] P01 — Data ingestion
- [x] P02 — Preprocessing
- [x] P03 — Baselines & evaluation framework
- [x] P04 — PyTorch LSTM
- [x] P05 — TensorFlow benchmark
- [x] P06 — Anomaly detection
- [x] P07 — MLflow tracking & registry
- [x] P08 — FastAPI serving & Docker
- [x] P09 — Drift detection & retraining
- [ ] P10 — Prometheus & Grafana monitoring
- [x] P11 — CI/CD & final documentation

## Results

Headline numbers, mean ± std over 5 seeds. Full tables (raw, auto-generated, nothing hand-typed)
in [`docs/RESULTS.md`](docs/RESULTS.md); the narrative walkthrough — training curves, PR curves,
the PyTorch/TensorFlow framework comparison, a real drift example — is in
[`docs/CASE_STUDY.md`](docs/CASE_STUDY.md).

| Task | Metric | Result |
|---|---|---|
| RUL regression (FD001, PyTorch LSTM) | Test RMSE | 15.431 ± 0.493 (vs. 15.365 ± 0.284 Random Forest baseline) |
| RUL regression (FD001, TensorFlow mirror) | Test RMSE | 15.738 ± 0.359 (within 2% of PyTorch) |
| Anomaly detection (fused AE + Isolation Forest) | Test F1 | 0.648 ± 0.007 (vs. 0.315 best single detector) |
| Drift detection (FD002 vs. FD001-trained model) | Aggregate PSI | 3.41 (threshold 0.25) — correctly flagged |

## Serving

```bash
pip install -e ".[torch,serving]"
PDM__SERVING__BUNDLE_DIR=artifacts/torch_rul_regressor__FD001__seed42 \
PDM__SERVING__ANOMALY_BUNDLE_DIR=artifacts/torch_lstm_ae__FD001__seed42 \
make serve
```

Or via Docker Compose (builds the image, starts the API alongside the MLflow UI):

```bash
make docker-up
```

Once the API is up (`http://localhost:8000`), get a feature order and window size from the
bundle you pointed it at, then predict:

```bash
curl localhost:8000/health

# feature_spec.json in the bundle you're serving tells you the exact feature order/window size
curl -s -X POST localhost:8000/predict/rul \
  -H 'Content-Type: application/json' \
  -d @examples/predict_rul_request.json

curl -s -X POST localhost:8000/detect/anomaly \
  -H 'Content-Type: application/json' \
  -d @examples/detect_anomaly_request.json
```

`/predict/rul` returns predicted RUL, a Gaussian-residual-derived failure probability within the
configured horizon `W`, an alert boolean, model version, and server-side latency — see
`docs/decisions/P08-serving.md` for what that probability does and doesn't mean statistically.
`/detect/anomaly` returns a reconstruction-error score, the shipped threshold, a boolean, and a
per-sensor breakdown of the reconstruction error (the explainability answer). If no anomaly
bundle is configured or it fails to load, `/detect/anomaly` returns a `503` with a clear reason
while `/predict/rul` keeps working — verified in `tests/test_serving_startup.py`.

## Monitoring: drift detection and drift-triggered retraining

```bash
pip install -e ".[torch,tracking,monitoring]"
pdm drift check --production-subset FD002   # or FD003; replays real "production" traffic
pdm drift report --result reports/drift/FD002_test/drift_result.json
```

`pdm drift check` compares incoming traffic against a bundle's `reference_stats.json` (per-feature
histograms and a reference sample frozen at training time) using PSI and a
Benjamini-Hochberg-corrected two-sample KS test, measures the model's actual RMSE/F1 degradation
on that traffic, writes a markdown report with per-feature before/after plots, emits Prometheus
metrics (`artifacts/monitoring/drift_metrics.prom`, in the textfile-collector format — see
`docs/decisions/P09-drift.md`), and — only above threshold and outside cooldown — launches a
retraining run registered as a **candidate** model version in `Staging`. It never promotes: that
stays exclusively `pdm registry promote`'s job, gated by the same P07 promotion margin as every
other candidate. `make drift` runs both subsets against the current default bundle.

## Limitations

- **Retraining on drifted data can be exactly the wrong move.** If drift reflects a failing
  sensor, or the genuine onset of the fault mode this system exists to catch, training on it
  teaches the model that the anomaly is normal — a confident, unalarmed, wrong model is worse
  than a stale one. This project's drift-triggered retrain never trains on the raw drifted
  traffic (it retrains on the same trusted, already-labelled subset the current model was trained
  on) and never auto-promotes the result (`pdm.monitoring.trigger` never imports the promotion
  gate) — see `docs/decisions/P09-drift.md` for the full argument. A real deployment would still
  need a human decision (and a relabelling process) before drifted traffic is ever trusted as
  training data at all; this project deliberately does not attempt to automate that judgment.
- **Input drift (PSI/KS) is detectable without labels; concept drift and label drift are not.**
  Prediction drift (comparing the model's *output* distribution) is used as a partial, indirect
  proxy for concept drift, but it's a symptom check, not a diagnosis. The honest RMSE/F1
  degradation numbers in `docs/decisions/P09-drift.md` are only possible because C-MAPSS ships
  labels for its "production" (FD002/FD003) data too — a real deployment has no such ground truth
  until an engine actually fails, and would need a delayed, retrospective evaluation pipeline to
  ever measure degradation directly at all.
- **The served RUL model's "failure probability within W" is a documented statistical
  approximation** (a Gaussian residual model, not a calibrated classifier) — see
  `docs/decisions/P08-serving.md`.
