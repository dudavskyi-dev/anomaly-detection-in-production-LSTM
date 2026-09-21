# P10 — Prometheus and Grafana monitoring decision log

## The metric list: what question each one answers, and what you'd do if it went red

| Metric | Type | Question it answers | If it went red |
|---|---|---|---|
| `pdm_requests_total{endpoint,status}` | Counter | Is the service getting traffic, and how much of it fails? | Check the `status` breakdown first — a spike in one status code (422 vs 503 vs 500) points at a different cause (bad client payloads vs. anomaly bundle down vs. an unhandled exception) than "the service is just failing." |
| `pdm_request_latency_seconds{endpoint}` | Histogram | Is the service still fast? | Compare against `docs/decisions/P08-serving.md`'s real baseline (p99 never exceeded 28ms) — if p95 is sustained above the 50ms alert budget, check host CPU contention first (this is a CPU-only PyTorch model; a noisy neighbor process directly steals inference time) before suspecting the model itself changed. |
| `pdm_predicted_rul` | Histogram | What does the model currently believe about fleet health? | A sudden shift in the distribution (e.g. p50 dropping sharply) is a leading indicator worth cross-checking against `pdm_drift_score_aggregate` *before* assuming a real fleet-wide failure wave — could be input drift confusing the model instead. |
| `pdm_anomaly_score` | Histogram | How close is "normal" traffic sitting to the alarm line? | If the whole distribution is creeping up toward `0.8522` (the shipped threshold) without crossing it yet, that's an early-warning signal `PdmHighAnomalyRate` (rate-based) won't catch until it's already over threshold. |
| `pdm_anomalies_detected_total` | Counter | How many windows has the anomaly detector actually flagged? | Paired with `pdm_requests_total{endpoint="/detect/anomaly"}` to get a *rate* — the raw count alone means nothing without knowing the traffic volume it came from (see `PdmHighAnomalyRate`, below). |
| `pdm_alerts_total{severity}` | Counter | How many RUL-based failure alerts is the service raising, and how urgent? | A jump in `critical` (not just `warning`) specifically means multiple engines are predicted well past the danger point — worth an immediate look, not just a dashboard glance. |
| `pdm_model_info{version,framework,bundle_id}` | Info (gauge, value 1) | Which exact model is currently answering requests? | Not something that "goes red" — it's the join key: when any other panel looks wrong, this says exactly which bundle produced it, so the investigation starts from "is this bundle itself known-bad" rather than guessing. |
| `pdm_last_retrain_timestamp{version}` / `pdm_model_age_seconds{version}` | Gauge | How stale is the currently-served model? | Above 30 days (`PdmModelStale`) means either the drift-monitor job has stopped running, or drift has genuinely stayed under threshold for a month straight — check the drift-monitor container's logs first to rule out the former before concluding the latter. |
| `pdm_drift_score{feature}` / `pdm_drift_score_aggregate` | Gauge | Does incoming traffic still look like what the model was trained on? | The aggregate crossing 0.25 (`PdmSignificantDataDrift`) says "something changed"; the per-feature panel says *which sensor*, which is the difference between "investigate the whole fleet" and "go look at sensor 11 specifically." |

## Counter vs. gauge vs. histogram vs. summary — when, and why

- **Counter** for anything that only ever goes up between restarts and is meaningful as a
  *rate* (`rate(...[5m])`), never as a raw value on its own: `pdm_requests_total`,
  `pdm_anomalies_detected_total`, `pdm_alerts_total`. None of these answer a useful question as
  an absolute number ("14,392 requests since when?") — they answer one as a rate ("how many
  requests/sec, right now").
- **Gauge** for anything that can go up *or* down and is meaningful as a point-in-time value:
  `pdm_drift_score`/`pdm_drift_score_aggregate` (a PSI value, not a count of anything),
  `pdm_model_age_seconds`/`pdm_last_retrain_timestamp` (both decrease to 0 the instant a new
  model is promoted and starts serving), `pdm_process_start_time_seconds`. `pdm_model_info` is
  technically an `Info` (a thin wrapper that's a gauge fixed at `1`, with the interesting content
  living entirely in its labels) — used here because the *value* is never the point, the
  *identity* (version/framework/bundle_id) is, which is exactly what `Info` is for.
- **Histogram** for anything where the question is "what's the distribution/percentile," not
  just "what's the average": `pdm_request_latency_seconds`, `pdm_predicted_rul`,
  `pdm_anomaly_score`. A Prometheus histogram (cumulative `_bucket` counters +
  `histogram_quantile()` at query time) was chosen over a `Summary` specifically because
  Summaries compute quantiles *client-side, per process*, which cannot be aggregated across
  multiple replicas of the API (their per-quantile streams can't be meaningfully combined) —
  Histograms can be, via `sum by (le) (...)` before `histogram_quantile()`, which matters even
  for this project's single-replica dev setup because it's the only one of the two that
  generalises to a real multi-replica deployment without a redesign. Client-side quantile
  computation (a Summary's whole reason to exist) also isn't needed here: this project doesn't
  need a *guaranteed-accurate* single-process quantile badly enough to give up cross-instance
  aggregation for it.
- **No `Summary` is used anywhere in this project.**

## Histogram buckets: chosen from measurements, not library defaults

- **`pdm_request_latency_seconds`**: `(0.0025, 0.005, ..., 0.05, 0.1, 0.25, 0.5, 1.0)` seconds,
  dense (2.5ms steps) between 2.5ms and 50ms. Source: `docs/decisions/P08-serving.md`'s real,
  containerized latency measurements — p50 ~14.5ms, p95 16-20ms, p99 17.5-27.8ms, max ~28ms,
  across every endpoint this project has ever actually measured. Every real observation this
  project has produced falls inside the dense region; the coarser buckets out to 1s exist only
  to still classify a genuine outlier (a cold start, a GC pause, a real regression) without
  spending resolution on a range nothing real has ever hit. The default `prometheus_client`
  buckets (`.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0`) would have
  put every single real measurement this project has taken into the *first three* buckets —
  useless resolution for a service this fast.
- **`pdm_anomaly_score`**: `(0.1, 0.2, ..., 0.8, 0.8522, 0.9, 1.0, 1.5, 2.0, 5.0)`, straddling the
  shipped AE bundle's actual threshold (`0.8522...`,
  `artifacts/torch_lstm_ae__FD001__seed42/thresholds.json`) with a bucket edge placed exactly on
  it. The question a dashboard needs answered is "how close is normal traffic sitting to the
  alarm line," which needs resolution *around 0.85*, not evenly-spaced buckets across whatever
  the largest score ever seen happens to be — P09's own FD002 drift scenario measured anomaly
  scores orders of magnitude past threshold under a genuine scaler mismatch, and wasting bucket
  resolution trying to distinguish 40 from 80 would come at the cost of the resolution that
  actually matters, right around 0.85.
- **`pdm_predicted_rul`**: `(0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300)` — unchanged from
  P08, since `settings.data.rul_cap = 125` already bounds nearly all real predictions into
  `[0, 130]`; the wider buckets past 150 exist only to still classify a healthy-engine
  prediction near the cap or a rare over-cap output without needing their own dense resolution.

## Alert thresholds: derived, not guessed

Every threshold in `docker/prometheus/alerts.yml` cites the specific measurement or existing
project setting it came from (also repeated as an inline YAML comment on each rule, so the
justification travels with the rule itself, not just this log):

- **Anomaly rate > 5% for 10m**: the shipped threshold is defined as the 99th percentile of
  *healthy* training scores (P06's `percentile-99-of-healthy-training-scores` method) — by
  construction, ~1% of genuinely healthy traffic crosses it on sampling noise alone. 5% is 5x
  that background rate, chosen to be clearly outside "just noise" without waiting for every
  request to alarm before paging anyone.
- **Aggregate drift score > 0.25**: this is `settings.monitoring.psi_significant_threshold`,
  and P09 didn't just adopt the PSI literature's convention for it — it *measured* the gap
  this threshold sits in on this project's own data: 0.017 on the genuine FD001 no-drift
  holdout vs. 0.77 (FD003) and 3.41 (FD002) on genuinely drifted traffic
  (`docs/decisions/P09-drift.md`). The alert threshold is empirically validated on this
  project's own numbers, not just a borrowed constant.
- **p95 latency > 50ms for 5m**: ~1.8x the highest p99 this project has ever actually measured
  (27.8ms, P08). Tight enough to catch a genuine regression, loose enough that ordinary jitter
  at this traffic volume won't false-page.
- **Model age > 30 days**: `settings.monitoring.retrain_cooldown_seconds` (1 day) is a *minimum
  gap* between retrain triggers, not a target cadence, so it can't be reused directly as a
  staleness bound. 30 days is chosen as "even in the most conservative realistic case (drift
  stays under threshold for a while, so nothing triggers), the drift-monitor job is still
  checking hourly (`settings.monitoring.drift_check_interval_seconds`) — a full month with zero
  retraining activity at all means either that scheduled job has stopped running (an operational
  failure worth paging on by itself) or the model has been left stale by neglect."

No Alertmanager is deployed alongside Prometheus for this milestone — there's no "route this
alert to a pager/Slack channel" requirement in scope, and firing/pending rules are fully visible
in Prometheus's own `/alerts` page without one. Adding an Alertmanager purely to have somewhere
to route alerts nobody asked to be routed anywhere would be scope creep, not thoroughness.

## Monitoring the service vs. monitoring the model — why both dashboards exist

These answer genuinely different questions, and a system that only has one of them has a real
blind spot:

- **The service dashboard** answers "is the software working": is it up, is it fast, is it
  returning errors. A service can be perfectly healthy by every metric on this dashboard —
  200s, low latency, no crashes — while returning **confidently wrong predictions** the whole
  time. Nothing about request rate, error rate, or latency has any way to notice that the model
  itself has stopped being accurate; those three numbers are about the HTTP layer, not the
  statistical relationship the model is supposed to have learned.
- **The model dashboard** answers "is the *prediction* still trustworthy": does incoming traffic
  still look like training data (`pdm_drift_score`), does the model's own output distribution
  still look normal, is the anomaly rate/alert rate consistent with a healthy fleet, how long
  since this model was last validated against fresh data. This is where P09's entire concern —
  a model can rot silently while every service metric stays green — actually gets caught. A
  service dashboard would never flag FD002-style drift at all: every one of P09's real FD002
  measurements (aggregate PSI 3.41, RUL RMSE 2.35x worse) happened while the API itself kept
  returning fast, well-formed 200 responses the entire time. The predictions inside those 200s
  were just wrong.

This is also a genuinely common interview question for exactly this reason: it's easy to build
a system that pages someone the instant the *server* breaks, and much easier to accidentally
ship one that stays silent for months while the *model* quietly stops being right — because
nothing about a broken model looks like a broken service from the HTTP layer's point of view.
Two dashboards, backed by two different classes of metric (request/latency/error vs.
prediction-distribution/drift/staleness), is what makes both failure modes visible instead of
just one.

## Bridging two independent processes without a shared import

`pdm_drift_score{feature}`/`pdm_drift_score_aggregate` are computed by `pdm.monitoring` — a
periodic batch job, not the live API — but P10's spec explicitly names them as metrics the
system exposes. The API cannot import `pdm.monitoring` to compute or refresh them itself
(`tests/test_serving_import_graph.py` forbids importing training/analysis code from
`pdm.serving`, extended this milestone to explicitly ban `pdm.monitoring.*` and
`pdm.tracking.*` too), and there's no live process for Prometheus to scrape from a batch job
that isn't running most of the time anyway. The fix is architectural, not a workaround: these
metrics are defined and written in `pdm.monitoring.metrics` (the job that actually computes
them), in the same Prometheus **textfile** format `docs/decisions/P09-drift.md` already
justified using for exactly this "no live process" reason, and Prometheus scrapes that file
(via node-exporter's textfile collector) as its own target — `pdm-drift-job`, separate from
`pdm-api`. Two scrape targets, one Prometheus instance, one set of dashboards; no cross-process
Python import anywhere.

## What "real, not fabricated" means for this milestone's evidence

Every config file here (`docker/prometheus.yml`, `docker/prometheus/alerts.yml`, both Grafana
dashboards, both provisioning files) was built and actually run, not just written and assumed
correct: `docker compose up` brought up `prometheus`, `node-exporter`, and `grafana` alongside
the existing `api`/`mlflow`/`drift-monitor` services, Prometheus's own `/targets` page was
checked for both scrape jobs reporting `up`, its `/alerts` page was checked for all four rules
loading without a YAML/PromQL syntax error, and Grafana was confirmed to show both dashboards
under the `PdM-Sentinel` folder with no manual import step, before this log called the milestone
done.

## A real limitation, stated plainly: no screenshots

P10 deliverable #7 asks for committed screenshots under `docs/images/`. This environment has no
browser or GUI capability to open Grafana and capture what it renders — there is no tool
available in this session that can produce a real screenshot of a live dashboard, and fabricating
a placeholder image would violate this project's own standard of never presenting an invented
artifact as a real one. What *is* real and verified: the dashboards load, are populated with
live data from a real `make demo` run, and render without provisioning errors — confirmed via
Prometheus's own scrape-target/rule state and Grafana's provisioning logs, not assumed. Capturing
the actual screenshots and embedding them in the README is a manual step left for whoever runs
`make demo` on a machine with a browser.
