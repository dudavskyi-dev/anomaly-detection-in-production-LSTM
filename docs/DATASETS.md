# Datasets

Three datasets, each earning its place. All are free, public, and permissively licensed.

---

## 1. NASA C-MAPSS Turbofan Engine Degradation — **primary**

The benchmark dataset of the prognostics field. Run-to-failure multivariate sensor
trajectories from simulated turbofan engines.

- **Download**: `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
- **Source page**: NASA PCoE Data Set Repository —
  `https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/`
- **Mirror**: PHM Society data repository — `https://data.phmsociety.org/nasa/`
- **Citation**: A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data
  Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

**Contents.** Four subsets, FD001–FD004, of increasing difficulty, simulated under different
combinations of operating conditions and fault modes. Each file is whitespace-separated text
with 26 columns: unit number, time in cycles, three operational settings, then 21 sensor
measurements. Engines start healthy and develop a fault over time; training engines run to
failure. Test sets are truncated before failure, with a separate `RUL_FDxxx.txt` giving the
true remaining life.

**Why it's here.** It is the "real-time IoT telemetry from industrial equipment" in your CV
bullet, and it supports RUL regression, failure classification, and anomaly detection from a
single source.

**Important gotchas the agent must handle:**
- Several sensors are constant within a subset and carry zero information — drop them, and
  document which and why.
- FD002 and FD004 have six distinct operating conditions; raw sensor values cluster by
  condition, so either normalise per condition or the model learns the condition, not the fault.
- RUL must be capped (125 is standard). Uncapped linear RUL assumes degradation starts at
  cycle 1, which is wrong.
- Split by engine unit, never by row.

**Drift use.** Train on FD001, feed FD002/FD003 as "production" traffic. Different operating
conditions and fault modes mean genuinely different feature distributions — a real drift
signal rather than injected noise.

---

## 2. AI4I 2020 Predictive Maintenance Dataset — **tabular baseline**

- **Download**: `https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset`
- **Programmatic**: `pip install ucimlrepo` then `fetch_ucirepo(id=601)`
- **Licence**: CC BY 4.0 — free to use and adapt with attribution.
- **DOI**: 10.24432/C5HS5C

**Contents.** 10,000 rows, 14 columns: product quality variant, air temperature (K), process
temperature (K), rotational speed (rpm), torque (Nm), tool wear (min), a binary machine-failure
label, and five specific failure-mode flags (tool wear TWF, heat dissipation HDF, power PWF,
overstrain OSF, random RNF). Synthetic but modelled on real industrial milling data.

**Why it's here.** Two reasons. It gives you a fast, honest classical-ML baseline
(scikit-learn on your CV), and it's the cleanest possible illustration of **label leakage**:
the five failure-mode flags must be dropped before modelling, because the target is defined
as their OR. An interviewer who knows this dataset will ask. Being the person who spotted it
is worth a lot.

Also note the class imbalance (~3.4% failures) — this is where PR-AUC over accuracy gets
demonstrated concretely.

---

## 3. Numenta Anomaly Benchmark (NAB) — **external validation**

- **Repo**: `https://github.com/numenta/NAB`
- **Key file**: `data/realKnownCause/machine_temperature_system_failure.csv` — real
  temperature readings from an industrial machine with a known component failure.
- **Citation**: Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time
  anomaly detection for streaming data. *Neurocomputing*.

**Contents.** Over 50 labelled real-world and artificial time series, timestamped
single-valued metrics, with labelled anomalous windows and a scoring mechanism designed for
streaming detection that rewards early detection.

**Why it's here.** Your anomaly detector needs to be tested on data it wasn't tuned for.
C-MAPSS is simulated; NAB `realKnownCause` is a real machine that really failed. Your metrics
will drop. Report the drop — a candidate who shows a synthetic-to-real generalisation gap and
can explain it reads as far more credible than one whose every number is 0.9+.

**Note on NAB scoring.** NAB's own scoring function is window-based and gives partial credit
for early detection, so it is not comparable to point-wise F1. Report both, and explain the
difference; that distinction is a strong signal in an interview.

---

## Storage and reproducibility

- Everything under `data/`, gitignored. Never commit datasets.
- `make data` downloads, verifies checksums, and converts to parquet under `data/processed/`.
- Record dataset version and file hashes in each bundle's `metadata.json`.
- If a download URL breaks, the agent should fail loudly with the source page URL, not
  silently substitute synthetic data.

## Mapping to CV claims

| CV claim | Backed by |
|---|---|
| "real-time IoT telemetry" | C-MAPSS sensor streams + replay simulator |
| "forecast machine failures" | RUL regression + failure-within-W classification on C-MAPSS |
| "F1-score in anomaly detection" | LSTM-AE on C-MAPSS, validated on NAB |
| "Scikit-learn" | Isolation Forest, Random Forest, Ridge baselines on AI4I + C-MAPSS |
| "data drift detection" | FD001 → FD002/FD003 distribution shift, PSI + KS |
| "time-series preprocessing" | per-unit labelling, capping, windowing, per-condition normalisation |
