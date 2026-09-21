# P00 — Bootstrap decision log

## Dependency versions chosen and why

Every dependency is pinned with a compatible-release specifier (`~=X.Y.0`) rather than an
unpinned or exact-pin requirement, so patch releases are picked up automatically but minor/major
bumps require a deliberate change.

At the time of writing, the newest available versions on PyPI for this environment were
`pandas 3.0.5`, `mlflow 3.16.0`, `pytest 9.1.1`, `mypy 2.3.1`, `black 26.5.1`, and
`ruff 0.16.7`. These are all recent major-version jumps (pandas 3.0, mlflow 3.x) or versions
released after this project's implementation knowledge was current. Rather than pin to the
absolute newest release and risk hitting undocumented breaking changes partway through a later
milestone, dev-tooling and data-library pins were rolled back one minor/major generation to
versions still available on PyPI and well understood:

- `numpy~=2.1.0`, `pandas~=2.2.0`, `pyarrow~=18.0.0`, `scikit-learn~=1.5.0`
- `pydantic~=2.9.0`, `pydantic-settings~=2.6.0`, `typer~=0.12.0`
- `pytest~=8.2.0`, `pytest-cov~=5.0.0`, `ruff~=0.6.0`, `black~=24.4.0`, `mypy~=1.10.0`,
  `hypothesis~=6.100.0`

**Torch/TensorFlow compatibility.** `torch~=2.14.0` and `tensorflow~=2.21.0` are the newest
releases of each framework available for this Python 3.11 / win_amd64 target. Both are pinned
to their own minor line independent of the numpy pin above; if their transitive numpy
requirements conflict with `numpy~=2.1.0` when P04/P05 actually install them, that will surface
as a `pip` resolution error at install time and the numpy pin will be widened then, with the
reason recorded in the P04 or P05 log. Rather than guess a numpy ceiling now, the conflict (if
any) is deferred to the milestone that actually exercises it.

**MLflow.** `mlflow~=2.22.0` (latest 2.x) was chosen over the newer `mlflow 3.x` line
deliberately: MLflow 3.0 changed core tracking concepts (notably `LoggedModel` replacing some of
the 2.x model-logging API), and the P07 milestone's implementation plan assumes 2.x semantics.
Revisit this in P07 if 2.22 turns out to be missing something needed.

**Heavy extras are not installed by `make install`.** `pyproject.toml` defines optional-dependency
groups (`torch`, `tf`, `serving`, `tracking`, `dev`) plus an `all` group that installs everything.
`make install` installs only `.[dev]` (core + dev tooling) so the bootstrap step stays fast and
doesn't pull multi-gigabyte framework wheels before any milestone needs them. Each later
milestone's Makefile target should install its own extra (or the target could grow an install
step) before running. This is a deliberate deviation from "install" meaning "everything, always,"
traded for a fast, low-risk P00.

## Why pydantic-settings over argparse/YAML/Hydra

`pydantic-settings` was chosen because:

- It gives typed, validated config for free — a wrong type in an env var override fails loudly
  at startup instead of silently propagating as a string.
- Nested settings map cleanly onto `PDM__SECTION__KEY` env vars via `env_nested_delimiter`,
  which is exactly the convention the spec requires, with no custom parsing code.
- The same `Settings` object serves as both the CLI's config source and (later) the FastAPI
  app's config source — one model, not two.

What this gives up, compared to Hydra: no config composition/overrides via multiple YAML files,
no command-line multirun/sweep support, and no structured config groups for swapping whole
subsystems (e.g., "use this model config vs that one") without writing that switching logic by
hand. For a project of this size, with sweeps implemented explicitly in P04's experiment scripts
rather than via a config-management framework, that tradeoff is fine — Hydra would be solving a
problem this project doesn't have yet. Plain argparse was rejected outright: it doesn't nest, and
env-var override was a hard requirement in the spec.

`extra="forbid"` is set on the root `Settings` model so a typo'd env var (or a typo'd field name)
fails fast instead of being silently ignored — this is tested in `tests/test_config.py`.

## Environment problem hit, and the fix

The first `pip install -e ".[dev]"` was run without an active virtual environment and installed
directly into the user's global Python (`AppData\Local\Microsoft\WindowsApps\python.exe`). Pip
resolved and **downgraded** several already-installed global packages to satisfy this project's
pins — `numpy 2.3.5 → 2.1.3`, `pandas 2.3.3 → 2.2.3`, `pytest 9.0.2 → 8.2.2`,
`pytest-cov 7.1.0 → 5.0.0`, `mypy 2.3.1 → 1.10.1`, `ruff 0.16.5 → 0.6.9`,
`pydantic 2.13.4 → 2.9.2` (and `pydantic-core` alongside it) — which could have broken any other
project relying on that same interpreter.

Before proceeding, this was checked and repaired:

1. Confirmed no separate, unrelated `pdm` CLI tool existed on the system that this project's own
   `pdm` console-script entry point could have clobbered (`pip show pdm` → not found).
2. Uninstalled every package this install had added net-new (`pdm-sentinel`, `typer`,
   `pydantic-settings`, `scikit-learn`, `black`, `hypothesis`, `pyarrow`, `shellingham`,
   `threadpoolctl`, `sortedcontainers`).
3. Reinstalled the exact prior versions of every package it had downgraded
   (`numpy==2.3.5 pandas==2.3.3 mypy==2.3.1 ruff==0.16.5 pytest==9.0.2 pytest-cov==7.1.0
   pydantic==2.13.4 pydantic-core==2.46.4`), verified with `pip list` against the versions
   `pip`'s own install log had reported before the change.
4. Created a project-local `.venv/` (already `.gitignore`d) and reran
   `pip install -e ".[dev]"` inside it. All subsequent verification (`pytest`, `ruff`, `black`,
   `mypy`) runs against `.venv`, not the global interpreter.

Lesson for every later milestone: always install into `.venv`, never the bare `python`/`pip`
found on `PATH`.

## Ambiguities / questions flagged, not guessed

- The spec's `make` target list (section 9) assumes a Unix-like environment; this development
  machine is Windows without `make` installed. The `Makefile` is still written exactly as
  specified (it's what CI in P11 and any Linux/macOS clone will use), but local verification for
  this milestone was done by running the underlying commands the Makefile wraps
  (`pip install -e ".[dev]"`, `pytest -m fast`, `ruff check`, `black --check`, `mypy`) directly
  against `.venv/Scripts/python.exe`. Not a spec conflict, just a local tooling gap worth noting
  before P01, since `make data` etc. will hit the same gap.
- Spec section 9's default list (`window_size=30`, `rul_cap=125`, `failure_horizon_w=30`,
  `stride=1`, `seed=42`) is explicitly called out as "every default from the spec", so
  `pdm/config/settings.py` also pre-populates fields for values named elsewhere in the spec that
  later milestones will need (LSTM hidden sizes 100/50, dropout 0.2, PSI bands 0.1/0.25, autoenc
  healthy-RUL threshold) even though nothing consumes them yet. A few fields
  (`training.batch_size`, `training.learning_rate`, `training.early_stopping_patience`,
  `training.grad_clip_norm`, `serving.max_batch_size`) are not spec-mandated numbers — they're
  ordinary engineering defaults with no measured basis, included so "no magic numbers" holds from
  day one. These are not results and are not claimed as such; P04/P05 may change them and should
  say why if they do.

No other part of the spec was ambiguous enough to block P00. `make install && make lint &&
make test-fast` — run via `.venv` directly — all pass on this clean checkout.
