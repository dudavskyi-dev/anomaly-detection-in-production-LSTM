"""Thin MLflow wrapper (P07): every *training path* logs through this — none of them
``import mlflow`` directly — so the tagging, param-flattening, and secret-scrubbing conventions
live in exactly one place instead of being reinvented (and drifting) per call site.
``pdm.tracking.registry``/``pdm.tracking.promotion`` are a different concern (registry
administration) and use the MLflow client API directly, since there's nothing to wrap there.
"""

import hashlib
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import mlflow

from pdm.config import settings

# Leaf param keys matching any of these substrings are dropped, not logged, regardless of value —
# "secrets never logged as params" (spec constraint), defensive even though nothing in this
# project's config currently holds one.
_SECRET_KEY_MARKERS = ("password", "secret", "token", "api_key", "apikey")


def _looks_secret(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_KEY_MARKERS)


def _scrub_uri_credentials(value: str) -> str:
    """Strips a ``user:pass@`` userinfo component out of a URI-shaped string before it's ever
    logged as a param — the one place this project's own config could realistically carry a
    credential (a remote ``PDM__TRACKING__URI`` with embedded basic-auth)."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.netloc or "@" not in parts.netloc:
        return value
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def _flatten(prefix: str, value: object, out: dict[str, object]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
        return
    key = prefix
    if _looks_secret(key.rsplit(".", 1)[-1]):
        return
    if isinstance(value, list | tuple):
        value = str(value)
    elif isinstance(value, str):
        value = _scrub_uri_credentials(value)
    out[key] = value


def flatten_params(config: dict) -> dict[str, object]:
    """Flattens a nested config dict into MLflow's flat ``{"a.b.c": value}`` param format,
    dropping any leaf whose key looks like a secret and scrubbing credentials out of any
    URI-shaped string value."""
    out: dict[str, object] = {}
    _flatten("", config, out)
    return out


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def git_is_dirty() -> bool | None:
    """``None`` when not in a git repo at all (can't answer the question); ``True``/``False``
    otherwise — distinct from "clean", which is a claim only a real repo can back up."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return bool(status.strip())
    except Exception:
        return None


def sha256_of_files(paths: list[Path]) -> str:
    """One combined digest over a set of files, order-independent (sorted first) — the
    "dataset hash" tag: which exact processed-data bytes produced this run's numbers."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def configure_tracking() -> None:
    """Points MLflow at ``settings.tracking.uri`` (default: a local ``./mlruns`` file store;
    override with ``PDM__TRACKING__URI`` for a remote tracking server) and selects/creates the
    configured experiment. Idempotent — safe to call before every run."""
    mlflow.set_tracking_uri(settings.tracking.uri)
    mlflow.set_experiment(settings.tracking.experiment_name)


@contextmanager
def start_run(
    *,
    run_name: str,
    framework: str,
    seed: int,
    dataset_hash: str | None = None,
    extra_tags: dict[str, object] | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Starts (and always ends, even on exception) one MLflow run, tagged with git sha,
    dirty-tree flag, dataset hash, framework, and seed — the tag set needed to answer "what code,
    what data, which run produced this number" for any metric this project reports.
    """
    configure_tracking()
    tags: dict[str, object] = {
        "git_sha": git_sha() or "unknown",
        "git_dirty": str(git_is_dirty()),
        "framework": framework,
        "seed": str(seed),
    }
    if dataset_hash is not None:
        tags["dataset_hash"] = dataset_hash
    if extra_tags:
        tags.update({k: str(v) for k, v in extra_tags.items()})
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_params(config: dict) -> None:
    """Logs every leaf of a (possibly nested) config dict as a flattened MLflow param."""
    for key, value in flatten_params(config).items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_epoch_history(history: list) -> None:
    """Logs one training run's full per-epoch curve as MLflow step metrics, from a list of
    objects with a ``to_dict()`` method shaped like ``pdm.models.torch.train.EpochStats`` /
    ``pdm.models.tf.train.EpochStats`` (both frameworks' training loops already produce this).
    Called once, after training finishes — MLflow doesn't need the epoch loop itself
    instrumented, only the recorded history it already keeps.
    """
    for entry in history:
        d = entry.to_dict()
        epoch = d.pop("epoch")
        mlflow.log_metrics({k: v for k, v in d.items() if isinstance(v, int | float)}, step=epoch)


def log_artifact(path: Path, artifact_path: str | None = None) -> None:
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_artifacts(directory: Path, artifact_path: str | None = None) -> None:
    mlflow.log_artifacts(str(directory), artifact_path=artifact_path)
