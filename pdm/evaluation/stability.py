"""Seed-stability harness. Every headline number in this project is reported as mean ± std
over multiple seeds — a single-run number is not an acceptable result on its own.
"""

from collections.abc import Callable, Sequence

import numpy as np

from pdm.config import settings


def aggregate_over_seeds(
    results: Sequence[dict[str, float]], seeds: Sequence[int]
) -> dict[str, dict]:
    """Turn a list of per-seed flat metric dicts (same keys every time) into
    ``{metric: {mean, std, values, seeds}}``."""
    keys = results[0].keys()
    summary: dict[str, dict] = {}
    for key in keys:
        values = [float(r[key]) for r in results]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
            "seeds": list(seeds),
        }
    return summary


def run_seeds(
    train_fn: Callable[[int], dict[str, float]], seeds: tuple[int, ...] | None = None
) -> dict[str, dict]:
    """Call ``train_fn(seed)`` once per seed; each call must return a flat ``{metric: float}``
    dict with the same keys every time. Returns the per-metric mean/std/values summary.
    """
    seeds = settings.training.seeds if seeds is None else seeds
    results = [train_fn(seed) for seed in seeds]
    return aggregate_over_seeds(results, seeds)
