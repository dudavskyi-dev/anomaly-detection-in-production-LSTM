"""Structural enforcement of "the test set is loaded exactly once, at final evaluation."

Wrapping a split's arrays in a :class:`Split` and evaluating it through :func:`evaluate` makes
a second evaluation of the same test split a hard error, rather than a convention someone has
to remember. Validation splits may be evaluated repeatedly (that's what tuning is for); only
``name="test"`` is single-use.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


class SplitAlreadyEvaluatedError(RuntimeError):
    pass


@dataclass
class Split:
    """A named, single-owner view of a dataset split's windows and targets."""

    name: str
    windows: np.ndarray
    targets: dict[str, np.ndarray]
    _consumed: bool = field(default=False, init=False, repr=False)


def evaluate(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    split: Split,
    target_col: str,
    metric_fn: Callable[[np.ndarray, np.ndarray], dict],
) -> dict:
    """Predict on ``split.windows`` and score against ``split.targets[target_col]``.

    If ``split.name == "test"``, this may be called at most once per :class:`Split` instance —
    a second call raises :class:`SplitAlreadyEvaluatedError` instead of silently re-scoring
    against test, which would defeat the point of holding it out in the first place.
    """
    if split.name == "test" and split._consumed:
        raise SplitAlreadyEvaluatedError(
            f"Split {split.name!r} has already been evaluated once. The test set may only be "
            "scored at final evaluation, never re-used for a second look — build a fresh Split "
            "(e.g. a new seed's run) if you need another measurement."
        )
    y_pred = predict_fn(split.windows)
    metrics = metric_fn(split.targets[target_col], y_pred)
    split._consumed = True
    return metrics
