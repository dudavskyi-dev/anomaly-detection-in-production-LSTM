"""C-MAPSS RUL regression baselines: dummy floors, Ridge, and RandomForest.

A deep model that beats no baseline is worthless — "RMSE 16" means nothing without knowing
ridge regression gets 21 and predicting the mean gets 40. These are that floor.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from pdm.config import settings
from pdm.evaluation.metrics import regression_metrics
from pdm.models.baseline.window_features import make_baseline_features


def dummy_mean_baseline(train_targets: np.ndarray, eval_targets: np.ndarray) -> dict:
    """Predict the training mean for every row — the floor any real model must clear."""
    pred = np.full_like(eval_targets, fill_value=float(np.mean(train_targets)), dtype=float)
    return regression_metrics(eval_targets, pred)


def dummy_cap_baseline(eval_targets: np.ndarray, cap: int | None = None) -> dict:
    """Predict the RUL cap for every row — the other floor: "assume every engine is healthy"."""
    cap = settings.data.rul_cap if cap is None else cap
    pred = np.full_like(eval_targets, fill_value=float(cap), dtype=float)
    return regression_metrics(eval_targets, pred)


def ridge_baseline(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    eval_windows: np.ndarray,
    eval_targets: np.ndarray,
    *,
    seed: int,
    alpha: float | None = None,
) -> dict:
    alpha = settings.training.baseline_ridge_alpha if alpha is None else alpha
    x_train = make_baseline_features(train_windows)
    x_eval = make_baseline_features(eval_windows)
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(x_train, train_targets)
    pred = model.predict(x_eval)
    return regression_metrics(eval_targets, pred)


def random_forest_regressor_baseline(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    eval_windows: np.ndarray,
    eval_targets: np.ndarray,
    *,
    seed: int,
    n_estimators: int | None = None,
) -> dict:
    n_estimators = (
        settings.training.baseline_rf_n_estimators if n_estimators is None else n_estimators
    )
    x_train = make_baseline_features(train_windows)
    x_eval = make_baseline_features(eval_windows)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=settings.training.baseline_rf_max_depth,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, train_targets)
    pred = model.predict(x_eval)
    return regression_metrics(eval_targets, pred)
