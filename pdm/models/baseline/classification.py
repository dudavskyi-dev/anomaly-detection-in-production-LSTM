"""C-MAPSS failure-classification baselines: dummy, class-weighted LogisticRegression, and
class-weighted RandomForest — the floor P06's LSTM-AE / P04's classifier head must clear.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pdm.config import settings
from pdm.evaluation.metrics import classification_metrics
from pdm.models.baseline.window_features import make_baseline_features


def dummy_classifier_baseline(
    train_targets: np.ndarray,
    eval_targets: np.ndarray,
    *,
    seed: int,
    strategy: str = "most_frequent",
) -> dict:
    x_train = np.zeros((len(train_targets), 1))
    x_eval = np.zeros((len(eval_targets), 1))
    model = DummyClassifier(strategy=strategy, random_state=seed)
    model.fit(x_train, train_targets)
    y_pred = model.predict(x_eval)
    y_score = model.predict_proba(x_eval)[:, 1]
    return classification_metrics(eval_targets, y_pred, y_score)


def logistic_regression_baseline(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    eval_windows: np.ndarray,
    eval_targets: np.ndarray,
    *,
    seed: int,
) -> dict:
    x_train = make_baseline_features(train_windows)
    x_eval = make_baseline_features(eval_windows)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(x_train, train_targets)
    y_pred = model.predict(x_eval)
    y_score = model.predict_proba(x_eval)[:, 1]
    return classification_metrics(eval_targets, y_pred, y_score)


def random_forest_classifier_baseline(
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
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=settings.training.baseline_rf_max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, train_targets)
    y_pred = model.predict(x_eval)
    y_score = model.predict_proba(x_eval)[:, 1]
    return classification_metrics(eval_targets, y_pred, y_score)
