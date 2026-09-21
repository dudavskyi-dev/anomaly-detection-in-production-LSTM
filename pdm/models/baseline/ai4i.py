"""AI4I tabular baselines: dummy, LogisticRegression, and RandomForest, with the five
failure-mode flags dropped before any feature ever reaches a model.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from pdm.config import settings
from pdm.evaluation.metrics import classification_metrics
from pdm.preprocessing.features import drop_ai4i_leakage_columns

NUMERIC_COLUMNS = (
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
)
TARGET_COLUMN = "machine_failure"


def prepare_ai4i_splits(
    df: pd.DataFrame, *, val_fraction: float | None = None, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the leakage flags, then a stratified train/val split (AI4I ships no separate held-out
    test file the way C-MAPSS does, so there is no third split here)."""
    val_fraction = settings.data.val_fraction if val_fraction is None else val_fraction
    df = drop_ai4i_leakage_columns(df)
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, stratify=df[TARGET_COLUMN], random_state=seed
    )
    return train_df, val_df


def _encode(df: pd.DataFrame, encoder: OneHotEncoder) -> np.ndarray:
    numeric = df[list(NUMERIC_COLUMNS)].to_numpy(dtype=float)
    type_encoded = encoder.transform(df[["type"]])
    return np.concatenate([numeric, type_encoded], axis=1)


def _fit_encoder(train_df: pd.DataFrame) -> OneHotEncoder:
    return OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(train_df[["type"]])


def dummy_ai4i_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame, *, seed: int) -> dict:
    x_train = np.zeros((len(train_df), 1))
    x_eval = np.zeros((len(eval_df), 1))
    model = DummyClassifier(strategy="most_frequent", random_state=seed)
    model.fit(x_train, train_df[TARGET_COLUMN])
    y_pred = model.predict(x_eval)
    y_score = model.predict_proba(x_eval)[:, 1]
    return classification_metrics(eval_df[TARGET_COLUMN], y_pred, y_score)


def logistic_regression_ai4i_baseline(
    train_df: pd.DataFrame, eval_df: pd.DataFrame, *, seed: int
) -> dict:
    encoder = _fit_encoder(train_df)
    x_train = _encode(train_df, encoder)
    x_eval = _encode(eval_df, encoder)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(x_train, train_df[TARGET_COLUMN])
    y_pred = model.predict(x_eval)
    y_score = model.predict_proba(x_eval)[:, 1]
    return classification_metrics(eval_df[TARGET_COLUMN], y_pred, y_score)


def random_forest_ai4i_baseline(
    train_df: pd.DataFrame, eval_df: pd.DataFrame, *, seed: int, n_estimators: int | None = None
) -> dict:
    n_estimators = (
        settings.training.baseline_rf_n_estimators if n_estimators is None else n_estimators
    )
    encoder = _fit_encoder(train_df)
    x_train = _encode(train_df, encoder)
    x_eval = _encode(eval_df, encoder)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=settings.training.baseline_rf_max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, train_df[TARGET_COLUMN])
    y_pred = model.predict(x_eval)
    y_score = model.predict_proba(x_eval)[:, 1]
    return classification_metrics(eval_df[TARGET_COLUMN], y_pred, y_score)
