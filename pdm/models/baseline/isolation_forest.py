"""Isolation Forest anomaly-detection baseline (P06 deliverable #3): the classical floor the
LSTM autoencoder must clear, on the same flattened-plus-summary-feature representation P03's
classical baselines use.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from pdm.models.baseline.window_features import make_baseline_features


def fit_isolation_forest(
    train_windows: np.ndarray, *, seed: int, contamination: float | str = "auto"
) -> IsolationForest:
    """Fit on **healthy-only** windows, exactly like the LSTM autoencoder — the same
    semi-supervised setup, so the two detectors are compared on equal footing."""
    x_train = make_baseline_features(train_windows)
    model = IsolationForest(contamination=contamination, random_state=seed, n_jobs=-1)
    model.fit(x_train)
    return model


def isolation_forest_scores(model: IsolationForest, windows: np.ndarray) -> np.ndarray:
    """Anomaly score, **higher = more anomalous** — the same convention as
    ``pdm.models.torch.autoencoder.reconstruction_error``, so the two can be fused directly.

    sklearn's own convention is the opposite (``score_samples`` is higher for *normal* points,
    the average isolation path length); negating it here keeps every detector in this project on
    one consistent "higher score = more anomalous" axis rather than requiring every caller to
    remember which detector's sign is flipped.
    """
    x = make_baseline_features(windows)
    return -model.score_samples(x)
