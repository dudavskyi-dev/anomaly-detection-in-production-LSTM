"""Session-wide pytest configuration.

Import ``torch`` and ``tensorflow`` before anything else (in particular before ``pandas``) gets
a chance to load. On this Windows environment, importing ``pandas`` first reliably breaks both
torch's (``OSError: [WinError 1114] ... c10.dll``) and TensorFlow's
(``ImportError: DLL load failed while importing _pywrap_tensorflow_internal``) native DLL
loading — importing them first avoids it entirely, and pytest's collection order (roughly
alphabetical by filename) would otherwise import several ``test_*.py`` files that pull in
pandas before it ever reaches a ``test_torch_*.py``/``test_tf_*.py`` file. See
``docs/decisions/P04-pytorch.md`` and ``docs/decisions/P05-tensorflow.md`` for the full story.
Both are optional extras (not installed by the base `make install`), so this is a no-op when
either isn't present — the rest of the suite must not depend on them being available.
"""

try:
    import torch  # noqa: F401
except ImportError:
    pass

try:
    import tensorflow  # noqa: F401
except ImportError:
    pass

import numpy as np
import pandas as pd
import pytest

TINY_FEATURE_NAMES = ["sensor_a", "sensor_b", "sensor_c"]
TINY_WINDOW_SIZE = 5
TINY_N_FEATURES = len(TINY_FEATURE_NAMES)
TINY_HIDDEN_SIZES = (4, 2)
TINY_LATENT_DIM = 2


def _tiny_scaler():
    from pdm.preprocessing.scaling import StandardScaler

    rng = np.random.default_rng(0)
    flat = pd.DataFrame(rng.normal(50, 10, size=(200, TINY_N_FEATURES)), columns=TINY_FEATURE_NAMES)
    return StandardScaler().fit(flat, TINY_FEATURE_NAMES)


def _tiny_feature_spec():
    from pdm.preprocessing.windowing import FeatureSpec

    return FeatureSpec(
        feature_names=TINY_FEATURE_NAMES,
        window_size=TINY_WINDOW_SIZE,
        stride=1,
        pad_short_units=True,
        per_condition_normalization=False,
    )


@pytest.fixture()
def tiny_rul_bundle_dir(tmp_path):
    """A small, fast, fully-valid PyTorch RUL-regressor bundle — real files on disk in the
    exact shape a training path would produce, just built from a randomly-initialised model on
    synthetic data (contract tests exercise the *service*, not model quality)."""
    from pdm.models.torch.architecture import LSTMRegressor, count_parameters
    from pdm.models.torch.bundle import save_torch_bundle

    model = LSTMRegressor(n_features=TINY_N_FEATURES, hidden_sizes=TINY_HIDDEN_SIZES, dropout=0.0)
    scaler = _tiny_scaler()
    feature_spec = _tiny_feature_spec()
    rng = np.random.default_rng(0)
    train_windows = rng.normal(size=(20, TINY_WINDOW_SIZE, TINY_N_FEATURES)).astype(np.float32)

    run_dir = tmp_path / "tiny_rul_bundle"
    save_torch_bundle(
        run_dir,
        model,
        scaler=scaler,
        feature_spec=feature_spec,
        train_windows=train_windows,
        metrics={
            "val": {"rmse": 12.0},
            "test": {"rmse": 12.0, "mae": 9.0, "r2": 0.6, "nasa_score": 500.0},
        },
        thresholds={"failure_horizon_w": 30},
        seed=0,
        dataset_version="tiny",
        parameter_count=count_parameters(model),
        architecture={"hidden_sizes": list(TINY_HIDDEN_SIZES), "dropout": 0.0},
    )
    return run_dir


@pytest.fixture()
def tiny_anomaly_bundle_dir(tmp_path):
    """A small, fast, fully-valid PyTorch LSTM-autoencoder bundle, mirroring
    ``tiny_rul_bundle_dir`` for ``/detect/anomaly``'s contract tests."""
    from pdm.models.torch.architecture import count_parameters
    from pdm.models.torch.autoencoder import LSTMAutoencoder
    from pdm.models.torch.bundle import save_torch_bundle

    model = LSTMAutoencoder(
        n_features=TINY_N_FEATURES, window_size=TINY_WINDOW_SIZE, latent_dim=TINY_LATENT_DIM
    )
    scaler = _tiny_scaler()
    feature_spec = _tiny_feature_spec()
    rng = np.random.default_rng(0)
    train_windows = rng.normal(size=(20, TINY_WINDOW_SIZE, TINY_N_FEATURES)).astype(np.float32)

    run_dir = tmp_path / "tiny_anomaly_bundle"
    save_torch_bundle(
        run_dir,
        model,
        scaler=scaler,
        feature_spec=feature_spec,
        train_windows=train_windows,
        metrics={"test": {"precision": 0.8, "recall": 0.7, "f1": 0.75}},
        thresholds={
            "healthy_rul_threshold": 125,
            "anomaly_score_threshold": 0.5,
            "anomaly_score_threshold_method": "percentile-99-of-healthy-training-scores",
        },
        seed=0,
        dataset_version="tiny",
        parameter_count=count_parameters(model),
        architecture={"latent_dim": TINY_LATENT_DIM},
    )
    return run_dir
