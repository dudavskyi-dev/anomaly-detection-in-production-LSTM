"""The TF training loop: overfits a tiny batch, is deterministic given a seed, checkpoint
round-trips through a bundle, and detects a diverging (NaN) loss — the same guarantees
``tests/test_torch_train.py`` checks for the PyTorch training loop."""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from pdm.models.tf.architecture import LSTMClassifier, LSTMRegressor
from pdm.models.tf.bundle import load_tf_model_weights, save_tf_bundle
from pdm.models.tf.dataset import make_tf_dataset
from pdm.models.tf.train import set_full_determinism, train_classifier, train_regressor
from pdm.preprocessing.scaling import StandardScaler
from pdm.preprocessing.windowing import FeatureSpec

pytestmark = pytest.mark.fast


def _tiny_regression_data(n=8, window_size=6, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    windows = rng.normal(size=(n, window_size, n_features)).astype(np.float32)
    # kept close to zero for the same reason `test_torch_train.py` does: a large target offset
    # would need far more than a few hundred optimiser steps to shift the output bias that far.
    targets = {"rul": windows.mean(axis=(1, 2)).astype(np.float32)}
    return windows, targets


def test_overfit_tiny_batch_reaches_near_zero_loss():
    windows, targets = _tiny_regression_data()
    ds = make_tf_dataset(windows, targets, "rul", shuffle=False, batch_size=8)
    model = LSTMRegressor(n_features=3, hidden_sizes=(8, 4), dropout=0.0)

    result = train_regressor(
        model, ds, ds, max_epochs=300, learning_rate=0.01, patience=300, seed=0
    )
    assert result.best_val_score < 0.1, f"expected near-zero RMSE, got {result.best_val_score}"


def test_training_is_deterministic_given_seed():
    windows, targets = _tiny_regression_data(n=16)

    def run():
        set_full_determinism(0)
        model = LSTMRegressor(n_features=3, hidden_sizes=(6, 3), dropout=0.0)
        ds = make_tf_dataset(windows, targets, "rul", shuffle=True, batch_size=4, seed=0)
        return train_regressor(model, ds, ds, max_epochs=5, patience=5, seed=0)

    result1 = run()
    result2 = run()
    assert [s.train_loss for s in result1.history] == [s.train_loss for s in result2.history]
    assert result1.best_val_score == result2.best_val_score


def test_checkpoint_round_trip_through_a_bundle(tmp_path):
    windows, targets = _tiny_regression_data(n=8)
    ds = make_tf_dataset(windows, targets, "rul", shuffle=False, batch_size=8)
    model = LSTMRegressor(n_features=3, hidden_sizes=(6, 3), dropout=0.0)
    train_regressor(model, ds, ds, max_epochs=3, patience=3, seed=0)

    flat = pd.DataFrame(windows.reshape(-1, 3), columns=["a", "b", "c"])
    scaler = StandardScaler().fit(flat, ["a", "b", "c"])
    feature_spec = FeatureSpec(
        feature_names=["a", "b", "c"],
        window_size=6,
        stride=1,
        pad_short_units=True,
        per_condition_normalization=False,
    )

    run_dir = tmp_path / "run"
    save_tf_bundle(
        run_dir,
        model,
        scaler=scaler,
        feature_spec=feature_spec,
        train_windows=windows,
        metrics={"rmse": 1.0},
        thresholds={},
        seed=0,
        dataset_version="test",
        parameter_count=sum(int(tf.size(v)) for v in model.trainable_variables),
    )

    reloaded = LSTMRegressor(n_features=3, hidden_sizes=(6, 3), dropout=0.0)
    load_tf_model_weights(run_dir, reloaded)

    x = tf.convert_to_tensor(windows)
    out1 = model(x, training=False)
    out2 = reloaded(x, training=False)
    np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-6, atol=1e-6)

    for name in (
        "model_tf.weights.h5",
        "scaler.json",
        "feature_spec.json",
        "thresholds.json",
        "metrics.json",
        "reference_stats.json",
        "metadata.json",
    ):
        assert (run_dir / name).exists(), f"missing bundle file: {name}"


def test_diverging_loss_is_detected_and_stops_training():
    windows, targets = _tiny_regression_data(n=8)
    ds = make_tf_dataset(windows, targets, "rul", shuffle=False, batch_size=8)
    model = LSTMRegressor(n_features=3, hidden_sizes=(4, 2), dropout=0.0)

    def nan_loss(y_true, y_pred):
        return tf.constant(float("nan"))

    from pdm.models.tf.train import _run_fit

    result = _run_fit(
        model,
        ds,
        ds,
        loss=nan_loss,
        monitor="val_loss",
        metrics=None,
        max_epochs=5,
        learning_rate=0.01,
        patience=5,
        grad_clip_norm=None,
        seed=0,
        verbose=False,
        csv_log_path=None,
    )
    assert result.diverged is True
    assert all(tf.reduce_all(tf.math.is_finite(v)).numpy() for v in model.trainable_variables)


def test_classifier_trains_with_and_without_pos_weight():
    rng = np.random.default_rng(0)
    n, window_size, n_features = 40, 5, 2
    windows = rng.normal(size=(n, window_size, n_features)).astype(np.float32)
    targets = {"will_fail": (rng.random(n) < 0.1).astype(np.float32)}
    ds = make_tf_dataset(windows, targets, "will_fail", shuffle=False, batch_size=8)

    unweighted = LSTMClassifier(n_features=n_features, hidden_sizes=(6, 3), dropout=0.0)
    result_unweighted = train_classifier(unweighted, ds, ds, max_epochs=5, patience=5, seed=0)

    weighted = LSTMClassifier(n_features=n_features, hidden_sizes=(6, 3), dropout=0.0)
    result_weighted = train_classifier(
        weighted, ds, ds, pos_weight=9.0, max_epochs=5, patience=5, seed=0
    )

    assert np.isfinite(result_unweighted.best_val_score)
    assert np.isfinite(result_weighted.best_val_score)
