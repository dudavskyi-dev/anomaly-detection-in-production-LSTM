"""The training loop: overfits a tiny batch, is deterministic given a seed, checkpoint
round-trips through a bundle, and detects (rather than silently propagating) a NaN loss."""

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.models.torch.architecture import LSTMClassifier, LSTMRegressor
from pdm.models.torch.bundle import load_torch_model_state, save_torch_bundle
from pdm.models.torch.dataset import make_dataloader
from pdm.models.torch.train import (
    rmse_score,
    set_full_determinism,
    train_classifier,
    train_model,
    train_regressor,
)
from pdm.preprocessing.scaling import StandardScaler
from pdm.preprocessing.windowing import FeatureSpec

pytestmark = pytest.mark.fast


def _tiny_regression_data(n=8, window_size=6, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    windows = rng.normal(size=(n, window_size, n_features)).astype(np.float32)
    # kept close to zero (not offset by some large constant): with n == batch_size there is
    # exactly one optimiser step per epoch, and Adam's step size is roughly lr-per-step
    # regardless of gradient magnitude — a large target offset would need thousands of epochs
    # to shift the output bias that far, which looks like a training bug but is really just an
    # unreasonably-scaled target for this test.
    targets = {"rul": windows.mean(axis=(1, 2)).astype(np.float32)}
    return windows, targets


def test_overfit_tiny_batch_reaches_near_zero_loss():
    windows, targets = _tiny_regression_data()
    loader = make_dataloader(windows, targets, shuffle=False, batch_size=8)
    model = LSTMRegressor(n_features=3, hidden_sizes=(8, 4), dropout=0.0)

    result = train_regressor(
        model, loader, loader, max_epochs=300, learning_rate=0.01, patience=300, seed=0
    )
    assert result.best_val_score < 0.1, f"expected near-zero RMSE, got {result.best_val_score}"


def test_training_is_deterministic_given_seed():
    windows, targets = _tiny_regression_data(n=16)

    def run():
        # determinism of weight *initialisation* requires seeding before construction — seeding
        # only inside train_model (which runs after the model already exists) cannot retroactively
        # fix already-drawn initial weights. See set_full_determinism's docstring.
        set_full_determinism(0)
        model = LSTMRegressor(n_features=3, hidden_sizes=(6, 3), dropout=0.0)
        loader = make_dataloader(windows, targets, shuffle=True, batch_size=4, seed=0)
        return train_regressor(model, loader, loader, max_epochs=5, patience=5, seed=0)

    result1 = run()
    result2 = run()
    assert [s.train_loss for s in result1.history] == [s.train_loss for s in result2.history]
    assert result1.best_val_score == result2.best_val_score


def test_checkpoint_round_trip_through_a_bundle(tmp_path):
    windows, targets = _tiny_regression_data(n=8)
    loader = make_dataloader(windows, targets, shuffle=False, batch_size=8)
    model = LSTMRegressor(n_features=3, hidden_sizes=(6, 3), dropout=0.0)
    train_regressor(model, loader, loader, max_epochs=3, patience=3, seed=0)

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
    save_torch_bundle(
        run_dir,
        model,
        scaler=scaler,
        feature_spec=feature_spec,
        train_windows=windows,
        metrics={"rmse": 1.0},
        thresholds={},
        seed=0,
        dataset_version="test",
        parameter_count=sum(p.numel() for p in model.parameters()),
    )

    reloaded = LSTMRegressor(n_features=3, hidden_sizes=(6, 3), dropout=0.0)
    load_torch_model_state(run_dir, reloaded)

    x = torch.from_numpy(windows)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        out1, out2 = model(x), reloaded(x)
    torch.testing.assert_close(out1, out2)

    for name in (
        "model_torch.pt",
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
    loader = make_dataloader(windows, targets, shuffle=False, batch_size=8)
    model = LSTMRegressor(n_features=3, hidden_sizes=(4, 2), dropout=0.0)

    def nan_loss(_pred: torch.Tensor, _true: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float("nan"), requires_grad=True)

    result = train_model(
        model,
        loader,
        loader,
        target_col="rul",
        loss_fn=nan_loss,
        eval_score_fn=rmse_score,
        max_epochs=5,
        patience=5,
        seed=0,
    )
    assert result.diverged is True
    # weights must have been restored to the last-known-good (initial) state, not left mid-NaN
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_classifier_trains_with_and_without_pos_weight():
    rng = np.random.default_rng(0)
    n, window_size, n_features = 40, 5, 2
    windows = rng.normal(size=(n, window_size, n_features)).astype(np.float32)
    targets = {"will_fail": (rng.random(n) < 0.1).astype(np.float32)}
    loader = make_dataloader(windows, targets, shuffle=False, batch_size=8)

    unweighted = LSTMClassifier(n_features=n_features, hidden_sizes=(6, 3), dropout=0.0)
    result_unweighted = train_classifier(
        unweighted, loader, loader, max_epochs=5, patience=5, seed=0
    )

    weighted = LSTMClassifier(n_features=n_features, hidden_sizes=(6, 3), dropout=0.0)
    result_weighted = train_classifier(
        weighted, loader, loader, pos_weight=torch.tensor([9.0]), max_epochs=5, patience=5, seed=0
    )

    assert np.isfinite(result_unweighted.best_val_score)
    assert np.isfinite(result_weighted.best_val_score)
