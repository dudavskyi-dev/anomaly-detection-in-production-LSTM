"""LSTM autoencoder: forward-pass shapes, that it drops straight into the shared
``train_model`` loop, and that a trained model reconstructs healthy data better than clearly
different (anomalous-looking) data — the whole point of the detector."""

import numpy as np
import pytest
import torch

from pdm.models.torch.autoencoder import (
    LSTMAutoencoder,
    per_sensor_reconstruction_error,
    reconstruction_error,
    train_autoencoder,
)
from pdm.models.torch.dataset import make_dataloader
from pdm.models.torch.train import set_full_determinism

pytestmark = pytest.mark.fast


def test_forward_shape_matches_input_shape():
    model = LSTMAutoencoder(n_features=4, window_size=6, latent_dim=3)
    out = model(torch.randn(5, 6, 4))
    assert out.shape == (5, 6, 4)


def test_encode_returns_bottleneck_shape():
    model = LSTMAutoencoder(n_features=4, window_size=6, latent_dim=3)
    z = model.encode(torch.randn(5, 6, 4))
    assert z.shape == (5, 3)


def test_overfit_tiny_healthy_batch_reaches_near_zero_reconstruction_error():
    """Unlike the regressor/classifier overfit tests (output dimensionality 1, always trivially
    overfittable), an autoencoder's bottleneck must actually be wide enough to *hold* enough
    information to reproduce ``window_size * n_features`` values per window — ``latent_dim=4``
    on 8 windows of shape ``(6, 3)`` (18 values each) plateaued around RMSE 0.6 no matter how
    long it trained, because 4 numbers genuinely cannot encode 18 independent random values.
    ``latent_dim=16`` has enough capacity to actually memorise this tiny batch; this test checks
    the training loop can drive reconstruction error to ~0 given enough bottleneck capacity, not
    that any bottleneck size can."""
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(8, 6, 3)).astype(np.float32)
    loader = make_dataloader(windows, {}, shuffle=False, batch_size=8)
    set_full_determinism(0)
    model = LSTMAutoencoder(n_features=3, window_size=6, latent_dim=16)

    result = train_autoencoder(
        model, loader, loader, max_epochs=500, learning_rate=0.05, patience=500, seed=0
    )
    assert result.best_val_score < 0.1, f"expected near-zero RMSE, got {result.best_val_score}"


def test_reconstruction_error_is_higher_for_out_of_distribution_windows():
    rng = np.random.default_rng(0)
    healthy = rng.normal(0, 1, size=(30, 6, 3)).astype(np.float32)
    anomalous = rng.normal(6, 1, size=(10, 6, 3)).astype(np.float32)

    loader = make_dataloader(healthy, {}, shuffle=True, batch_size=8, seed=0)
    set_full_determinism(0)
    model = LSTMAutoencoder(n_features=3, window_size=6, latent_dim=2)
    train_autoencoder(model, loader, loader, max_epochs=100, patience=20, seed=0)

    healthy_err = reconstruction_error(model, healthy)
    anomalous_err = reconstruction_error(model, anomalous)
    assert anomalous_err.mean() > healthy_err.mean()


def test_per_sensor_reconstruction_error_shape():
    model = LSTMAutoencoder(n_features=4, window_size=6, latent_dim=3)
    windows = np.random.default_rng(0).normal(size=(5, 6, 4)).astype(np.float32)
    per_sensor = per_sensor_reconstruction_error(model, windows)
    assert per_sensor.shape == (5, 4)
    assert np.all(per_sensor >= 0)
