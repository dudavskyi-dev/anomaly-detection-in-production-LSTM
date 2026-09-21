"""TF/Keras LSTM autoencoder mirror: forward shape and the same healthy-vs-anomalous
reconstruction-error sanity check as the PyTorch version."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from pdm.models.tf.autoencoder import LSTMAutoencoder, reconstruction_error
from pdm.models.tf.dataset import make_tf_dataset
from pdm.models.tf.train import set_full_determinism

pytestmark = pytest.mark.fast


def test_forward_shape_matches_input_shape():
    model = LSTMAutoencoder(n_features=4, window_size=6, latent_dim=3)
    out = model(tf.random.normal((5, 6, 4)))
    assert out.shape == (5, 6, 4)


def test_reconstruction_error_is_higher_for_out_of_distribution_windows():
    rng = np.random.default_rng(0)
    healthy = rng.normal(0, 1, size=(30, 6, 3)).astype(np.float32)
    anomalous = rng.normal(6, 1, size=(10, 6, 3)).astype(np.float32)

    set_full_determinism(0)
    model = LSTMAutoencoder(n_features=3, window_size=6, latent_dim=2)
    ds = make_tf_dataset(
        healthy, {"windows": healthy}, "windows", shuffle=True, batch_size=8, seed=0
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
    model.fit(ds, validation_data=ds, epochs=60, verbose=0)

    healthy_err = reconstruction_error(model, healthy)
    anomalous_err = reconstruction_error(model, anomalous)
    assert anomalous_err.mean() > healthy_err.mean()
