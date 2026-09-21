"""TensorFlow/Keras LSTM autoencoder — the "production path" mirror of
``pdm.models.torch.autoencoder`` (spec: "PyTorch primary, Keras mirror for the production
path"). Same encoder-bottleneck-decoder shape; unlike P05's regressor, this mirror is not
parameter-matched or benchmarked head-to-head against the PyTorch version — P06's deliverables
ask for a working production-track artifact, not a second full framework bake-off.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


class LSTMAutoencoder(keras.Model):
    """Encoder LSTM (hidden size = bottleneck) -> repeat across time -> decoder LSTM ->
    per-timestep ``Dense`` back to ``n_features``, mirroring
    ``pdm.models.torch.autoencoder.LSTMAutoencoder`` layer-for-layer."""

    def __init__(self, n_features: int, window_size: int, latent_dim: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.encoder = keras.layers.LSTM(latent_dim)
        self.repeat = keras.layers.RepeatVector(window_size)
        self.decoder = keras.layers.LSTM(latent_dim, return_sequences=True)
        self.output_layer = keras.layers.TimeDistributed(keras.layers.Dense(n_features))
        self(tf.zeros((1, window_size, n_features)))  # force variable creation

    def encode(self, x):
        return self.encoder(x)

    def call(self, x, training: bool = False):
        z = self.encoder(x, training=training)
        decoded = self.decoder(self.repeat(z), training=training)
        return self.output_layer(decoded)


def reconstruction_error(model: LSTMAutoencoder, windows: np.ndarray) -> np.ndarray:
    """Per-window MSE, higher = more anomalous — same convention as the PyTorch version."""
    x = tf.convert_to_tensor(windows, dtype=tf.float32)
    recon = model(x, training=False)
    return tf.reduce_mean(tf.square(recon - x), axis=[1, 2]).numpy()
