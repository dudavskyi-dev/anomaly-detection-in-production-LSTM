"""TensorFlow/Keras LSTM architectures mirroring ``pdm.models.torch.architecture`` exactly:
``LSTM(100) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(1)``.

A plain ``keras.layers.LSTM`` does **not** produce the same parameter count as
``torch.nn.LSTM``: PyTorch keeps two separate bias vectors per layer (``bias_ih``, ``bias_hh``,
each ``(4*hidden,)``) that are always used additively (``... + b_ih + ... + b_hh``), whereas
Keras's ``LSTM`` layer has a single bias vector of the same shape. The two are mathematically
equivalent in what they can represent (only the *sum* of PyTorch's two biases ever matters to
the output), but PyTorch's version has twice as many bias parameters — 600 extra for the
default (100, 50) architecture at 17 input features (78,051 vs 77,451). Rather than pad the
Keras side with dead parameters to force a numeric match, :class:`SplitBiasLSTMCell` below
reimplements the LSTM cell with the same two-bias parameterisation PyTorch uses, so both
frameworks' models are not just parameter-count-identical but structurally identical. See
``docs/decisions/P05-tensorflow.md`` for the discovery story.
"""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import tensorflow as tf
from tensorflow import keras

# Lets mypy see `_GlobalNormClipMixin`'s `self` as a `keras.Model` (so `self.compute_loss`,
# `self.optimizer`, etc. resolve) without actually inheriting from it at runtime, which would
# make every `LSTMRegressor(_GlobalNormClipMixin, keras.Model)` instance's MRO include
# `keras.Model` twice.
if TYPE_CHECKING:
    _TrainStepBase: type = keras.Model
else:
    _TrainStepBase = object


class SplitBiasLSTMCell(keras.layers.Layer):
    """An LSTM cell with two separate additive bias vectors (``bias_ih`` for the input
    projection, ``bias_hh`` for the recurrent projection), matching ``torch.nn.LSTM``'s
    parameterisation gate-for-gate (input, forget, cell, output order) instead of Keras's
    default single-bias ``LSTM`` layer.
    """

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]
        self.output_size = units

    def build(self, input_shape) -> None:
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, 4 * self.units), name="kernel", initializer="glorot_uniform"
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units), name="recurrent_kernel", initializer="orthogonal"
        )
        self.bias_ih = self.add_weight(shape=(4 * self.units,), name="bias_ih", initializer="zeros")
        self.bias_hh = self.add_weight(shape=(4 * self.units,), name="bias_hh", initializer="zeros")
        self.built = True

    def call(self, inputs, states):
        h_tm1, c_tm1 = states
        z = (
            tf.matmul(inputs, self.kernel)
            + self.bias_ih
            + tf.matmul(h_tm1, self.recurrent_kernel)
            + self.bias_hh
        )
        z_i, z_f, z_g, z_o = tf.split(z, num_or_size_splits=4, axis=-1)
        i = tf.sigmoid(z_i)
        f = tf.sigmoid(z_f)
        g = tf.tanh(z_g)
        o = tf.sigmoid(z_o)
        c = f * c_tm1 + i * g
        h = o * tf.tanh(c)
        return h, [h, c]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"units": self.units})
        return config


class LSTMTrunk(keras.layers.Layer):
    """One ``SplitBiasLSTMCell`` per entry in ``hidden_sizes``, each followed by dropout on its
    full output sequence before the next layer (or the dense head) sees it — matching
    ``pdm.models.torch.architecture.LSTMTrunk`` layer-for-layer.
    """

    def __init__(self, hidden_sizes: tuple[int, ...] = (100, 50), dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout_rate = dropout
        self.output_size = self.hidden_sizes[-1]
        self.rnns = [
            keras.layers.RNN(SplitBiasLSTMCell(h), return_sequences=True) for h in self.hidden_sizes
        ]
        self.drops = [keras.layers.Dropout(dropout) for _ in self.hidden_sizes]

    def call(self, x, training: bool = False):
        """``(batch, window_size, n_features) -> (batch, hidden_sizes[-1])``."""
        out = x
        for rnn, drop in zip(self.rnns, self.drops, strict=True):
            out = rnn(out)
            out = drop(out, training=training)
        return out[:, -1, :]


def count_parameters(model: keras.Model) -> int:
    return int(sum(int(tf.size(v)) for v in model.trainable_variables))


def model_summary(model: keras.Model) -> str:
    lines = [f"{v.name}: {tuple(v.shape)} = {int(tf.size(v))}" for v in model.trainable_variables]
    lines.append(f"Total trainable parameters: {count_parameters(model)}")
    return "\n".join(lines)


@dataclass
class ArchitectureConfig:
    n_features: int
    hidden_sizes: tuple[int, ...] = (100, 50)
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


class _GlobalNormClipMixin(_TrainStepBase):
    """Overrides ``keras.Model.train_step`` to clip gradients by their **global** norm across
    all parameters (``tf.clip_by_global_norm``), matching
    ``torch.nn.utils.clip_grad_norm_``'s semantics exactly. Keras's built-in
    ``optimizer(clipnorm=...)`` clips each gradient *tensor* independently by its own norm,
    which is a different (and for a multi-layer LSTM, more aggressive) operation — see
    ``docs/decisions/P05-tensorflow.md`` for the measured effect of this discrepancy.
    """

    clip_norm: float | None = None

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
        # Keras's own default `train_step` updates the loss tracker explicitly at this point
        # (see `keras.src.backend.tensorflow.trainer.TensorFlowTrainer.train_step`) — overriding
        # `train_step` without this line leaves the "loss" metric permanently at its initial
        # value (0.0), silently hiding a diverging (NaN) loss from `history.history["loss"]"``
        # even though gradients were computed from the real (possibly NaN) value.
        self._loss_tracker.update_state(loss, sample_weight=tf.shape(x)[0])
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # A loss fully disconnected from the model's parameters (e.g. a constant NaN, as
        # tests/test_tf_train.py's divergence test uses to force a NaN loss without an actual
        # unstable computation) yields all-``None`` gradients; Keras's optimiser raises rather
        # than skip the step in that case. Filtering keeps the (rare, pathological) all-None
        # case a no-op step instead of a crash, and behaves identically to the unfiltered path
        # whenever every gradient is defined, which is the normal case.
        pairs = zip(gradients, trainable_vars, strict=False)
        grads_and_vars = [(g, v) for g, v in pairs if g is not None]
        if grads_and_vars:
            grads, variables = zip(*grads_and_vars, strict=True)
            if self.clip_norm:
                grads, _ = tf.clip_by_global_norm(list(grads), self.clip_norm)
            self.optimizer.apply_gradients(zip(grads, variables, strict=True))
        # `compute_metrics` updates the loss tracker plus every metric passed to `compile()`
        # and returns the standard {name: result} dict — the same bookkeeping Keras's own
        # default `train_step` does, reused here so only the gradient-clipping step differs.
        return self.compute_metrics(x, y, y_pred)


class LSTMRegressor(_GlobalNormClipMixin, keras.Model):
    """Trunk + dense head -> a single RUL value per window (no output activation)."""

    def __init__(
        self,
        n_features: int,
        hidden_sizes: tuple[int, ...] = (100, 50),
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = ArchitectureConfig(n_features, tuple(hidden_sizes), dropout)
        self.trunk = LSTMTrunk(hidden_sizes, dropout)
        self.head = keras.layers.Dense(1)
        self(tf.zeros((1, 2, n_features)))  # force variable creation so params can be counted

    def call(self, x, training: bool = False):
        """``(batch, window_size, n_features) -> (batch,)``, the predicted RUL."""
        return tf.squeeze(self.head(self.trunk(x, training=training)), axis=-1)

    def summary_text(self) -> str:
        return model_summary(self)


class LSTMClassifier(_GlobalNormClipMixin, keras.Model):
    """Same trunk, a dense head producing a single **logit** per window.

    Returns raw logits, not probabilities — pair with
    ``tf.nn.sigmoid_cross_entropy_with_logits``/``tf.nn.weighted_cross_entropy_with_logits``
    rather than a sigmoid activation plus binary cross-entropy on probabilities.
    """

    def __init__(
        self,
        n_features: int,
        hidden_sizes: tuple[int, ...] = (100, 50),
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.config = ArchitectureConfig(n_features, tuple(hidden_sizes), dropout)
        self.trunk = LSTMTrunk(hidden_sizes, dropout)
        self.head = keras.layers.Dense(1)
        self(tf.zeros((1, 2, n_features)))

    def call(self, x, training: bool = False):
        """``(batch, window_size, n_features) -> (batch,)`` logits."""
        return tf.squeeze(self.head(self.trunk(x, training=training)), axis=-1)

    def summary_text(self) -> str:
        return model_summary(self)
