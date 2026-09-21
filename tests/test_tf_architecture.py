"""TF/Keras LSTM architecture: forward-pass shapes, parameter counting, and — the headline
P05 check — exact parameter-count parity with the PyTorch mirror."""

import pytest
import tensorflow as tf

from pdm.models.tf.architecture import LSTMClassifier, LSTMRegressor, count_parameters
from pdm.models.torch.architecture import LSTMRegressor as TorchLSTMRegressor
from pdm.models.torch.architecture import count_parameters as torch_count_parameters

pytestmark = pytest.mark.fast


def test_regressor_forward_shape():
    model = LSTMRegressor(n_features=5, hidden_sizes=(8, 4), dropout=0.0)
    out = model(tf.random.normal((3, 10, 5)))
    assert out.shape == (3,)


def test_classifier_forward_shape():
    model = LSTMClassifier(n_features=5, hidden_sizes=(8, 4), dropout=0.0)
    out = model(tf.random.normal((3, 10, 5)))
    assert out.shape == (3,)


def test_single_layer_trunk_is_supported():
    model = LSTMRegressor(n_features=5, hidden_sizes=(8,), dropout=0.0)
    assert model(tf.random.normal((2, 10, 5))).shape == (2,)


def test_count_parameters_matches_manual_split_bias_lstm_formula():
    n_features, hidden = 3, 4
    model = LSTMRegressor(n_features=n_features, hidden_sizes=(hidden,), dropout=0.0)
    # kernel (4h,in), recurrent_kernel (4h,h), bias_ih (4h), bias_hh (4h) — see architecture.py's
    # SplitBiasLSTMCell docstring for why this mirrors torch.nn.LSTM's parameterisation exactly.
    lstm_params = 4 * hidden * n_features + 4 * hidden * hidden + 4 * hidden + 4 * hidden
    head_params = hidden * 1 + 1
    assert count_parameters(model) == lstm_params + head_params


def test_summary_contains_total_parameter_count():
    model = LSTMRegressor(n_features=4, hidden_sizes=(6, 3))
    text = model.summary_text()
    assert "Total trainable parameters" in text
    assert str(count_parameters(model)) in text


@pytest.mark.parametrize("hidden_sizes", [(100, 50), (8, 4), (16,), (32, 16)])
def test_parameter_count_matches_pytorch_mirror_exactly(hidden_sizes):
    """The P05 headline requirement (spec deliverable #1): the two frameworks' architectures
    must have identical parameter counts, not just "close" ones — see
    docs/decisions/P05-tensorflow.md for why a plain keras.layers.LSTM does not satisfy this."""
    n_features = 17
    tf_model = LSTMRegressor(n_features=n_features, hidden_sizes=hidden_sizes, dropout=0.2)
    torch_model = TorchLSTMRegressor(n_features=n_features, hidden_sizes=hidden_sizes, dropout=0.2)
    assert count_parameters(tf_model) == torch_count_parameters(torch_model)
