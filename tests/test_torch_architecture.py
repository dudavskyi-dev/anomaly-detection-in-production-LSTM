"""LSTM architecture: forward-pass shapes, parameter counting, and the configurable trunk
depth/dropout the P04 ablation experiments rely on."""

import pytest
import torch

from pdm.models.torch.architecture import LSTMClassifier, LSTMRegressor, count_parameters

pytestmark = pytest.mark.fast


def test_regressor_forward_shape():
    model = LSTMRegressor(n_features=5, hidden_sizes=(8, 4), dropout=0.0)
    out = model(torch.randn(3, 10, 5))
    assert out.shape == (3,)


def test_classifier_forward_shape():
    model = LSTMClassifier(n_features=5, hidden_sizes=(8, 4), dropout=0.0)
    out = model(torch.randn(3, 10, 5))
    assert out.shape == (3,)


def test_single_layer_trunk_is_supported_for_the_architecture_ablation():
    model = LSTMRegressor(n_features=5, hidden_sizes=(8,), dropout=0.0)
    assert model(torch.randn(2, 10, 5)).shape == (2,)


def test_count_parameters_matches_manual_lstm_formula():
    n_features, hidden = 3, 4
    model = LSTMRegressor(n_features=n_features, hidden_sizes=(hidden,), dropout=0.0)
    # PyTorch LSTM: weight_ih (4h,in), weight_hh (4h,h), bias_ih (4h), bias_hh (4h)
    lstm_params = 4 * hidden * n_features + 4 * hidden * hidden + 4 * hidden + 4 * hidden
    head_params = hidden * 1 + 1  # nn.Linear(hidden, 1)
    assert count_parameters(model) == lstm_params + head_params


def test_summary_contains_total_parameter_count():
    model = LSTMRegressor(n_features=4, hidden_sizes=(6, 3))
    text = model.summary()
    assert "Total trainable parameters" in text
    assert str(count_parameters(model)) in text
