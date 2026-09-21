"""Pydantic request validation: rectangular shape, non-empty, and no NaN/Inf — the checks that
hold regardless of which bundle is loaded. Feature-count/order/window-length checks (which
*do* depend on the loaded bundle) are exercised in ``tests/test_serving_app.py`` instead."""

import pytest
from pydantic import ValidationError

from pdm.serving.schemas import BatchWindowRequest, WindowRequest

pytestmark = pytest.mark.fast


def test_valid_window_request_is_accepted():
    req = WindowRequest(feature_names=["a", "b"], window=[[1.0, 2.0], [3.0, 4.0]])
    assert req.window == [[1.0, 2.0], [3.0, 4.0]]


def test_empty_window_is_rejected():
    with pytest.raises(ValidationError, match="must not be empty"):
        WindowRequest(feature_names=["a"], window=[])


def test_empty_feature_names_is_rejected():
    with pytest.raises(ValidationError):
        WindowRequest(feature_names=[], window=[[1.0]])


def test_ragged_window_is_rejected():
    with pytest.raises(ValidationError, match="not rectangular"):
        WindowRequest(feature_names=["a", "b"], window=[[1.0, 2.0], [3.0]])


def test_nan_in_window_is_rejected():
    with pytest.raises(ValidationError, match="NaN/Inf"):
        WindowRequest(feature_names=["a"], window=[[float("nan")]])


def test_infinity_in_window_is_rejected():
    with pytest.raises(ValidationError, match="NaN/Inf"):
        WindowRequest(feature_names=["a"], window=[[float("inf")]])


def test_negative_infinity_in_window_is_rejected():
    with pytest.raises(ValidationError, match="NaN/Inf"):
        WindowRequest(feature_names=["a"], window=[[float("-inf")]])


def test_valid_batch_request_is_accepted():
    req = BatchWindowRequest(feature_names=["a"], windows=[[[1.0]], [[2.0]]])
    assert len(req.windows) == 2


def test_batch_request_rejects_nan_in_any_window():
    with pytest.raises(ValidationError, match="NaN/Inf"):
        BatchWindowRequest(feature_names=["a"], windows=[[[1.0]], [[float("nan")]]])


def test_batch_request_rejects_an_empty_batch():
    with pytest.raises(ValidationError):
        BatchWindowRequest(feature_names=["a"], windows=[])
