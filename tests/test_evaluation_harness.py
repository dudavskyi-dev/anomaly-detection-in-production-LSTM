"""The test-set-once harness: evaluating a `Split(name="test")` twice must raise, while
validation splits may be evaluated as many times as needed."""

import numpy as np
import pytest

from pdm.evaluation.harness import Split, SplitAlreadyEvaluatedError, evaluate

pytestmark = pytest.mark.fast


def _dummy_metric_fn(y_true, y_pred) -> dict:
    return {"mae": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))}


def _identity_predict(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(windows.shape[0])


def test_evaluate_computes_metrics_correctly():
    split = Split(
        name="validation",
        windows=np.array([1.0, 2.0, 3.0]),
        targets={"y": np.array([1.0, 2.0, 5.0])},
    )
    result = evaluate(_identity_predict, split, "y", _dummy_metric_fn)
    assert result["mae"] == pytest.approx((0 + 0 + 2) / 3)


def test_validation_split_can_be_evaluated_multiple_times():
    split = Split(
        name="validation", windows=np.array([1.0, 2.0]), targets={"y": np.array([1.0, 2.0])}
    )
    evaluate(_identity_predict, split, "y", _dummy_metric_fn)
    evaluate(_identity_predict, split, "y", _dummy_metric_fn)  # no error


def test_test_split_can_only_be_evaluated_once():
    split = Split(name="test", windows=np.array([1.0, 2.0]), targets={"y": np.array([1.0, 2.0])})
    evaluate(_identity_predict, split, "y", _dummy_metric_fn)
    with pytest.raises(SplitAlreadyEvaluatedError):
        evaluate(_identity_predict, split, "y", _dummy_metric_fn)


def test_a_fresh_test_split_instance_may_be_evaluated_again():
    # e.g. a new seed's run gets a brand-new Split, not a reused one
    split_a = Split(name="test", windows=np.array([1.0]), targets={"y": np.array([1.0])})
    split_b = Split(name="test", windows=np.array([2.0]), targets={"y": np.array([2.0])})
    evaluate(_identity_predict, split_a, "y", _dummy_metric_fn)
    evaluate(_identity_predict, split_b, "y", _dummy_metric_fn)  # different instance, fine
