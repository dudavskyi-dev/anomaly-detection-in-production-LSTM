"""Score standardisation and fusion: train-only statistics, and the weight sweep's two
end-points reducing to "one detector alone"."""

import numpy as np
import pytest

from pdm.evaluation.fusion import ScoreStandardizer, fuse_scores

pytestmark = pytest.mark.fast


def test_standardizer_fit_on_train_gives_zero_mean_unit_std_on_train():
    train_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    standardizer = ScoreStandardizer.fit(train_scores)
    transformed = standardizer.transform(train_scores)
    assert transformed.mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed.std() == pytest.approx(1.0, abs=1e-9)


def test_standardizer_guards_against_zero_std():
    constant_scores = np.array([5.0, 5.0, 5.0])
    standardizer = ScoreStandardizer.fit(constant_scores)
    assert standardizer.std == 1.0
    # should not raise or produce inf/nan
    transformed = standardizer.transform(constant_scores)
    assert np.all(np.isfinite(transformed))


def test_standardizer_transforms_new_scores_using_frozen_train_statistics():
    standardizer = ScoreStandardizer.fit(np.array([0.0, 10.0]))  # mean=5, std=5
    transformed = standardizer.transform(np.array([5.0, 15.0]))
    np.testing.assert_allclose(transformed, [0.0, 2.0])


def test_standardizer_round_trips_through_a_dict():
    standardizer = ScoreStandardizer.fit(np.array([1.0, 2.0, 3.0]))
    restored = ScoreStandardizer.from_dict(standardizer.to_dict())
    assert restored == standardizer


def test_fuse_scores_weight_one_is_detector_a_alone():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(fuse_scores(a, b, 1.0), a)


def test_fuse_scores_weight_zero_is_detector_b_alone():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(fuse_scores(a, b, 0.0), b)


def test_fuse_scores_weight_half_is_the_average():
    a = np.array([1.0, 3.0])
    b = np.array([3.0, 1.0])
    np.testing.assert_allclose(fuse_scores(a, b, 0.5), [2.0, 2.0])


def test_fuse_scores_rejects_weight_outside_unit_interval():
    with pytest.raises(ValueError):
        fuse_scores(np.array([1.0]), np.array([1.0]), 1.5)
