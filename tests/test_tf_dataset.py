"""tf.data pipeline: correct shapes/dtypes, and seeded deterministic shuffling — the same
contract ``tests/test_torch_dataset.py`` checks for the PyTorch ``DataLoader``."""

import numpy as np
import pytest

from pdm.models.tf.dataset import collect_windows, make_tf_dataset

pytestmark = pytest.mark.fast


def test_dataset_returns_correct_shapes_and_dtypes():
    windows = np.random.default_rng(0).normal(size=(10, 5, 3)).astype(np.float32)
    targets = {"rul": np.arange(10, dtype=np.float64)}
    ds = make_tf_dataset(windows, targets, "rul", shuffle=False, batch_size=10)
    batch_x, batch_y = next(iter(ds))
    assert batch_x.shape == (10, 5, 3)
    assert batch_x.dtype.as_numpy_dtype == np.float32
    assert batch_y.dtype.as_numpy_dtype == np.float32


def test_dataset_batches_correctly():
    windows = np.zeros((17, 4, 2), dtype=np.float32)
    targets = {"rul": np.arange(17, dtype=np.float32)}
    ds = make_tf_dataset(windows, targets, "rul", shuffle=False, batch_size=5)
    assert [int(b[0].shape[0]) for b in ds] == [5, 5, 5, 2]


def test_shuffling_is_deterministic_given_seed():
    windows = np.arange(20 * 3 * 2, dtype=np.float32).reshape(20, 3, 2)
    targets = {"rul": np.arange(20, dtype=np.float32)}

    ds1 = make_tf_dataset(windows, targets, "rul", shuffle=True, batch_size=4, seed=42)
    ds2 = make_tf_dataset(windows, targets, "rul", shuffle=True, batch_size=4, seed=42)

    order1 = [b[1].numpy().tolist() for b in ds1]
    order2 = [b[1].numpy().tolist() for b in ds2]
    assert order1 == order2


def test_shuffling_differs_across_seeds():
    windows = np.arange(20 * 3 * 2, dtype=np.float32).reshape(20, 3, 2)
    targets = {"rul": np.arange(20, dtype=np.float32)}

    ds1 = make_tf_dataset(windows, targets, "rul", shuffle=True, batch_size=4, seed=1)
    ds2 = make_tf_dataset(windows, targets, "rul", shuffle=True, batch_size=4, seed=2)

    order1 = [b[1].numpy().tolist() for b in ds1]
    order2 = [b[1].numpy().tolist() for b in ds2]
    assert order1 != order2


def test_collect_windows_round_trips_a_non_shuffled_dataset():
    windows = np.random.default_rng(0).normal(size=(9, 4, 2)).astype(np.float32)
    targets = {"rul": np.arange(9, dtype=np.float32)}
    ds = make_tf_dataset(windows, targets, "rul", shuffle=False, batch_size=4)
    np.testing.assert_array_equal(collect_windows(ds), windows)
