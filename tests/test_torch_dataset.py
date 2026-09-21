"""WindowDataset/DataLoader: correct shapes/dtypes, and seeded deterministic shuffling."""

import numpy as np
import pytest
import torch

from pdm.models.torch.dataset import WindowDataset, make_dataloader

pytestmark = pytest.mark.fast


def test_dataset_returns_correct_shapes_and_dtypes():
    windows = np.random.default_rng(0).normal(size=(10, 5, 3)).astype(np.float32)
    targets = {"rul": np.arange(10, dtype=np.float64)}
    ds = WindowDataset(windows, targets)
    item = ds[0]
    assert item["windows"].shape == (5, 3)
    assert item["windows"].dtype == torch.float32
    assert item["rul"].dtype == torch.float32


def test_dataloader_batches_correctly():
    windows = np.zeros((17, 4, 2), dtype=np.float32)
    targets = {"rul": np.arange(17, dtype=np.float32)}
    loader = make_dataloader(windows, targets, shuffle=False, batch_size=5)
    assert [b["windows"].shape[0] for b in loader] == [5, 5, 5, 2]


def test_shuffling_is_deterministic_given_seed():
    windows = np.arange(20 * 3 * 2, dtype=np.float32).reshape(20, 3, 2)
    targets = {"rul": np.arange(20, dtype=np.float32)}

    loader1 = make_dataloader(windows, targets, shuffle=True, batch_size=4, seed=42)
    loader2 = make_dataloader(windows, targets, shuffle=True, batch_size=4, seed=42)

    assert [b["rul"].tolist() for b in loader1] == [b["rul"].tolist() for b in loader2]


def test_shuffling_differs_across_seeds():
    windows = np.arange(20 * 3 * 2, dtype=np.float32).reshape(20, 3, 2)
    targets = {"rul": np.arange(20, dtype=np.float32)}

    loader1 = make_dataloader(windows, targets, shuffle=True, batch_size=4, seed=1)
    loader2 = make_dataloader(windows, targets, shuffle=True, batch_size=4, seed=2)

    assert [b["rul"].tolist() for b in loader1] != [b["rul"].tolist() for b in loader2]
