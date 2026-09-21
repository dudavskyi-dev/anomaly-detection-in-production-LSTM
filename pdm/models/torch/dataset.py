"""PyTorch ``Dataset``/``DataLoader`` over the preprocessed window arrays from
``pdm.preprocessing.pipeline``, with pinned memory and deterministic, config-seeded shuffling.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pdm.config import settings


class WindowDataset(Dataset):
    """Wraps a ``(N, window_size, n_features)`` window array plus one or more aligned targets."""

    def __init__(self, windows: np.ndarray, targets: dict[str, np.ndarray]) -> None:
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.targets = {
            name: torch.as_tensor(arr, dtype=torch.float32) for name, arr in targets.items()
        }

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"windows": self.windows[idx]}
        for name, arr in self.targets.items():
            item[name] = arr[idx]
        return item


def make_dataloader(
    windows: np.ndarray,
    targets: dict[str, np.ndarray],
    *,
    shuffle: bool,
    batch_size: int | None = None,
    seed: int | None = None,
) -> DataLoader:
    """A ``DataLoader`` whose shuffling (when ``shuffle=True``) is seeded from ``seed`` (default:
    ``config.seed``), so two runs with the same seed draw batches in the same order."""
    batch_size = settings.training.batch_size if batch_size is None else batch_size
    seed = settings.seed if seed is None else seed
    dataset = WindowDataset(windows, targets)

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=True,
    )
