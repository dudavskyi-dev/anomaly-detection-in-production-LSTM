"""``tf.data`` pipeline over the same preprocessed window arrays
``pdm.models.torch.dataset.make_dataloader`` consumes — same files, same arrays, no
TF-side reimplementation of preprocessing. Shuffling is seeded so two runs with the same seed
draw batches in the same order, mirroring the PyTorch ``DataLoader``'s determinism contract.
"""

import numpy as np
import tensorflow as tf

from pdm.config import settings


def make_tf_dataset(
    windows: np.ndarray,
    targets: dict[str, np.ndarray],
    target_col: str,
    *,
    shuffle: bool,
    batch_size: int | None = None,
    seed: int | None = None,
) -> tf.data.Dataset:
    """A ``(windows, target)`` dataset, batched, whose shuffling (when ``shuffle=True``) is
    seeded from ``seed`` (default: ``config.seed``) — the same contract as
    ``pdm.models.torch.dataset.make_dataloader``.
    """
    batch_size = settings.training.batch_size if batch_size is None else batch_size
    seed = settings.seed if seed is None else seed

    windows = windows.astype(np.float32)
    target = targets[target_col].astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((windows, target))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(windows), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def collect_windows(ds: tf.data.Dataset) -> np.ndarray:
    """Concatenate every batch's windows back into one ``(N, window_size, n_features)`` array —
    used by the byte-identical-inputs test to compare against the PyTorch loader's view of the
    same data without depending on either framework's internal batching.
    """
    return np.concatenate([batch[0].numpy() for batch in ds], axis=0)
