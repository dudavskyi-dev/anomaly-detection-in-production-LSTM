"""Telemetry replay simulator: yields rows of a processed parquet file in order, as if the
system were being fed a live IoT sensor stream, with optional configurable noise and dropout.
"""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def replay(
    path: Path,
    *,
    rate_hz: float = 10.0,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    seed: int = 0,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield rows of ``path`` in file order as dicts.

    ``rate_hz <= 0`` disables the inter-record sleep, for tests and offline batch replay.
    ``noise_std`` adds independent Gaussian noise to every numeric column, in place.
    ``dropout_prob`` randomly skips a record, simulating a sensor missing a reading.
    """
    df = pd.read_parquet(path)
    rng = np.random.default_rng(seed)
    interval = 1.0 / rate_hz if rate_hz > 0 else 0.0
    # Only float columns are genuine continuous measurements. Integer columns in these datasets
    # are unit/cycle counters or binary flags/labels (e.g. AI4I's TWF/HDF/machine_failure) —
    # adding Gaussian noise to those would silently corrupt labels, not simulate sensor jitter.
    numeric_cols = df.select_dtypes(include="float").columns

    yielded = 0
    for record in df.to_dict(orient="records"):
        if limit is not None and yielded >= limit:
            break
        if dropout_prob and rng.random() < dropout_prob:
            continue
        if noise_std:
            for col in numeric_cols:
                record[col] = float(record[col]) + float(rng.normal(0.0, noise_std))
        yield record
        yielded += 1
        if interval:
            time.sleep(interval)
