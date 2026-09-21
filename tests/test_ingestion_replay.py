"""The replay simulator must preserve row order, only skip (never reorder) on dropout, only
perturb genuinely continuous columns on noise, and respect `limit`."""

import pandas as pd
import pytest

from pdm.ingestion.replay import replay

pytestmark = pytest.mark.fast


def test_replay_preserves_order(tmp_path):
    df = pd.DataFrame({"unit": [1, 1, 2], "cycle": [1, 2, 1], "sensor_1": [0.1, 0.2, 0.3]})
    path = tmp_path / "sample.parquet"
    df.to_parquet(path, index=False)

    records = list(replay(path, rate_hz=0))
    assert [(r["unit"], r["cycle"]) for r in records] == [(1, 1), (1, 2), (2, 1)]


def test_replay_dropout_skips_but_never_reorders(tmp_path):
    df = pd.DataFrame({"i": list(range(10))})
    path = tmp_path / "sample.parquet"
    df.to_parquet(path, index=False)

    records = list(replay(path, rate_hz=0, dropout_prob=0.5, seed=42))
    indices = [r["i"] for r in records]
    assert indices == sorted(indices)
    assert len(indices) < 10, "expected at least one drop out of 10 draws at p=0.5"


def test_replay_noise_only_touches_float_columns(tmp_path):
    df = pd.DataFrame({"flag": [0, 1, 0], "value": [1.0, 2.0, 3.0]})
    path = tmp_path / "sample.parquet"
    df.to_parquet(path, index=False)

    records = list(replay(path, rate_hz=0, noise_std=100.0, seed=0))
    assert [r["flag"] for r in records] == [0, 1, 0]
    assert any(r["value"] != orig for r, orig in zip(records, [1.0, 2.0, 3.0], strict=True))


def test_replay_respects_limit(tmp_path):
    df = pd.DataFrame({"i": list(range(10))})
    path = tmp_path / "sample.parquet"
    df.to_parquet(path, index=False)

    records = list(replay(path, rate_hz=0, limit=3))
    assert len(records) == 3
    assert [r["i"] for r in records] == [0, 1, 2]
