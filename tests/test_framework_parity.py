"""Cross-framework equivalence checks (P05 spec deliverables #3 and #5).

Deliverable #3's byte-identical-inputs check uses small synthetic arrays (no real dataset
needed, kept fast) — its job is to prove neither loader reorders, casts, or otherwise mutates
the array it's handed, not to exercise real C-MAPSS data.

Deliverable #5's equivalence test reads the **real**, already-written benchmark artifact from
``artifacts/framework_benchmark/`` (produced by ``pdm.evaluation.framework_benchmark`` — see
``docs/decisions/P05-tensorflow.md`` for how it was run) rather than retraining inside the test
suite: a full 5-seed-per-framework run takes on the order of tens of minutes on this CPU-only
machine, the same reason ``tests/test_torch_experiments.py`` doesn't re-run P04's real sweeps
either. It skips (not fails) when that artifact doesn't exist, e.g. on a fresh clone before
`pdm benchmark` has been run.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pdm.models.tf.dataset import collect_windows, make_tf_dataset
from pdm.models.torch.dataset import make_dataloader

pytestmark = pytest.mark.fast

BENCHMARK_ARTIFACT = Path("artifacts/framework_benchmark/FD001.json")

# Documented tolerance for deliverable #5: 5% relative, per the spec's suggested starting point.
# Both frameworks train the same parameter-count-identical architecture (see
# pdm.models.tf.architecture's SplitBiasLSTMCell) on the exact same preprocessed windows/targets
# and seeds; a gap beyond 5% would point at a real, uninvestigated difference (initialisation
# distribution, gradient-clipping semantics, optimiser numerics) rather than expected run-to-run
# noise, and should be chased down rather than absorbed by widening this number further.
RMSE_RELATIVE_TOLERANCE = 0.05


def test_torch_and_tf_loaders_see_byte_identical_windows():
    """Neither framework's data pipeline may reorder, cast, or mutate the windows array it's
    handed — both must be reading the *same* preprocessed arrays from P02, not a reimplemented
    copy (spec deliverable #3)."""
    rng = np.random.default_rng(0)
    n, window_size, n_features = 23, 6, 4
    windows = rng.normal(size=(n, window_size, n_features)).astype(np.float32)
    targets = {"rul": rng.normal(size=n).astype(np.float32)}

    torch_loader = make_dataloader(windows, targets, shuffle=False, batch_size=1000)
    torch_windows = torch.cat([b["windows"] for b in torch_loader]).numpy()

    tf_ds = make_tf_dataset(windows, targets, "rul", shuffle=False, batch_size=1000)
    tf_windows = collect_windows(tf_ds)

    np.testing.assert_array_equal(torch_windows, windows)
    np.testing.assert_array_equal(tf_windows, windows)
    np.testing.assert_array_equal(torch_windows, tf_windows)
    assert torch_windows.dtype == tf_windows.dtype == np.float32


def test_torch_and_tf_loaders_see_byte_identical_targets():
    rng = np.random.default_rng(1)
    n, window_size, n_features = 17, 5, 3
    windows = rng.normal(size=(n, window_size, n_features)).astype(np.float32)
    targets = {"rul": rng.normal(size=n).astype(np.float32)}

    torch_loader = make_dataloader(windows, targets, shuffle=False, batch_size=1000)
    torch_targets = torch.cat([b["rul"] for b in torch_loader]).numpy()

    tf_ds = make_tf_dataset(windows, targets, "rul", shuffle=False, batch_size=1000)
    tf_targets = np.concatenate([b[1].numpy() for b in tf_ds])

    np.testing.assert_array_equal(torch_targets, targets["rul"])
    np.testing.assert_array_equal(tf_targets, targets["rul"])


def test_frameworks_agree_on_test_rmse_within_documented_tolerance():
    if not BENCHMARK_ARTIFACT.exists():
        pytest.skip(
            f"{BENCHMARK_ARTIFACT} not found — run `pdm benchmark` to produce it "
            "(a real, multi-seed training run; not exercised inside the test suite)."
        )
    payload = json.loads(BENCHMARK_ARTIFACT.read_text(encoding="utf-8"))
    torch_rmse = payload["regression"]["pytorch"]["test_rmse"]["mean"]
    tf_rmse = payload["regression"]["tensorflow"]["test_rmse"]["mean"]

    relative_gap = abs(torch_rmse - tf_rmse) / min(torch_rmse, tf_rmse)
    assert relative_gap <= RMSE_RELATIVE_TOLERANCE, (
        f"PyTorch test RMSE {torch_rmse:.3f} vs TensorFlow test RMSE {tf_rmse:.3f} "
        f"differ by {relative_gap:.1%}, beyond the documented {RMSE_RELATIVE_TOLERANCE:.0%} "
        "tolerance — see docs/decisions/P05-tensorflow.md for the investigation this should "
        "trigger rather than silently widening the tolerance."
    )
