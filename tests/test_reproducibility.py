"""P07 deliverable #5: load a logged run's params back from MLflow, retrain purely from those
values (not from whatever variables happened to be in scope when the run first ran), and assert
the metrics match within tolerance. If this fails, something the training path needs to
reproduce itself was never logged — the fix is to log it, not to loosen this test.

Uses tiny synthetic data and a small model (mirroring ``tests/test_torch_train.py``'s
overfit-tiny-batch setup) so this stays fast; the mechanism under test is "do the logged params
fully determine the run," not "does training converge well on real data" — that's already
covered by the real per-milestone experiment runs.
"""

import mlflow
import numpy as np
import pytest

from pdm.config import settings
from pdm.models.torch.architecture import LSTMRegressor
from pdm.models.torch.dataset import make_dataloader
from pdm.models.torch.train import set_full_determinism, train_regressor
from pdm.tracking.mlflow_client import log_metrics, log_params, start_run

pytestmark = pytest.mark.fast


@pytest.fixture()
def local_tracking(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.tracking, "uri", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setattr(settings.tracking, "experiment_name", "test-reproducibility")
    yield tmp_path


def _tiny_data(seed: int = 0, n: int = 16, window_size: int = 5, n_features: int = 3):
    rng = np.random.default_rng(seed)
    windows = rng.normal(size=(n, window_size, n_features)).astype(np.float32)
    targets = {"rul": windows.mean(axis=(1, 2)).astype(np.float32)}
    return windows, targets


def _train(
    *, seed: int, hidden: int, max_epochs: int, patience: int, learning_rate: float
) -> float:
    windows, targets = _tiny_data(seed=0)  # the data itself is fixed; only training params vary
    loader = make_dataloader(windows, targets, shuffle=True, batch_size=8, seed=seed)
    set_full_determinism(seed)
    model = LSTMRegressor(n_features=3, hidden_sizes=(hidden, hidden // 2), dropout=0.0)
    result = train_regressor(
        model,
        loader,
        loader,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
    )
    return result.best_val_score


def test_retraining_from_logged_params_reproduces_the_logged_metric(local_tracking):
    original_params = {
        "seed": 3,
        "hidden": 8,
        "max_epochs": 15,
        "patience": 15,
        "learning_rate": 0.02,
    }

    with start_run(
        run_name="repro-original", framework="pytorch", seed=original_params["seed"]
    ) as run:
        log_params(original_params)
        original_score = _train(**original_params)
        log_metrics({"val_rmse": original_score})
        run_id = run.info.run_id

    # Reload params purely from the MLflow run — not from `original_params` above — so this
    # test actually exercises "were the params logged completely and correctly," not just
    # "does training with the same Python variables give the same answer twice" (already
    # covered by tests/test_torch_train.py's determinism test).
    fetched = mlflow.get_run(run_id)
    reloaded_params = {
        "seed": int(fetched.data.params["seed"]),
        "hidden": int(fetched.data.params["hidden"]),
        "max_epochs": int(fetched.data.params["max_epochs"]),
        "patience": int(fetched.data.params["patience"]),
        "learning_rate": float(fetched.data.params["learning_rate"]),
    }
    assert reloaded_params == original_params  # nothing was lost or mangled in flatten/log/fetch

    reproduced_score = _train(**reloaded_params)

    logged_metric = fetched.data.metrics["val_rmse"]
    assert reproduced_score == pytest.approx(logged_metric, rel=1e-6)
    assert reproduced_score == pytest.approx(original_score, rel=1e-6)
