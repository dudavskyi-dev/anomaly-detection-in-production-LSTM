"""P08 constraint: "No model training code imported by the serving package. Serving depends on
the bundle format only." Checked here by importing ``pdm.serving.app`` fresh (in a subprocess,
so this test's own imports of other ``pdm`` modules elsewhere in the suite can't taint the
result) and asserting none of the true training-orchestration modules ever entered
``sys.modules``.

Serving *does* transitively import each framework's architecture/training-loop module (e.g.
``pdm.models.torch.train`` — reused for its deterministic training loop's *inference*
counterparts like the LSTM autoencoder's scoring functions, which happen to live in the same
file as the training helpers). That's a deliberate, documented line, not a loophole: what
serving must never depend on is the *orchestration* that assumes a training environment, real
datasets on disk, or non-deterministic experiment sweeps — see docs/decisions/P08-serving.md
for the reasoning.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.fast

BANNED_MODULES = (
    "pdm.preprocessing.pipeline",
    "pdm.models.baseline.runner",
    "pdm.models.torch.experiments",
    "pdm.models.torch.anomaly_experiments",
    "pdm.evaluation.framework_benchmark",
    "pdm.ingestion.cmapss",
    "pdm.ingestion.ai4i",
    "pdm.ingestion.nab",
    "pdm.ingestion.replay",
    # P09/P10: the drift-check/retraining-trigger job and MLflow tracking are, like the modules
    # above, batch/orchestration code that assumes a training environment (real datasets on
    # disk, an MLflow server, a willingness to launch a multi-minute retrain) -- none of which a
    # request handler can assume or afford. The drift *metrics* the API's own dashboard needs
    # come from a shared Prometheus textfile (docker/prometheus.yml scrapes it as its own
    # target), never from importing this package. See docs/decisions/P10-monitoring.md.
    "pdm.monitoring.drift",
    "pdm.monitoring.trigger",
    "pdm.monitoring.reports",
    "pdm.monitoring.metrics",
    "pdm.tracking.mlflow_client",
    "pdm.tracking.promotion",
    "pdm.tracking.registry",
)


def test_serving_app_does_not_import_training_orchestration_modules():
    script = (
        "import sys\n"
        "import pdm.serving.app\n"
        "banned = " + repr(BANNED_MODULES) + "\n"
        "leaked = [m for m in banned if m in sys.modules]\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=None,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    leaked = [m for m in result.stdout.strip().split(",") if m]
    assert leaked == [], f"pdm.serving.app transitively imported banned module(s): {leaked}"
