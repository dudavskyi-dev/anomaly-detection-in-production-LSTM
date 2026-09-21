"""Score fusion between two anomaly detectors (P06 deliverable #4): a weighted combination of
the standardised LSTM-autoencoder reconstruction error and the standardised Isolation Forest
score.

Raw reconstruction error (an unbounded, right-skewed MSE) and IF's score (roughly
zero-centred, from an average path length) live on completely different, arbitrary scales — a
window with reconstruction error 4.0 is not "as anomalous as" one with IF score 4.0. Averaging
them directly would let whichever detector happens to produce numerically larger raw scores
dominate the fusion regardless of which one is actually more discriminative. Standardising each
detector's scores first (z-score, using **training-score** mean/std, saved at train time — never
recomputed on validation/test, the same train-only-statistics discipline the feature scaler
uses) puts both on a comparable "how many training-score standard deviations above typical" axis
before they're combined.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pdm.models.bundle import write_json


@dataclass
class ScoreStandardizer:
    """Frozen mean/std of one detector's scores on its own training data, for z-scoring later
    scores from the same detector — never refit on validation or test."""

    mean: float
    std: float

    @classmethod
    def fit(cls, train_scores: np.ndarray) -> "ScoreStandardizer":
        std = float(np.std(train_scores))
        return cls(mean=float(np.mean(train_scores)), std=std if std > 0 else 1.0)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return (np.asarray(scores, dtype=float) - self.mean) / self.std

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "ScoreStandardizer":
        return cls(**data)


def fuse_scores(score_a_std: np.ndarray, score_b_std: np.ndarray, weight: float) -> np.ndarray:
    """``weight * a + (1 - weight) * b``, both already standardised. ``weight=1`` is "detector A
    alone", ``weight=0`` is "detector B alone" — the two ends of the sweep this module's caller
    (``pdm.models.torch.anomaly_experiments.fusion_weight_sweep``) walks over."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    return weight * np.asarray(score_a_std, dtype=float) + (1 - weight) * np.asarray(
        score_b_std, dtype=float
    )
