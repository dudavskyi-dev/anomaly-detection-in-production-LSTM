"""PyTorch LSTM architectures: a configurable stacked-LSTM trunk shared by the RUL regressor
and the failure classifier, matching exactly what P05's TensorFlow/Keras mirror builds
(``LSTM(100) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(1)``) so parameter counts and
behaviour can be compared 1:1 across frameworks.

Both models take input shaped ``(batch, window_size, n_features)`` and read the **last
timestep's** hidden state of the final LSTM layer as the representation fed to the dense head.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn


class LSTMTrunk(nn.Module):
    """One ``nn.LSTM`` per entry in ``hidden_sizes``, each followed by dropout on its full
    output sequence before the next layer (or the dense head) sees it — matching the Keras
    reference architecture layer-for-layer. ``dropout=0.0`` and/or a single-entry
    ``hidden_sizes`` reproduce the P04 architecture-ablation variants without a separate class.
    """

    def __init__(
        self, n_features: int, hidden_sizes: tuple[int, ...] = (100, 50), dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.output_size = self.hidden_sizes[-1]
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_size = n_features
        for h in self.hidden_sizes:
            self.layers.append(nn.LSTM(in_size, h, batch_first=True))
            self.dropouts.append(nn.Dropout(dropout))
            in_size = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, window_size, n_features) -> (batch, hidden_sizes[-1])``."""
        out = x
        for lstm, drop in zip(self.layers, self.dropouts, strict=True):
            out, _ = lstm(out)
            out = drop(out)
        return out[:, -1, :]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    lines = [f"{name}: {tuple(p.shape)} = {p.numel()}" for name, p in model.named_parameters()]
    lines.append(f"Total trainable parameters: {count_parameters(model)}")
    return "\n".join(lines)


@dataclass
class ArchitectureConfig:
    n_features: int
    hidden_sizes: tuple[int, ...] = (100, 50)
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


class LSTMRegressor(nn.Module):
    """Trunk + dense head -> a single RUL value per window (no output activation)."""

    def __init__(
        self, n_features: int, hidden_sizes: tuple[int, ...] = (100, 50), dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.config = ArchitectureConfig(n_features, tuple(hidden_sizes), dropout)
        self.trunk = LSTMTrunk(n_features, hidden_sizes, dropout)
        self.head = nn.Linear(self.trunk.output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, window_size, n_features) -> (batch,)``, the predicted RUL."""
        return self.head(self.trunk(x)).squeeze(-1)

    def summary(self) -> str:
        return model_summary(self)


class LSTMClassifier(nn.Module):
    """Same trunk, a dense head producing a single **logit** per window.

    Returns raw logits, not probabilities — pair with ``nn.BCEWithLogitsLoss`` (numerically
    stable, and the natural place to apply ``pos_weight`` for class imbalance) rather than a
    sigmoid activation plus ``nn.BCELoss``.
    """

    def __init__(
        self, n_features: int, hidden_sizes: tuple[int, ...] = (100, 50), dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.config = ArchitectureConfig(n_features, tuple(hidden_sizes), dropout)
        self.trunk = LSTMTrunk(n_features, hidden_sizes, dropout)
        self.head = nn.Linear(self.trunk.output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, window_size, n_features) -> (batch,)`` logits."""
        return self.head(self.trunk(x)).squeeze(-1)

    def summary(self) -> str:
        return model_summary(self)
