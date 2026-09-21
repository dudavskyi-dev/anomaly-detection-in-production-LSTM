"""PyTorch LSTM autoencoder for semi-supervised anomaly detection (P06): trained only on
**healthy** windows (see ``docs/decisions/P06-anomaly.md`` for why that makes this
semi-supervised rather than unsupervised), scored by reconstruction error on unseen windows.

The encoder's hidden size *is* the bottleneck dimension — there is no separate projection layer
— so the ``latent_dim`` sweep in ``pdm.models.torch.anomaly_experiments`` directly controls how
much the network can memorise. A too-large bottleneck lets the decoder reconstruct anything,
healthy or anomalous, equally well: the identity-function collapse the ablation is designed to
find.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pdm.models.torch.train import TrainResult, rmse_score, train_model


class LSTMAutoencoder(nn.Module):
    """Encoder LSTM -> final hidden state (the bottleneck) -> repeated across time -> decoder
    LSTM -> per-timestep linear projection back to ``n_features``.

    ``forward`` returns only the reconstruction (not the latent code) so this drops into
    ``pdm.models.torch.train.train_model`` unchanged: pass ``target_col="windows"`` and a batch's
    own ``"windows"`` entry serves as both the model's input and its reconstruction target.
    """

    def __init__(self, n_features: int, window_size: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.encoder = nn.LSTM(n_features, latent_dim, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.output_layer = nn.Linear(latent_dim, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, window_size, n_features) -> (batch, latent_dim)``, the bottleneck code."""
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(batch, window_size, n_features) -> (batch, window_size, n_features)``."""
        z = self.encode(x)
        decoder_input = z.unsqueeze(1).repeat(1, self.window_size, 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output_layer(decoded)


def train_autoencoder(
    model: LSTMAutoencoder, train_loader: DataLoader, val_loader: DataLoader, **kwargs
) -> TrainResult:
    """Thin wrapper around ``pdm.models.torch.train.train_model``: reconstruction MSE loss,
    RMSE early-stopping criterion, target *is* the input — ``target_col="windows"`` reads the
    same ``"windows"`` entry every ``WindowDataset`` batch already carries, so no dataset changes
    were needed to make the shared training loop work for an autoencoder."""
    return train_model(
        model,
        train_loader,
        val_loader,
        target_col="windows",
        loss_fn=nn.MSELoss(),
        eval_score_fn=rmse_score,
        **kwargs,
    )


@torch.no_grad()
def reconstruction_error(model: LSTMAutoencoder, windows: np.ndarray) -> np.ndarray:
    """Per-window mean-squared reconstruction error — the anomaly score (higher = more
    anomalous). ``(N, window_size, n_features) -> (N,)``."""
    model.eval()
    x = torch.as_tensor(windows, dtype=torch.float32)
    recon = model(x)
    return ((recon - x) ** 2).mean(dim=(1, 2)).numpy()


@torch.no_grad()
def per_sensor_reconstruction_error(model: LSTMAutoencoder, windows: np.ndarray) -> np.ndarray:
    """Per-window, per-sensor mean-squared error over the time axis — the explainability signal
    (spec §6.3): which sensors drove a given window's anomaly score.
    ``(N, window_size, n_features) -> (N, n_features)``."""
    model.eval()
    x = torch.as_tensor(windows, dtype=torch.float32)
    recon = model(x)
    return ((recon - x) ** 2).mean(dim=1).numpy()
