import torch
import torch.nn as nn


class DenoisingAutoencoder1D(nn.Module):
    """
    Convolutional denoising autoencoder.

    Input:  (batch, 2, 1000)
    Output: (batch, 1, 1000)
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def missing_region_mse(prediction, target, mask):
    """MSE calculated only on deliberately hidden samples."""
    missing = 1.0 - mask
    squared_error = ((prediction - target) ** 2) * missing
    return squared_error.sum() / missing.sum().clamp_min(1.0)


def combine_observed_and_prediction(corrupted_signal, prediction, mask):
    """Keep observed samples unchanged and predict only the gap."""
    return mask * corrupted_signal + (1.0 - mask) * prediction
