import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = dilation * 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=5,
            dilation=dilation, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=5,
            dilation=dilation, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class UNetAutoencoder1D(nn.Module):
    """
    U-Net-style deterministic ECG imputation baseline.

    Input:  (batch, 2, 1000)
      channel 0 = corrupted normalized ECG
      channel 1 = mask (1 observed, 0 missing)

    Output: (batch, 1, 1000)
    """

    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(2, 32)

        self.down1 = nn.Conv1d(
            32, 64, kernel_size=4, stride=2, padding=1
        )
        self.enc2 = ConvBlock(64, 64)

        self.down2 = nn.Conv1d(
            64, 128, kernel_size=4, stride=2, padding=1
        )

        self.bottleneck = nn.Sequential(
            DilatedResidualBlock(128, dilation=1),
            DilatedResidualBlock(128, dilation=2),
            DilatedResidualBlock(128, dilation=4),
            DilatedResidualBlock(128, dilation=8),
        )

        self.up2 = nn.ConvTranspose1d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose1d(
            64, 32, kernel_size=4, stride=2, padding=1
        )
        self.dec1 = ConvBlock(64, 32)

        self.output = nn.Conv1d(
            32, 1, kernel_size=7, padding=3
        )

    def forward(self, x):
        e1 = self.enc1(x)

        x = F.relu(self.down1(e1))
        e2 = self.enc2(x)

        x = F.relu(self.down2(e2))
        x = self.bottleneck(x)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return self.output(x)


def masked_mse_loss(prediction, target, mask):
    missing = 1.0 - mask
    error = ((prediction - target) ** 2) * missing
    return error.sum() / missing.sum().clamp_min(1.0)


def masked_derivative_loss(prediction, target, mask):
    pred_diff = prediction[:, :, 1:] - prediction[:, :, :-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]

    missing = 1.0 - mask
    derivative_mask = torch.maximum(
        missing[:, :, 1:],
        missing[:, :, :-1]
    )

    error = ((pred_diff - target_diff) ** 2) * derivative_mask
    return error.sum() / derivative_mask.sum().clamp_min(1.0)


def masked_correlation_loss(prediction, target, mask, eps=1e-8):
    missing = 1.0 - mask
    count = missing.sum(dim=2, keepdim=True).clamp_min(1.0)

    pred_mean = (prediction * missing).sum(dim=2, keepdim=True) / count
    target_mean = (target * missing).sum(dim=2, keepdim=True) / count

    pred_centered = (prediction - pred_mean) * missing
    target_centered = (target - target_mean) * missing

    covariance = (pred_centered * target_centered).sum(dim=2)
    pred_energy = (pred_centered ** 2).sum(dim=2)
    target_energy = (target_centered ** 2).sum(dim=2)

    correlation = covariance / torch.sqrt(
        pred_energy * target_energy + eps
    )
    correlation = torch.clamp(correlation, -1.0, 1.0)

    return (1.0 - correlation).mean()


def morphology_aware_loss(
    prediction,
    target,
    mask,
    derivative_weight=0.2,
    correlation_weight=0.1,
):
    mse = masked_mse_loss(prediction, target, mask)
    derivative = masked_derivative_loss(prediction, target, mask)
    correlation = masked_correlation_loss(prediction, target, mask)

    total = (
        mse
        + derivative_weight * derivative
        + correlation_weight * correlation
    )

    return total, {
        "mse": mse.detach(),
        "derivative": derivative.detach(),
        "correlation_loss": correlation.detach(),
    }


def combine_observed_and_prediction(
    corrupted_signal,
    prediction,
    mask
):
    return (
        mask * corrupted_signal
        + (1.0 - mask) * prediction
    )
