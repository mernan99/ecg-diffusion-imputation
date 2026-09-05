import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(num_steps, s=0.008):
    """Cosine beta schedule."""
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alpha_bar = torch.cos((((x / num_steps) + s) / (1 + s)) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, 1e-5, 0.999).float()


class DiffusionSchedule:
    """Precomputed DDPM coefficients."""

    def __init__(self, num_steps=100, device="cpu"):
        self.num_steps = int(num_steps)
        self.betas = cosine_beta_schedule(self.num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_bars[:-1]], dim=0
        )

        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.posterior_variance = (
            self.betas
            * (1.0 - self.alpha_bars_prev)
            / (1.0 - self.alpha_bars)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

        # q(x_{t-1} | x_t, x_0) posterior mean coefficients.
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alpha_bars_prev)
            / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bars_prev)
            * self.sqrt_alphas
            / (1.0 - self.alpha_bars)
        )


def extract(values, timesteps, x_shape):
    batch_size = timesteps.shape[0]
    out = values.gather(0, timesteps)
    return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))


def q_sample(x0, timesteps, noise, schedule):
    sqrt_alpha_bar = extract(schedule.sqrt_alpha_bars, timesteps, x0.shape)
    sqrt_one_minus = extract(
        schedule.sqrt_one_minus_alpha_bars, timesteps, x0.shape
    )
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise


def predict_x0_from_noise(x_t, timesteps, predicted_noise, schedule):
    """Recover predicted clean signal x_0 from x_t and epsilon prediction."""
    sqrt_alpha_bar = extract(schedule.sqrt_alpha_bars, timesteps, x_t.shape)
    sqrt_one_minus = extract(
        schedule.sqrt_one_minus_alpha_bars, timesteps, x_t.shape
    )
    sqrt_alpha_bar = torch.clamp(sqrt_alpha_bar, min=1e-8)
    return (x_t - sqrt_one_minus * predicted_noise) / sqrt_alpha_bar


def clip_predicted_x0(predicted_x0, mask, clip_value=10.0):
    """
    Clip only the generated/missing region in normalized ECG units.

    A conservative +/-10 limit removes catastrophic reverse-process
    outliers while leaving ordinary normalized ECG amplitudes unchanged.
    """
    if clip_value is None:
        return predicted_x0
    if clip_value <= 0:
        raise ValueError("clip_value must be > 0 or None.")

    missing = 1.0 - mask
    clipped = torch.clamp(
        predicted_x0, min=-float(clip_value), max=float(clip_value)
    )
    return mask * predicted_x0 + missing * clipped


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -scale
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dilation=1):
        super().__init__()
        padding = 2 * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=5,
            dilation=dilation, padding=padding
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=5,
            dilation=dilation, padding=padding
        )
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x, time_embedding):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = x + self.time_projection(time_embedding)[:, :, None]
        x = F.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        return x + residual


class ConditionalDiffusionUNet1D(nn.Module):
    """Conditional 1D U-Net epsilon predictor."""

    def __init__(self, time_dim=128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.enc1 = TimeResidualBlock(3, 32, time_dim)
        self.down1 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc2 = TimeResidualBlock(64, 64, time_dim)
        self.down2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)

        self.mid1 = TimeResidualBlock(128, 128, time_dim, dilation=1)
        self.mid2 = TimeResidualBlock(128, 128, time_dim, dilation=2)
        self.mid3 = TimeResidualBlock(128, 128, time_dim, dilation=4)
        self.mid4 = TimeResidualBlock(128, 128, time_dim, dilation=8)

        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = TimeResidualBlock(128, 64, time_dim)
        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec1 = TimeResidualBlock(64, 32, time_dim)
        self.output = nn.Conv1d(32, 1, kernel_size=5, padding=2)

    def forward(self, model_input, timesteps):
        time_embedding = self.time_embedding(timesteps)
        e1 = self.enc1(model_input, time_embedding)
        x = self.down1(e1)
        e2 = self.enc2(x, time_embedding)
        x = self.down2(e2)
        x = self.mid1(x, time_embedding)
        x = self.mid2(x, time_embedding)
        x = self.mid3(x, time_embedding)
        x = self.mid4(x, time_embedding)
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, time_embedding)
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, time_embedding)
        return self.output(x)


def diffusion_training_loss(
    model, x0, observed, mask, schedule, generator=None
):
    """
    Epsilon-prediction DDPM loss on missing samples only.

    This is unchanged from the original implementation, so existing
    diffusion checkpoints remain compatible.
    """
    device = x0.device
    batch_size = x0.shape[0]

    timesteps = torch.randint(
        low=0,
        high=schedule.num_steps,
        size=(batch_size,),
        device=device,
        generator=generator,
    )
    noise = torch.randn(
        x0.shape,
        device=device,
        dtype=x0.dtype,
        generator=generator,
    )

    x_t = q_sample(x0, timesteps, noise, schedule)
    current = mask * x0 + (1.0 - mask) * x_t

    model_input = torch.cat([current, observed, mask], dim=1)
    predicted_noise = model(model_input, timesteps)

    missing = 1.0 - mask
    squared_error = ((predicted_noise - noise) ** 2) * missing
    denominator = missing.sum().clamp_min(1.0)
    return squared_error.sum() / denominator


@torch.no_grad()
def sample_conditional_ddpm(
    model,
    observed,
    mask,
    schedule,
    generator=None,
    clip_x0=10.0,
):
    """
    Stable conditional DDPM sampler.

    Instead of directly using the epsilon-based reverse mean, it:
      1. predicts epsilon,
      2. reconstructs predicted x_0,
      3. clips implausible x_0 values only in the missing region,
      4. evaluates q(x_{t-1} | x_t, x_0) using the posterior mean.

    This substantially reduces rare catastrophic amplitude explosions.
    """
    device = observed.device

    x = torch.randn(
        observed.shape,
        device=device,
        dtype=observed.dtype,
        generator=generator,
    )

    # Noise only in the missing region; observed ECG stays exact.
    x = mask * observed + (1.0 - mask) * x

    for step in reversed(range(schedule.num_steps)):
        timesteps = torch.full(
            (observed.shape[0],),
            step,
            device=device,
            dtype=torch.long,
        )

        model_input = torch.cat([x, observed, mask], dim=1)
        predicted_noise = model(model_input, timesteps)

        predicted_x0 = predict_x0_from_noise(
            x_t=x,
            timesteps=timesteps,
            predicted_noise=predicted_noise,
            schedule=schedule,
        )

        predicted_x0 = clip_predicted_x0(
            predicted_x0=predicted_x0,
            mask=mask,
            clip_value=clip_x0,
        )

        # Clean conditioning values are known exactly.
        predicted_x0 = mask * observed + (1.0 - mask) * predicted_x0

        coef1 = extract(
            schedule.posterior_mean_coef1,
            timesteps,
            x.shape,
        )
        coef2 = extract(
            schedule.posterior_mean_coef2,
            timesteps,
            x.shape,
        )

        posterior_mean = coef1 * predicted_x0 + coef2 * x

        if step > 0:
            variance = extract(
                schedule.posterior_variance,
                timesteps,
                x.shape,
            )
            noise = torch.randn(
                x.shape,
                device=device,
                dtype=x.dtype,
                generator=generator,
            )
            x = posterior_mean + torch.sqrt(variance) * noise
        else:
            x = posterior_mean

        # Re-apply the conditioning signal after every reverse step.
        x = mask * observed + (1.0 - mask) * x

    return x


@torch.no_grad()
def update_ema(ema_model, model, decay=0.995):
    """Exponential moving average of model parameters."""
    ema_parameters = dict(ema_model.named_parameters())
    model_parameters = dict(model.named_parameters())

    for name, parameter in model_parameters.items():
        ema_parameters[name].mul_(decay).add_(parameter, alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())

    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)
