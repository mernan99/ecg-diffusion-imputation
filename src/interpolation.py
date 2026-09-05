import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator


def _validate(signal, mask):
    signal = np.asarray(signal, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    if signal.shape != mask.shape:
        raise ValueError("signal and mask must have the same shape.")

    if signal.ndim != 1:
        raise ValueError("signal and mask must be 1D.")

    return signal, mask


def linear_interpolate(signal, mask):
    signal, mask = _validate(signal, mask)

    x = np.arange(len(signal))
    observed_x = x[mask]
    observed_y = signal[mask]

    if len(observed_x) < 2:
        raise ValueError("At least two observed samples are required.")

    reconstructed = signal.copy()
    reconstructed[~mask] = np.interp(
        x[~mask],
        observed_x,
        observed_y
    )
    return reconstructed


def cubic_spline_interpolate(signal, mask):
    signal, mask = _validate(signal, mask)

    x = np.arange(len(signal))
    observed_x = x[mask]
    observed_y = signal[mask]

    if len(observed_x) < 4:
        raise ValueError(
            "At least four observed samples are required "
            "for cubic spline interpolation."
        )

    spline = CubicSpline(
        observed_x,
        observed_y,
        extrapolate=True
    )

    reconstructed = signal.copy()
    reconstructed[~mask] = spline(
        x[~mask]
    ).astype(np.float32)
    return reconstructed


def pchip_interpolate(signal, mask):
    signal, mask = _validate(signal, mask)

    x = np.arange(len(signal))
    observed_x = x[mask]
    observed_y = signal[mask]

    if len(observed_x) < 2:
        raise ValueError("At least two observed samples are required.")

    interpolator = PchipInterpolator(
        observed_x,
        observed_y,
        extrapolate=True
    )

    reconstructed = signal.copy()
    reconstructed[~mask] = interpolator(
        x[~mask]
    ).astype(np.float32)
    return reconstructed
