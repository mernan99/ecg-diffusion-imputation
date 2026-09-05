import numpy as np


def make_block_mask(length, gap_samples, rng=None, margin_samples=0):
    """
    Create one contiguous missing block.

    True = observed, False = missing.
    margin_samples keeps the gap away from the signal boundaries.
    """
    if rng is None:
        rng = np.random.default_rng()

    if gap_samples <= 0:
        raise ValueError("gap_samples must be greater than 0.")
    if margin_samples < 0:
        raise ValueError("margin_samples must be >= 0.")

    min_start = margin_samples
    max_start = length - margin_samples - gap_samples

    if max_start < min_start:
        raise ValueError(
            "Signal is too short for the requested gap and margin."
        )

    start = rng.integers(min_start, max_start + 1)

    mask = np.ones(length, dtype=bool)
    mask[start:start + gap_samples] = False
    return mask


def make_random_point_mask(length, missing_fraction, rng=None):
    """Randomly hide individual samples."""
    if rng is None:
        rng = np.random.default_rng()

    if not 0 < missing_fraction < 1:
        raise ValueError("missing_fraction must be between 0 and 1.")

    n_missing = int(round(length * missing_fraction))
    missing_indices = rng.choice(length, size=n_missing, replace=False)

    mask = np.ones(length, dtype=bool)
    mask[missing_indices] = False
    return mask


def apply_mask(signal, mask, fill_value=0.0):
    """Replace missing samples with fill_value."""
    signal = np.asarray(signal)
    mask = np.asarray(mask, dtype=bool)

    if signal.shape != mask.shape:
        raise ValueError("signal and mask must have the same shape.")

    corrupted = signal.copy()
    corrupted[~mask] = fill_value
    return corrupted


def mask_to_channel(mask, dtype=np.float32):
    """Convert mask to a neural-network channel: 1 observed, 0 missing."""
    return np.asarray(mask).astype(dtype)
