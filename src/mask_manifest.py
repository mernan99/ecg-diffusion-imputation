from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "record",
    "filename_lr",
    "lead",
    "gap_seconds",
    "trial",
    "fs",
    "start_sample",
    "end_sample",
}


def mask_from_bounds(length, start_sample, end_sample):
    """Recreate one fixed boolean observation mask.

    True = observed, False = missing. end_sample is exclusive.
    """
    start_sample = int(start_sample)
    end_sample = int(end_sample)

    if start_sample < 0:
        raise ValueError("start_sample must be >= 0.")
    if end_sample > length:
        raise ValueError(
            f"end_sample={end_sample} exceeds signal length={length}."
        )
    if end_sample <= start_sample:
        raise ValueError("end_sample must be greater than start_sample.")

    mask = np.ones(length, dtype=bool)
    mask[start_sample:end_sample] = False
    return mask


def load_mask_manifest(path):
    """Load and validate a saved mask manifest CSV."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Mask manifest not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Mask manifest is missing required columns: "
            f"{sorted(missing)}"
        )

    return df
