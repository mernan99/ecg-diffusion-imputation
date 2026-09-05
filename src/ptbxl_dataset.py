from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data import load_ptbxl_record, get_lead
from src.masking import make_block_mask, apply_mask


def load_ptbxl_metadata(csv_path, folds):
    """Load PTB-XL metadata and keep only requested stratified folds."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = {"filename_lr", "strat_fold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    folds = set(int(f) for f in folds)
    return df[df["strat_fold"].isin(folds)].reset_index(drop=True)


class PTBXLImputationDataset(Dataset):
    """
    PTB-XL Lead-II dataset for denoising/imputation training.

    Input channels:
      0: corrupted normalized ECG
      1: observation mask (1 observed, 0 missing)

    Target:
      complete normalized ECG
    """

    def __init__(
        self,
        data_root,
        csv_path,
        folds,
        lead="II",
        gap_seconds=(0.25, 0.5, 1.0, 2.0),
        margin_seconds=1.0,
        mean=0.0,
        std=1.0,
        deterministic_masks=False,
        seed=42,
    ):
        self.data_root = Path(data_root)
        self.df = load_ptbxl_metadata(csv_path, folds)
        self.lead = lead
        self.gap_seconds = tuple(float(x) for x in gap_seconds)
        self.margin_seconds = float(margin_seconds)
        self.mean = float(mean)
        self.std = float(std)

        if self.std <= 0:
            raise ValueError("std must be greater than 0.")

        self.deterministic_masks = deterministic_masks
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def __len__(self):
        return len(self.df)

    def _rng_for_index(self, index):
        if self.deterministic_masks:
            return np.random.default_rng(self.seed + int(index))
        return self.rng

    def __getitem__(self, index):
        row = self.df.iloc[index]
        record_path = self.data_root / str(row["filename_lr"])

        ecg, metadata = load_ptbxl_record(record_path)
        signal_mv = get_lead(ecg, metadata, self.lead).astype(np.float32)
        fs = int(metadata["fs"])

        signal = (signal_mv - self.mean) / self.std
        rng = self._rng_for_index(index)
        gap_seconds = float(rng.choice(np.asarray(self.gap_seconds)))
        gap_samples = int(round(gap_seconds * fs))
        margin_samples = int(round(self.margin_seconds * fs))

        mask = make_block_mask(
            length=len(signal),
            gap_samples=gap_samples,
            rng=rng,
            margin_samples=margin_samples,
        )

        corrupted = apply_mask(signal, mask, fill_value=0.0).astype(np.float32)
        mask_float = mask.astype(np.float32)

        model_input = np.stack([corrupted, mask_float], axis=0)
        target = signal[None, :].astype(np.float32)
        mask_channel = mask_float[None, :]

        return {
            "input": torch.from_numpy(model_input),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask_channel),
            "gap_seconds": torch.tensor(gap_seconds, dtype=torch.float32),
        }
