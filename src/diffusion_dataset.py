from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data import load_ptbxl_record, get_lead
from src.mask_manifest import (
    load_mask_manifest,
    mask_from_bounds,
)
from src.masking import apply_mask


class PTBXLManifestDataset(Dataset):
    """
    Dataset that recreates the exact fixed Fold-10 masks stored in
    fold10_mask_manifest.csv.

    Sequential access reuses the most recently loaded ECG because the
    manifest is grouped by record.
    """

    def __init__(
        self,
        dataset_dir,
        manifest_path,
        mean,
        std,
        lead="II",
    ):
        self.dataset_dir = Path(
            dataset_dir
        )
        self.manifest = (
            load_mask_manifest(
                manifest_path
            )
            .reset_index(drop=True)
        )

        self.mean = float(mean)
        self.std = float(std)
        self.lead = lead

        if self.std <= 0:
            raise ValueError(
                "std must be greater than zero."
            )

        self._cached_filename = None
        self._cached_signal_mv = None
        self._cached_fs = None

    def __len__(self):
        return len(self.manifest)

    def _load_signal(self, filename_lr):
        filename_lr = str(
            filename_lr
        )

        if (
            filename_lr
            == self._cached_filename
        ):
            return (
                self._cached_signal_mv,
                self._cached_fs,
            )

        record_path = (
            self.dataset_dir
            / filename_lr
        )

        ecg, metadata = (
            load_ptbxl_record(
                record_path
            )
        )

        signal_mv = get_lead(
            ecg,
            metadata,
            self.lead
        ).astype(np.float32)

        fs = int(metadata["fs"])

        self._cached_filename = (
            filename_lr
        )
        self._cached_signal_mv = (
            signal_mv
        )
        self._cached_fs = fs

        return signal_mv, fs

    def __getitem__(self, index):
        row = self.manifest.iloc[
            index
        ]

        signal_mv, fs = (
            self._load_signal(
                row["filename_lr"]
            )
        )

        mask = mask_from_bounds(
            length=len(signal_mv),
            start_sample=int(
                row["start_sample"]
            ),
            end_sample=int(
                row["end_sample"]
            ),
        )

        normalized = (
            signal_mv - self.mean
        ) / self.std

        observed = apply_mask(
            normalized,
            mask,
            fill_value=0.0
        ).astype(np.float32)

        return {
            "row_index":
                torch.tensor(
                    index,
                    dtype=torch.long
                ),

            "signal_mv":
                torch.from_numpy(
                    signal_mv[None, :]
                ),

            "target":
                torch.from_numpy(
                    normalized[
                        None, :
                    ].astype(
                        np.float32
                    )
                ),

            "observed":
                torch.from_numpy(
                    observed[
                        None, :
                    ]
                ),

            "mask":
                torch.from_numpy(
                    mask.astype(
                        np.float32
                    )[
                        None, :
                    ]
                ),

            "fs":
                torch.tensor(
                    fs,
                    dtype=torch.long
                ),
        }
