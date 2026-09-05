from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_ptbxl_record, get_lead
from src.masking import make_block_mask, apply_mask
from src.interpolation import (
    linear_interpolate,
    pchip_interpolate,
)
from src.metrics import detect_r_peaks
from src.autoencoder import (
    DenoisingAutoencoder1D,
    combine_observed_and_prediction as combine_basic,
)
from src.unet_autoencoder import (
    UNetAutoencoder1D,
    combine_observed_and_prediction as combine_unet,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def reconstruct(
    model,
    combine_fn,
    signal_mv,
    mask,
    mean,
    std,
    device,
):
    normalized = (
        signal_mv - mean
    ) / std

    corrupted = apply_mask(
        normalized,
        mask,
        fill_value=0.0
    ).astype(np.float32)

    mask_float = mask.astype(
        np.float32
    )

    model_input = np.stack(
        [corrupted, mask_float],
        axis=0
    )[None, ...]

    with torch.no_grad():
        pred = model(
            torch.from_numpy(
                model_input
            ).to(device)
        )

        observed_tensor = (
            torch.from_numpy(corrupted)
            .view(1, 1, -1)
            .to(device)
        )

        mask_tensor = (
            torch.from_numpy(mask_float)
            .view(1, 1, -1)
            .to(device)
        )

        combined = combine_fn(
            observed_tensor,
            pred,
            mask_tensor,
        )

    return (
        combined.squeeze()
        .cpu()
        .numpy()
        * std
        + mean
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET
    )
    parser.add_argument(
        "--record-index",
        type=int,
        default=0
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"

    basic_checkpoint = torch.load(
        results_dir / "autoencoder_best.pt",
        map_location="cpu"
    )

    unet_checkpoint = torch.load(
        results_dir / "unet_autoencoder_best.pt",
        map_location="cpu"
    )

    mean = float(
        unet_checkpoint["mean_mv"]
    )
    std = float(
        unet_checkpoint["std_mv"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    basic_model = (
        DenoisingAutoencoder1D()
        .to(device)
    )
    basic_model.load_state_dict(
        basic_checkpoint[
            "model_state_dict"
        ]
    )
    basic_model.eval()

    unet_model = (
        UNetAutoencoder1D()
        .to(device)
    )
    unet_model.load_state_dict(
        unet_checkpoint[
            "model_state_dict"
        ]
    )
    unet_model.eval()

    df = pd.read_csv(
        args.dataset_dir
        / "ptbxl_database.csv"
    )

    test_df = (
        df[df["strat_fold"] == 10]
        .reset_index(drop=True)
    )

    if not (
        0 <= args.record_index
        < len(test_df)
    ):
        raise IndexError(
            f"record-index must be between "
            f"0 and {len(test_df) - 1}"
        )

    row = test_df.iloc[
        args.record_index
    ]

    record_path = (
        args.dataset_dir
        / str(row["filename_lr"])
    )

    ecg, metadata = (
        load_ptbxl_record(
            record_path
        )
    )

    signal_mv = get_lead(
        ecg,
        metadata,
        "II"
    ).astype(np.float32)

    fs = int(metadata["fs"])

    gap_samples = int(
        round(
            args.gap_seconds
            * fs
        )
    )

    rng = np.random.default_rng(
        args.seed
    )

    mask = make_block_mask(
        length=len(signal_mv),
        gap_samples=gap_samples,
        rng=rng,
        margin_samples=fs,
    )

    corrupted_mv = apply_mask(
        signal_mv,
        mask,
        fill_value=0.0
    )

    linear = linear_interpolate(
        corrupted_mv,
        mask
    )

    pchip = pchip_interpolate(
        corrupted_mv,
        mask
    )

    basic = reconstruct(
        basic_model,
        combine_basic,
        signal_mv,
        mask,
        mean,
        std,
        device,
    )

    unet = reconstruct(
        unet_model,
        combine_unet,
        signal_mv,
        mask,
        mean,
        std,
        device,
    )

    corrupted_plot = (
        corrupted_mv.copy()
    )
    corrupted_plot[~mask] = np.nan

    t = (
        np.arange(
            len(signal_mv)
        )
        / fs
    )

    missing = np.flatnonzero(
        ~mask
    )

    start = missing[0]
    end = missing[-1]

    context = int(
        round(
            0.5 * fs
        )
    )

    zoom_start = max(
        0,
        start - context
    )

    zoom_end = min(
        len(signal_mv),
        end + context + 1
    )

    true_peaks = detect_r_peaks(
        signal_mv,
        fs
    )

    unet_peaks = detect_r_peaks(
        unet,
        fs
    )

    true_gap_peaks = [
        p for p in true_peaks
        if not mask[p]
    ]

    unet_gap_peaks = [
        p for p in unet_peaks
        if not mask[p]
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9)
    )

    ax = axes[0]

    ax.plot(
        t,
        signal_mv,
        label="Ground truth",
        linewidth=2
    )
    ax.plot(
        t,
        corrupted_plot,
        label="Observed signal",
        linewidth=1.3
    )
    ax.plot(
        t,
        linear,
        label="Linear",
        linewidth=1.1
    )
    ax.plot(
        t,
        pchip,
        label="PCHIP",
        linewidth=1.1
    )
    ax.plot(
        t,
        basic,
        label="Basic AE",
        linewidth=1.3
    )
    ax.plot(
        t,
        unet,
        label="U-Net AE",
        linewidth=1.5
    )

    ax.axvspan(
        t[start],
        t[end],
        alpha=0.15,
        label="Missing region"
    )

    ax.set_title(
        f"{record_path.name} — Lead II — full signal"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend()

    ax = axes[1]

    sl = slice(
        zoom_start,
        zoom_end
    )

    ax.plot(
        t[sl],
        signal_mv[sl],
        label="Ground truth",
        linewidth=2
    )
    ax.plot(
        t[sl],
        corrupted_plot[sl],
        label="Observed signal",
        linewidth=1.3
    )
    ax.plot(
        t[sl],
        linear[sl],
        label="Linear",
        linewidth=1.2
    )
    ax.plot(
        t[sl],
        pchip[sl],
        label="PCHIP",
        linewidth=1.2
    )
    ax.plot(
        t[sl],
        basic[sl],
        label="Basic AE",
        linewidth=1.4
    )
    ax.plot(
        t[sl],
        unet[sl],
        label="U-Net AE",
        linewidth=1.7
    )

    ax.axvspan(
        t[start],
        t[end],
        alpha=0.15,
        label="Missing region"
    )

    if true_gap_peaks:
        ax.scatter(
            t[true_gap_peaks],
            signal_mv[true_gap_peaks],
            marker="o",
            s=55,
            label="True R-peak-like event"
        )

    if unet_gap_peaks:
        ax.scatter(
            t[unet_gap_peaks],
            unet[unet_gap_peaks],
            marker="x",
            s=65,
            label="U-Net R-peak-like event"
        )

    ax.set_title(
        f"{record_path.name} — {args.gap_seconds:g}s gap — zoom"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend()

    plt.tight_layout()

    output = (
        results_dir
        / "unet_reconstruction_visualization.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    print("Device:", device)
    print("Record:", record_path.name)
    print("Gap seconds:", args.gap_seconds)
    print("True peak-like events in gap:", len(true_gap_peaks))
    print("U-Net peak-like events in gap:", len(unet_gap_peaks))
    print("Saved:", output)

    plt.show()


if __name__ == "__main__":
    main()
