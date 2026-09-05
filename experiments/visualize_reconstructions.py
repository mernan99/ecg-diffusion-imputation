from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_ptbxl_record, get_lead
from src.masking import make_block_mask, apply_mask
from src.interpolation import (
    linear_interpolate,
    pchip_interpolate,
)
from src.autoencoder import (
    DenoisingAutoencoder1D,
    combine_observed_and_prediction,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def get_record_from_fold10(dataset_dir, record_index):
    df = pd.read_csv(dataset_dir / "ptbxl_database.csv")
    test_df = df[df["strat_fold"] == 10].reset_index(drop=True)

    if len(test_df) == 0:
        raise RuntimeError("No Fold-10 records found.")

    if record_index < 0 or record_index >= len(test_df):
        raise IndexError(
            f"record_index must be between 0 and {len(test_df) - 1}"
        )

    row = test_df.iloc[record_index]
    record_path = dataset_dir / str(row["filename_lr"])

    return record_path, len(test_df)


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
        default=0,
        help="Index within Fold 10. Default: 0"
    )

    parser.add_argument(
        "--record-path",
        type=str,
        default=None,
        help=(
            "Optional explicit record path WITHOUT extension, "
            "for example: "
            "'data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000/00001_lr'"
        )
    )

    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=1.0,
        help="Missing block length in seconds. Default: 1.0"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mask placement. Default: 42"
    )

    parser.add_argument(
        "--lead",
        type=str,
        default="II",
        help="Lead to visualise. Default: II"
    )

    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    checkpoint_path = results_dir / "autoencoder_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "results/autoencoder_best.pt not found. "
            "Run train_autoencoder.py first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    mean = float(checkpoint["mean_mv"])
    std = float(checkpoint["std_mv"])

    model = DenoisingAutoencoder1D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.record_path is not None:
        record_path = Path(args.record_path)
        total_test_records = None
    else:
        record_path, total_test_records = get_record_from_fold10(
            args.dataset_dir,
            args.record_index
        )

    ecg, metadata = load_ptbxl_record(record_path)
    signal_mv = get_lead(ecg, metadata, args.lead).astype(np.float32)

    fs = int(metadata["fs"])
    gap_samples = int(round(args.gap_seconds * fs))

    if gap_samples <= 0:
        raise ValueError("gap_seconds must be large enough to create a gap.")

    if gap_samples >= len(signal_mv):
        raise ValueError("Gap is too long for this signal.")

    rng = np.random.default_rng(args.seed)

    mask = make_block_mask(
        length=len(signal_mv),
        gap_samples=gap_samples,
        rng=rng,
        margin_samples=fs
    )

    corrupted_mv = apply_mask(
        signal_mv,
        mask,
        fill_value=0.0
    )

    # -----------------------------
    # Classical baselines
    # -----------------------------
    linear_mv = linear_interpolate(
        corrupted_mv,
        mask
    )

    pchip_mv = pchip_interpolate(
        corrupted_mv,
        mask
    )

    # -----------------------------
    # Autoencoder
    # -----------------------------
    normalized = (signal_mv - mean) / std
    corrupted_norm = apply_mask(
        normalized,
        mask,
        fill_value=0.0
    ).astype(np.float32)

    mask_float = mask.astype(np.float32)

    model_input = np.stack(
        [corrupted_norm, mask_float],
        axis=0
    )[None, ...]

    with torch.no_grad():
        x = torch.from_numpy(model_input).to(device)
        pred_norm = model(x)

        corrupted_tensor = (
            torch.from_numpy(corrupted_norm)
            .view(1, 1, -1)
            .to(device)
        )

        mask_tensor = (
            torch.from_numpy(mask_float)
            .view(1, 1, -1)
            .to(device)
        )

        reconstruction_norm = combine_observed_and_prediction(
            corrupted_tensor,
            pred_norm,
            mask_tensor
        )

    autoencoder_mv = (
        reconstruction_norm.squeeze().cpu().numpy() * std + mean
    )

    # -----------------------------
    # Plot
    # -----------------------------
    t = np.arange(len(signal_mv)) / fs
    missing_idx = np.where(~mask)[0]

    start_idx = missing_idx[0]
    end_idx = missing_idx[-1]

    # Add context around the missing region for the zoomed plot
    context_seconds = 0.5
    context_samples = int(round(context_seconds * fs))

    zoom_start = max(0, start_idx - context_samples)
    zoom_end = min(len(signal_mv) - 1, end_idx + context_samples)

    corrupted_for_plot = corrupted_mv.copy()
    corrupted_for_plot[~mask] = np.nan

    fig = plt.figure(figsize=(14, 9))

    # Full signal
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, signal_mv, label="Ground truth", linewidth=2)
    ax1.plot(t, corrupted_for_plot, label="Observed signal", linewidth=1.5)
    ax1.plot(t, linear_mv, label="Linear interpolation", linewidth=1.2)
    ax1.plot(t, pchip_mv, label="PCHIP", linewidth=1.2)
    ax1.plot(t, autoencoder_mv, label="Autoencoder", linewidth=1.5)

    ax1.axvspan(
        t[start_idx],
        t[end_idx],
        alpha=0.15,
        label="Missing region"
    )

    ax1.set_title(
        f"{record_path.name} — Lead {args.lead} — full signal"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (mV)")
    ax1.legend()

    # Zoomed missing region
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(
        t[zoom_start:zoom_end + 1],
        signal_mv[zoom_start:zoom_end + 1],
        label="Ground truth",
        linewidth=2
    )
    ax2.plot(
        t[zoom_start:zoom_end + 1],
        corrupted_for_plot[zoom_start:zoom_end + 1],
        label="Observed signal",
        linewidth=1.5
    )
    ax2.plot(
        t[zoom_start:zoom_end + 1],
        linear_mv[zoom_start:zoom_end + 1],
        label="Linear interpolation",
        linewidth=1.2
    )
    ax2.plot(
        t[zoom_start:zoom_end + 1],
        pchip_mv[zoom_start:zoom_end + 1],
        label="PCHIP",
        linewidth=1.2
    )
    ax2.plot(
        t[zoom_start:zoom_end + 1],
        autoencoder_mv[zoom_start:zoom_end + 1],
        label="Autoencoder",
        linewidth=1.5
    )

    ax2.axvspan(
        t[start_idx],
        t[end_idx],
        alpha=0.15,
        label="Missing region"
    )

    ax2.set_title(
        f"{record_path.name} — Lead {args.lead} — zoomed missing region"
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (mV)")
    ax2.legend()

    plt.tight_layout()

    output_path = results_dir / "reconstruction_visualization.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved figure to: {output_path}")

    if total_test_records is not None:
        print(
            f"Using Fold-10 record index {args.record_index} "
            f"out of {total_test_records} test records."
        )

    print(f"Record: {record_path}")
    print(f"Lead: {args.lead}")
    print(f"Gap length: {args.gap_seconds} seconds")
    print(f"Seed: {args.seed}")

    plt.show()


if __name__ == "__main__":
    main()