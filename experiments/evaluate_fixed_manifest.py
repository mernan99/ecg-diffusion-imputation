from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_ptbxl_record, get_lead
from src.masking import apply_mask
from src.mask_manifest import load_mask_manifest, mask_from_bounds
from src.interpolation import (
    linear_interpolate,
    cubic_spline_interpolate,
    pchip_interpolate,
)
from src.metrics import evaluate_reconstruction, evaluate_r_peaks
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


def reconstruct_with_model(
    model,
    combine_fn,
    signal_mv,
    mask,
    mean,
    std,
    device,
):
    normalized = (signal_mv - mean) / std

    corrupted_norm = apply_mask(
        normalized,
        mask,
        fill_value=0.0,
    ).astype(np.float32)

    mask_float = mask.astype(np.float32)

    model_input = np.stack(
        [corrupted_norm, mask_float],
        axis=0,
    )[None, ...]

    with torch.no_grad():
        prediction_norm = model(
            torch.from_numpy(model_input).to(device)
        )

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

        reconstruction_norm = combine_fn(
            corrupted_tensor,
            prediction_norm,
            mask_tensor,
        )

    return (
        reconstruction_norm.squeeze().cpu().numpy() * std + mean
    )


def add_result(
    rows,
    manifest_row,
    method,
    signal_mv,
    reconstruction,
    mask,
    fs,
):
    numerical = evaluate_reconstruction(
        signal_mv,
        reconstruction,
        mask,
    )

    peaks = evaluate_r_peaks(
        signal_mv,
        reconstruction,
        mask,
        fs,
    )

    rows.append({
        "record": manifest_row["record"],
        "gap_seconds": float(manifest_row["gap_seconds"]),
        "trial": int(manifest_row["trial"]),
        "start_sample": int(manifest_row["start_sample"]),
        "end_sample": int(manifest_row["end_sample"]),
        "method": method,
        **numerical,
        **peaks,
    })


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all deterministic baselines on the exact fixed "
            "Fold-10 mask manifest."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "results" / "fold10_mask_manifest.csv",
    )
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"

    basic_path = results_dir / "autoencoder_best.pt"
    unet_path = results_dir / "unet_autoencoder_best.pt"

    if not basic_path.exists():
        raise FileNotFoundError(f"Missing: {basic_path}")
    if not unet_path.exists():
        raise FileNotFoundError(f"Missing: {unet_path}")

    manifest = load_mask_manifest(args.manifest)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    basic_checkpoint = torch.load(
        basic_path,
        map_location=device,
    )
    unet_checkpoint = torch.load(
        unet_path,
        map_location=device,
    )

    mean = float(unet_checkpoint["mean_mv"])
    std = float(unet_checkpoint["std_mv"])

    basic_model = DenoisingAutoencoder1D().to(device)
    basic_model.load_state_dict(
        basic_checkpoint["model_state_dict"]
    )
    basic_model.eval()

    unet_model = UNetAutoencoder1D().to(device)
    unet_model.load_state_dict(
        unet_checkpoint["model_state_dict"]
    )
    unet_model.eval()

    print("Device:", device)
    print("Manifest rows:", len(manifest))
    print("Unique records:", manifest["filename_lr"].nunique())

    rows = []
    grouped = manifest.groupby("filename_lr", sort=False)
    total_records = manifest["filename_lr"].nunique()

    for record_index, (filename_lr, record_rows) in enumerate(grouped, start=1):
        record_path = args.dataset_dir / str(filename_lr)

        ecg, metadata = load_ptbxl_record(record_path)
        signal_mv = get_lead(
            ecg,
            metadata,
            "II",
        ).astype(np.float32)

        fs = int(metadata["fs"])

        for _, manifest_row in record_rows.iterrows():
            mask = mask_from_bounds(
                length=len(signal_mv),
                start_sample=manifest_row["start_sample"],
                end_sample=manifest_row["end_sample"],
            )

            corrupted_mv = apply_mask(
                signal_mv,
                mask,
                fill_value=0.0,
            )

            classical = {
                "linear": linear_interpolate(corrupted_mv, mask),
                "cubic": cubic_spline_interpolate(corrupted_mv, mask),
                "pchip": pchip_interpolate(corrupted_mv, mask),
            }

            for method, prediction in classical.items():
                add_result(
                    rows,
                    manifest_row,
                    method,
                    signal_mv,
                    prediction,
                    mask,
                    fs,
                )

            basic_prediction = reconstruct_with_model(
                basic_model,
                combine_basic,
                signal_mv,
                mask,
                mean,
                std,
                device,
            )

            add_result(
                rows,
                manifest_row,
                "basic_autoencoder",
                signal_mv,
                basic_prediction,
                mask,
                fs,
            )

            unet_prediction = reconstruct_with_model(
                unet_model,
                combine_unet,
                signal_mv,
                mask,
                mean,
                std,
                device,
            )

            add_result(
                rows,
                manifest_row,
                "unet_autoencoder",
                signal_mv,
                unet_prediction,
                mask,
                fs,
            )

        if record_index == 1 or record_index % 100 == 0:
            print(f"Processed {record_index}/{total_records} records")

    results = pd.DataFrame(rows)

    raw_path = results_dir / "fixed_test_results.csv"
    results.to_csv(raw_path, index=False)

    summary = (
        results
        .groupby(["method", "gap_seconds"])
        .agg(
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_correlation=("correlation", "mean"),
            mean_r_peak_recall=("r_peak_recall", "mean"),
            mean_r_peak_precision=("r_peak_precision", "mean"),
            mean_r_peak_f1=("r_peak_f1", "mean"),
            mean_r_peak_timing_error_ms=("r_peak_timing_error_ms", "mean"),
            n_peak_evaluable=("r_peak_recall", "count"),
            n=("rmse", "count"),
        )
        .reset_index()
    )

    summary_path = results_dir / "fixed_test_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nFixed Fold-10 comparison:\n")
    print(summary.to_string(index=False))
    print(f"\nRaw results: {raw_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
