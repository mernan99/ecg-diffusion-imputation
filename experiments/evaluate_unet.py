from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_ptbxl_record, get_lead
from src.masking import make_block_mask, apply_mask
from src.interpolation import (
    linear_interpolate,
    cubic_spline_interpolate,
    pchip_interpolate,
)
from src.metrics import (
    evaluate_reconstruction,
    evaluate_r_peaks,
)
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
    signal_mv,
    mask,
    mean,
    std,
    device,
    combine_fn,
):
    normalized = (
        signal_mv - mean
    ) / std

    corrupted_norm = apply_mask(
        normalized,
        mask,
        fill_value=0.0
    ).astype(np.float32)

    mask_float = mask.astype(np.float32)

    model_input = np.stack(
        [
            corrupted_norm,
            mask_float
        ],
        axis=0
    )[None, ...]

    x = torch.from_numpy(
        model_input
    ).to(device)

    with torch.no_grad():
        pred_norm = model(x)

        corrupted_tensor = (
            torch.from_numpy(
                corrupted_norm
            )
            .view(1, 1, -1)
            .to(device)
        )

        mask_tensor = (
            torch.from_numpy(
                mask_float
            )
            .view(1, 1, -1)
            .to(device)
        )

        reconstruction_norm = combine_fn(
            corrupted_tensor,
            pred_norm,
            mask_tensor,
        )

    reconstruction_mv = (
        reconstruction_norm
        .squeeze()
        .cpu()
        .numpy()
        * std
        + mean
    )

    return reconstruction_mv


def add_result(
    rows,
    record_name,
    gap_seconds,
    trial,
    method,
    original,
    reconstruction,
    mask,
    fs,
):
    numerical = evaluate_reconstruction(
        original,
        reconstruction,
        mask
    )

    peak_metrics = evaluate_r_peaks(
        original,
        reconstruction,
        mask,
        fs
    )

    rows.append({
        "record": record_name,
        "gap_seconds": gap_seconds,
        "trial": trial,
        "method": method,
        **numerical,
        **peak_metrics,
    })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"

    basic_path = (
        results_dir
        / "autoencoder_best.pt"
    )

    unet_path = (
        results_dir
        / "unet_autoencoder_best.pt"
    )

    if not basic_path.exists():
        raise FileNotFoundError(
            "results/autoencoder_best.pt not found."
        )

    if not unet_path.exists():
        raise FileNotFoundError(
            "results/unet_autoencoder_best.pt not found. "
            "Run experiments/train_unet.py first."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    basic_checkpoint = torch.load(
        basic_path,
        map_location=device
    )

    unet_checkpoint = torch.load(
        unet_path,
        map_location=device
    )

    mean = float(
        unet_checkpoint["mean_mv"]
    )

    std = float(
        unet_checkpoint["std_mv"]
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

    print("Device:", device)
    print(
        "Fold-10 test records:",
        len(test_df)
    )

    gap_seconds_list = [
        0.25,
        0.5,
        1.0,
        2.0,
    ]

    rng = np.random.default_rng(
        args.seed
    )

    rows = []

    for record_index, row in (
        test_df.iterrows()
    ):

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

        for gap_seconds in (
            gap_seconds_list
        ):

            gap_samples = int(
                round(
                    gap_seconds * fs
                )
            )

            for trial in range(
                args.trials
            ):

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

                # Classical methods
                linear = linear_interpolate(
                    corrupted_mv,
                    mask
                )

                cubic = (
                    cubic_spline_interpolate(
                        corrupted_mv,
                        mask
                    )
                )

                pchip = pchip_interpolate(
                    corrupted_mv,
                    mask
                )

                for name, prediction in [
                    ("linear", linear),
                    ("cubic", cubic),
                    ("pchip", pchip),
                ]:
                    add_result(
                        rows,
                        record_path.name,
                        gap_seconds,
                        trial,
                        name,
                        signal_mv,
                        prediction,
                        mask,
                        fs,
                    )

                # Basic AE
                basic_prediction = (
                    reconstruct_with_model(
                        basic_model,
                        signal_mv,
                        mask,
                        mean,
                        std,
                        device,
                        combine_basic,
                    )
                )

                add_result(
                    rows,
                    record_path.name,
                    gap_seconds,
                    trial,
                    "basic_autoencoder",
                    signal_mv,
                    basic_prediction,
                    mask,
                    fs,
                )

                # U-Net AE
                unet_prediction = (
                    reconstruct_with_model(
                        unet_model,
                        signal_mv,
                        mask,
                        mean,
                        std,
                        device,
                        combine_unet,
                    )
                )

                add_result(
                    rows,
                    record_path.name,
                    gap_seconds,
                    trial,
                    "unet_autoencoder",
                    signal_mv,
                    unet_prediction,
                    mask,
                    fs,
                )

        if (
            (record_index + 1) % 100
            == 0
            or record_index == 0
        ):
            print(
                f"Processed "
                f"{record_index + 1}"
                f"/{len(test_df)}"
            )

    results = pd.DataFrame(rows)

    raw_path = (
        results_dir
        / "unet_test_results.csv"
    )

    results.to_csv(
        raw_path,
        index=False
    )

    summary = (
        results
        .groupby(
            [
                "method",
                "gap_seconds"
            ]
        )
        .agg(
            mean_mae=(
                "mae",
                "mean"
            ),
            mean_rmse=(
                "rmse",
                "mean"
            ),
            mean_correlation=(
                "correlation",
                "mean"
            ),
            mean_r_peak_recall=(
                "r_peak_recall",
                "mean"
            ),
            mean_r_peak_precision=(
                "r_peak_precision",
                "mean"
            ),
            mean_r_peak_f1=(
                "r_peak_f1",
                "mean"
            ),
            mean_r_peak_timing_error_ms=(
                "r_peak_timing_error_ms",
                "mean"
            ),
            n=(
                "rmse",
                "count"
            ),
        )
        .reset_index()
    )

    summary_path = (
        results_dir
        / "unet_test_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(
        "\nFold-10 comparison:\n"
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nRaw results: {raw_path}"
    )
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
