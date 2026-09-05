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
from src.metrics import evaluate_reconstruction
from src.autoencoder import (
    DenoisingAutoencoder1D,
    combine_observed_and_prediction,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    checkpoint_path = results_dir / "autoencoder_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "autoencoder_best.pt not found. Run train_autoencoder.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    mean = float(checkpoint["mean_mv"])
    std = float(checkpoint["std_mv"])

    model = DenoisingAutoencoder1D().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(args.dataset_dir / "ptbxl_database.csv")
    test_df = df[df["strat_fold"] == 10].reset_index(drop=True)

    print("Device:", device)
    print("Fold-10 test records:", len(test_df))

    gap_seconds_list = [0.25, 0.5, 1.0, 2.0]
    rng = np.random.default_rng(args.seed)
    rows = []

    with torch.no_grad():
        for record_index, row in test_df.iterrows():
            record_path = args.dataset_dir / str(row["filename_lr"])
            ecg, metadata = load_ptbxl_record(record_path)
            signal_mv = get_lead(ecg, metadata, "II").astype(np.float32)
            fs = int(metadata["fs"])
            normalized = (signal_mv - mean) / std

            for gap_seconds in gap_seconds_list:
                gap_samples = int(round(gap_seconds * fs))

                for trial in range(args.trials):
                    mask = make_block_mask(
                        length=len(signal_mv),
                        gap_samples=gap_samples,
                        rng=rng,
                        margin_samples=fs,
                    )

                    # Same missing gap for all methods.
                    corrupted_mv = apply_mask(signal_mv, mask, fill_value=0.0)

                    baseline_predictions = {
                        "linear": linear_interpolate(corrupted_mv, mask),
                        "cubic": cubic_spline_interpolate(corrupted_mv, mask),
                        "pchip": pchip_interpolate(corrupted_mv, mask),
                    }

                    for method, prediction in baseline_predictions.items():
                        metrics = evaluate_reconstruction(signal_mv, prediction, mask)
                        rows.append({
                            "record": record_path.name,
                            "gap_seconds": gap_seconds,
                            "trial": trial,
                            "method": method,
                            **metrics,
                        })

                    # Autoencoder uses normalized input.
                    corrupted_norm = apply_mask(normalized, mask, fill_value=0.0).astype(np.float32)
                    mask_float = mask.astype(np.float32)

                    model_input = np.stack(
                        [corrupted_norm, mask_float], axis=0
                    )[None, ...]

                    x = torch.from_numpy(model_input).to(device)
                    pred_norm = model(x)

                    corrupted_tensor = torch.from_numpy(corrupted_norm).view(1, 1, -1).to(device)
                    mask_tensor = torch.from_numpy(mask_float).view(1, 1, -1).to(device)

                    reconstruction_norm = combine_observed_and_prediction(
                        corrupted_tensor,
                        pred_norm,
                        mask_tensor,
                    )

                    reconstruction_mv = (
                        reconstruction_norm.squeeze().cpu().numpy() * std + mean
                    )

                    metrics = evaluate_reconstruction(
                        signal_mv,
                        reconstruction_mv,
                        mask,
                    )

                    rows.append({
                        "record": record_path.name,
                        "gap_seconds": gap_seconds,
                        "trial": trial,
                        "method": "autoencoder",
                        **metrics,
                    })

            if (record_index + 1) % 100 == 0 or record_index == 0:
                print(f"Processed {record_index + 1}/{len(test_df)}")

    results = pd.DataFrame(rows)
    raw_path = results_dir / "autoencoder_test_results.csv"
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
            n=("rmse", "count"),
        )
        .reset_index()
    )

    summary_path = results_dir / "autoencoder_test_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nFinal Fold-10 comparison:\n")
    print(summary.to_string(index=False))
    print(f"\nRaw results: {raw_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
