from pathlib import Path
import argparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def choose_block_start(length, gap_samples, margin_samples, rng):
    min_start = int(margin_samples)
    max_start = int(length - margin_samples - gap_samples)

    if max_start < min_start:
        raise ValueError(
            "Signal is too short for the requested gap and margin."
        )

    return int(rng.integers(min_start, max_start + 1))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed Fold-10 PTB-XL missing blocks for reproducible "
            "model comparison."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lead", type=str, default="II")
    args = parser.parse_args()

    csv_path = args.dataset_dir / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find: {csv_path}")

    df = pd.read_csv(csv_path)
    test_df = df[df["strat_fold"] == 10].reset_index(drop=True)

    print("Fold-10 records:", len(test_df))

    gap_seconds_list = [0.25, 0.5, 1.0, 2.0]
    fs = 100
    signal_length = 1000
    margin_samples = fs

    rng = np.random.default_rng(args.seed)
    rows = []

    for record_index, row in test_df.iterrows():
        filename_lr = str(row["filename_lr"])
        record_name = Path(filename_lr).name

        for gap_seconds in gap_seconds_list:
            gap_samples = int(round(gap_seconds * fs))

            for trial in range(args.trials):
                start_sample = choose_block_start(
                    signal_length,
                    gap_samples,
                    margin_samples,
                    rng,
                )
                end_sample = start_sample + gap_samples

                rows.append({
                    "record": record_name,
                    "filename_lr": filename_lr,
                    "lead": args.lead,
                    "gap_seconds": gap_seconds,
                    "trial": trial,
                    "fs": fs,
                    "signal_length": signal_length,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_seconds": start_sample / fs,
                    "end_seconds": end_sample / fs,
                    "manifest_seed": args.seed,
                })

        if (record_index + 1) % 500 == 0:
            print(f"Prepared {record_index + 1}/{len(test_df)} records")

    manifest = pd.DataFrame(rows)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "fold10_mask_manifest.csv"
    manifest.to_csv(output_path, index=False)

    expected = len(test_df) * len(gap_seconds_list) * args.trials

    print("\nFinished.")
    print("Manifest rows:", len(manifest))
    print("Expected rows:", expected)
    print("\nRows per gap:")
    print(manifest.groupby("gap_seconds").size().to_string())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
