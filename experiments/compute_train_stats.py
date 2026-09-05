from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_ptbxl_record, get_lead


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lead", default="II")
    args = parser.parse_args()

    csv_path = args.dataset_dir / "ptbxl_database.csv"
    df = pd.read_csv(csv_path)
    train_df = df[df["strat_fold"].isin(range(1, 9))].reset_index(drop=True)

    print(f"Training records: {len(train_df)}")

    total_sum = 0.0
    total_squared_sum = 0.0
    total_count = 0

    for i, row in train_df.iterrows():
        record_path = args.dataset_dir / str(row["filename_lr"])
        ecg, metadata = load_ptbxl_record(record_path)
        signal = get_lead(ecg, metadata, args.lead).astype(np.float64)

        total_sum += signal.sum()
        total_squared_sum += np.square(signal).sum()
        total_count += signal.size

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(train_df)}")

    mean = total_sum / total_count
    variance = total_squared_sum / total_count - mean ** 2
    std = float(np.sqrt(max(variance, 0.0)))

    stats = {
        "lead": args.lead,
        "folds": list(range(1, 9)),
        "n_records": int(len(train_df)),
        "n_samples": int(total_count),
        "mean_mv": float(mean),
        "std_mv": std,
    }

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output = results_dir / "train_stats.json"

    with output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nTraining statistics:")
    print(json.dumps(stats, indent=2))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
