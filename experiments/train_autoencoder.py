from pathlib import Path
import argparse
import csv
import json
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ptbxl_dataset import PTBXLImputationDataset
from src.autoencoder import DenoisingAutoencoder1D, missing_region_mse


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    total_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            if training:
                optimizer.zero_grad()

            prediction = model(x)
            loss = missing_region_mse(prediction, target, mask)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Keep at 0 on Windows initially."
    )
    args = parser.parse_args()

    set_seed(args.seed)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    stats_path = results_dir / "train_stats.json"

    if not stats_path.exists():
        raise FileNotFoundError(
            "results/train_stats.json does not exist. "
            "Run compute_train_stats.py first."
        )

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = float(stats["mean_mv"])
    std = float(stats["std_mv"])
    csv_path = args.dataset_dir / "ptbxl_database.csv"

    train_dataset = PTBXLImputationDataset(
        data_root=args.dataset_dir,
        csv_path=csv_path,
        folds=range(1, 9),
        lead="II",
        mean=mean,
        std=std,
        deterministic_masks=False,
        seed=args.seed,
    )

    val_dataset = PTBXLImputationDataset(
        data_root=args.dataset_dir,
        csv_path=csv_path,
        folds=[9],
        lead="II",
        mean=mean,
        std=std,
        deterministic_masks=True,
        seed=args.seed + 100000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Training records:", len(train_dataset))
    print("Validation records:", len(val_dataset))
    print("Training mean (mV):", mean)
    print("Training std (mV):", std)

    model = DenoisingAutoencoder1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    checkpoint_path = results_dir / "autoencoder_best.pt"
    history_path = results_dir / "autoencoder_history.csv"
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, device, training=True
        )
        val_loss = run_epoch(
            model, val_loader, None, device, training=False
        )

        history_rows.append((epoch, train_loss, val_loss))

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train MSE={train_loss:.6f} "
            f"| val MSE={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean_mv": mean,
                    "std_mv": std,
                    "lead": "II",
                    "best_val_mse": best_val,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"  saved best checkpoint (val={best_val:.6f})")

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_missing_mse", "val_missing_mse"])
        writer.writerows(history_rows)

    print(f"\nBest validation MSE: {best_val:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
