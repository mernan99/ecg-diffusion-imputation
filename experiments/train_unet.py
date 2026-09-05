from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ptbxl_dataset import PTBXLImputationDataset
from src.unet_autoencoder import (
    UNetAutoencoder1D,
    morphology_aware_loss,
)


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


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training,
    derivative_weight,
    correlation_weight,
):
    model.train() if training else model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_derivative = 0.0
    total_correlation = 0.0
    n_batches = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch in loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            if training:
                optimizer.zero_grad()

            prediction = model(x)

            loss, parts = morphology_aware_loss(
                prediction,
                target,
                mask,
                derivative_weight=derivative_weight,
                correlation_weight=correlation_weight,
            )

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_mse += parts["mse"].item()
            total_derivative += parts["derivative"].item()
            total_correlation += parts["correlation_loss"].item()
            n_batches += 1

    n_batches = max(n_batches, 1)

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "derivative": total_derivative / n_batches,
        "correlation_loss": total_correlation / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--derivative-weight",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--correlation-weight",
        type=float,
        default=0.1
    )

    args = parser.parse_args()

    set_seed(args.seed)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    stats_path = results_dir / "train_stats.json"

    if not stats_path.exists():
        raise FileNotFoundError(
            "results/train_stats.json does not exist. "
            "Run experiments/compute_train_stats.py first."
        )

    with stats_path.open("r") as f:
        stats = json.load(f)

    mean = float(stats["mean_mv"])
    std = float(stats["std_mv"])

    csv_path = (
        args.dataset_dir
        / "ptbxl_database.csv"
    )

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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Training records:", len(train_dataset))
    print("Validation records:", len(val_dataset))
    print("Derivative weight:", args.derivative_weight)
    print("Correlation weight:", args.correlation_weight)

    model = UNetAutoencoder1D().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val = float("inf")

    checkpoint_path = (
        results_dir
        / "unet_autoencoder_best.pt"
    )

    history_path = (
        results_dir
        / "unet_autoencoder_history.csv"
    )

    history_rows = []

    for epoch in range(1, args.epochs + 1):

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True,
            derivative_weight=args.derivative_weight,
            correlation_weight=args.correlation_weight,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            training=False,
            derivative_weight=args.derivative_weight,
            correlation_weight=args.correlation_weight,
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_derivative": train_metrics["derivative"],
            "train_correlation_loss": train_metrics["correlation_loss"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_derivative": val_metrics["derivative"],
            "val_correlation_loss": val_metrics["correlation_loss"],
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train={train_metrics['loss']:.6f} "
            f"| val={val_metrics['loss']:.6f} "
            f"| val MSE={val_metrics['mse']:.6f} "
            f"| val corr-loss={val_metrics['correlation_loss']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "mean_mv": mean,
                    "std_mv": std,
                    "lead": "II",
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "derivative_weight":
                        args.derivative_weight,
                    "correlation_weight":
                        args.correlation_weight,
                },
                checkpoint_path
            )

            print(
                f"  saved best checkpoint "
                f"(val={best_val:.6f})"
            )

    import pandas as pd

    pd.DataFrame(
        history_rows
    ).to_csv(
        history_path,
        index=False
    )

    print(f"\nBest validation loss: {best_val:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
