from pathlib import Path
import argparse
import copy
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.ptbxl_dataset import PTBXLImputationDataset
from src.diffusion import (
    ConditionalDiffusionUNet1D,
    DiffusionSchedule,
    diffusion_training_loss,
    update_ema,
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
    schedule,
    device,
    training,
    validation_seed=123456,
    ema_model=None,
    ema_decay=0.995,
):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0

    generator = None

    if not training:
        generator = torch.Generator(
            device=device
        )
        generator.manual_seed(
            validation_seed
        )

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch in loader:
            target = batch["target"].to(
                device
            )

            # Existing PTBXLImputationDataset stores:
            # input[:, 0] = corrupted normalized ECG
            # input[:, 1] = mask
            observed = (
                batch["input"][:, 0:1, :]
                .to(device)
            )

            mask = batch["mask"].to(
                device
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            loss = diffusion_training_loss(
                model=model,
                x0=target,
                observed=observed,
                mask=mask,
                schedule=schedule,
                generator=generator,
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                if ema_model is not None:
                    update_ema(
                        ema_model,
                        model,
                        decay=ema_decay
                    )

            total_loss += loss.item()
            n_batches += 1

    return (
        total_loss
        / max(
            n_batches,
            1
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=100
    )

    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.995
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "0 is the safest setting on Windows."
        )
    )

    args = parser.parse_args()

    set_seed(args.seed)

    results_dir = (
        PROJECT_ROOT
        / "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    stats_path = (
        results_dir
        / "train_stats.json"
    )

    if not stats_path.exists():
        raise FileNotFoundError(
            "results/train_stats.json not found. "
            "Run compute_train_stats.py first."
        )

    with stats_path.open(
        "r"
    ) as f:
        stats = json.load(f)

    mean = float(
        stats["mean_mv"]
    )
    std = float(
        stats["std_mv"]
    )

    csv_path = (
        args.dataset_dir
        / "ptbxl_database.csv"
    )

    train_dataset = (
        PTBXLImputationDataset(
            data_root=args.dataset_dir,
            csv_path=csv_path,
            folds=range(1, 9),
            lead="II",
            mean=mean,
            std=std,
            deterministic_masks=False,
            seed=args.seed,
        )
    )

    val_dataset = (
        PTBXLImputationDataset(
            data_root=args.dataset_dir,
            csv_path=csv_path,
            folds=[9],
            lead="II",
            mean=mean,
            std=std,
            deterministic_masks=True,
            seed=args.seed + 100000,
        )
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
    print(
        "Training records:",
        len(train_dataset)
    )
    print(
        "Validation records:",
        len(val_dataset)
    )
    print(
        "Diffusion steps:",
        args.diffusion_steps
    )
    print(
        "EMA decay:",
        args.ema_decay
    )

    model = (
        ConditionalDiffusionUNet1D()
        .to(device)
    )

    ema_model = copy.deepcopy(
        model
    ).to(device)

    ema_model.eval()

    for parameter in (
        ema_model.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    schedule = DiffusionSchedule(
        num_steps=args.diffusion_steps,
        device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    best_val = float("inf")

    checkpoint_path = (
        results_dir
        / "diffusion_best.pt"
    )

    history_path = (
        results_dir
        / "diffusion_history.csv"
    )

    history = []

    for epoch in range(
        1,
        args.epochs + 1
    ):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            training=True,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
        )

        val_loss = run_epoch(
            model=ema_model,
            loader=val_loader,
            optimizer=None,
            schedule=schedule,
            device=device,
            training=False,
            validation_seed=987654,
        )

        history.append({
            "epoch":
                epoch,
            "train_noise_mse":
                train_loss,
            "val_noise_mse_ema":
                val_loss,
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train noise MSE="
            f"{train_loss:.6f} "
            f"| val EMA noise MSE="
            f"{val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "ema_model_state_dict":
                        ema_model.state_dict(),

                    "mean_mv":
                        mean,

                    "std_mv":
                        std,

                    "lead":
                        "II",

                    "epoch":
                        epoch,

                    "best_val_noise_mse":
                        best_val,

                    "diffusion_steps":
                        args.diffusion_steps,

                    "ema_decay":
                        args.ema_decay,
                },
                checkpoint_path
            )

            print(
                "  saved best diffusion "
                f"checkpoint "
                f"(val={best_val:.6f})"
            )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False
    )

    print(
        "\nBest validation noise MSE:",
        f"{best_val:.6f}"
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )

    print(
        "History:",
        history_path
    )


if __name__ == "__main__":
    main()
