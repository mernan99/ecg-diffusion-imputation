from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.diffusion import (
    ConditionalDiffusionUNet1D,
    DiffusionSchedule,
    sample_conditional_ddpm,
)
from src.diffusion_dataset import (
    PTBXLManifestDataset,
)
from src.metrics import (
    evaluate_reconstruction,
    evaluate_r_peaks,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Sampling seed. Keep fixed for reproducible "
            "stochastic diffusion evaluation."
        )
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Optional quick-test limit. "
            "Omit for the full fixed Fold-10 benchmark."
        )
    )

    parser.add_argument(
        "--samples-per-input",
        type=int,
        default=1,
        help=(
            "Number of diffusion samples to average for each "
            "test gap. Default 1 for the full benchmark."
        )
    )

    args = parser.parse_args()

    if args.samples_per_input < 1:
        raise ValueError(
            "--samples-per-input must be >= 1."
        )

    results_dir = (
        PROJECT_ROOT
        / "results"
    )

    checkpoint_path = (
        results_dir
        / "diffusion_best.pt"
    )

    manifest_path = (
        results_dir
        / "fold10_mask_manifest.csv"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "results/diffusion_best.pt not found. "
            "Run experiments/train_diffusion.py first."
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "results/fold10_mask_manifest.csv not found."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    mean = float(
        checkpoint["mean_mv"]
    )

    std = float(
        checkpoint["std_mv"]
    )

    diffusion_steps = int(
        checkpoint[
            "diffusion_steps"
        ]
    )

    model = (
        ConditionalDiffusionUNet1D()
        .to(device)
    )

    # Use EMA weights for diffusion sampling.
    model.load_state_dict(
        checkpoint[
            "ema_model_state_dict"
        ]
    )

    model.eval()

    schedule = DiffusionSchedule(
        num_steps=diffusion_steps,
        device=device
    )

    full_dataset = (
        PTBXLManifestDataset(
            dataset_dir=args.dataset_dir,
            manifest_path=manifest_path,
            mean=mean,
            std=std,
            lead="II",
        )
    )

    manifest = (
        full_dataset.manifest
    )

    if args.max_rows is not None:
        n_rows = min(
            int(args.max_rows),
            len(full_dataset)
        )

        dataset = Subset(
            full_dataset,
            range(n_rows)
        )

        suffix = "_quick"

    else:
        dataset = full_dataset
        suffix = ""

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Device:", device)
    print(
        "Diffusion steps:",
        diffusion_steps
    )
    print(
        "Rows to evaluate:",
        len(dataset)
    )
    print(
        "Samples per input:",
        args.samples_per_input
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        args.seed
    )

    rows = []

    processed = 0

    for batch in loader:
        observed = (
            batch["observed"]
            .to(device)
        )

        mask = (
            batch["mask"]
            .to(device)
        )

        sample_sum = None

        for _ in range(
            args.samples_per_input
        ):
            sample = (
                sample_conditional_ddpm(
                    model=model,
                    observed=observed,
                    mask=mask,
                    schedule=schedule,
                    generator=generator,
                )
            )

            if sample_sum is None:
                sample_sum = sample
            else:
                sample_sum = (
                    sample_sum
                    + sample
                )

        reconstruction_norm = (
            sample_sum
            / args.samples_per_input
        )

        reconstruction_mv = (
            reconstruction_norm
            * std
            + mean
        )

        signal_mv = (
            batch["signal_mv"]
        )

        row_indices = (
            batch["row_index"]
            .cpu()
            .numpy()
        )

        reconstruction_np = (
            reconstruction_mv
            .detach()
            .cpu()
            .numpy()
        )

        signal_np = (
            signal_mv
            .cpu()
            .numpy()
        )

        mask_np = (
            batch["mask"]
            .cpu()
            .numpy()
            .astype(bool)
        )

        fs_np = (
            batch["fs"]
            .cpu()
            .numpy()
        )

        for j, row_index in (
            enumerate(
                row_indices
            )
        ):
            metadata_row = (
                manifest.iloc[
                    int(row_index)
                ]
            )

            original = (
                signal_np[j, 0]
            )

            reconstruction = (
                reconstruction_np[j, 0]
            )

            observation_mask = (
                mask_np[j, 0]
            )

            fs = int(
                fs_np[j]
            )

            numerical = (
                evaluate_reconstruction(
                    original,
                    reconstruction,
                    observation_mask,
                )
            )

            peaks = evaluate_r_peaks(
                original,
                reconstruction,
                observation_mask,
                fs,
            )

            rows.append({
                "record":
                    metadata_row[
                        "record"
                    ],

                "gap_seconds":
                    float(
                        metadata_row[
                            "gap_seconds"
                        ]
                    ),

                "trial":
                    int(
                        metadata_row[
                            "trial"
                        ]
                    ),

                "start_sample":
                    int(
                        metadata_row[
                            "start_sample"
                        ]
                    ),

                "end_sample":
                    int(
                        metadata_row[
                            "end_sample"
                        ]
                    ),

                "method":
                    "diffusion",

                "samples_per_input":
                    args.samples_per_input,

                **numerical,
                **peaks,
            })

        processed += len(
            row_indices
        )

        if (
            processed % 1000
            < len(row_indices)
            or processed
            == len(dataset)
        ):
            print(
                f"Processed "
                f"{processed}"
                f"/{len(dataset)}"
            )

    results = pd.DataFrame(
        rows
    )

    raw_path = (
        results_dir
        / f"diffusion_test_results{suffix}.csv"
    )

    results.to_csv(
        raw_path,
        index=False
    )

    def finite_count(series):
        return int(
            np.isfinite(
                series.to_numpy(
                    dtype=float
                )
            ).sum()
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
            std_mae=(
                "mae",
                "std"
            ),
            mean_rmse=(
                "rmse",
                "mean"
            ),
            std_rmse=(
                "rmse",
                "std"
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
            n_peak_evaluable=(
                "r_peak_recall",
                finite_count
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
        / f"diffusion_test_summary{suffix}.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(
        "\nDiffusion results:\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nRaw results: "
        f"{raw_path}"
    )

    print(
        f"Summary: "
        f"{summary_path}"
    )

    # Only create the final comparison on the complete benchmark.
    if args.max_rows is None:
        baseline_path = (
            results_dir
            / "fixed_test_summary.csv"
        )

        if baseline_path.exists():
            baseline = pd.read_csv(
                baseline_path
            )

            comparison = pd.concat(
                [
                    baseline,
                    summary
                ],
                ignore_index=True,
                sort=False
            )

            comparison_path = (
                results_dir
                / "final_model_comparison.csv"
            )

            comparison.to_csv(
                comparison_path,
                index=False
            )

            print(
                "Combined comparison:",
                comparison_path
            )


if __name__ == "__main__":
    main()
