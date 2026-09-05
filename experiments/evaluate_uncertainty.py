from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion import (
    ConditionalDiffusionUNet1D,
    DiffusionSchedule,
    sample_conditional_ddpm,
)
from src.diffusion_dataset import PTBXLManifestDataset
from src.metrics import evaluate_reconstruction


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

INTERVALS = {
    50: (25.0, 75.0),
    80: (10.0, 90.0),
    90: (5.0, 95.0),
}


def make_stratified_subset(manifest, cases_per_gap, seed):
    selected = []

    for gap_seconds in sorted(manifest["gap_seconds"].unique()):
        group = manifest[
            np.isclose(manifest["gap_seconds"], gap_seconds)
        ]

        n = min(int(cases_per_gap), len(group))

        sampled = group.sample(
            n=n,
            replace=False,
            random_state=int(seed),
        )

        selected.append(sampled)

    subset = pd.concat(selected, ignore_index=True)

    return (
        subset
        .sort_values(
            ["gap_seconds", "record", "trial", "start_sample"]
        )
        .reset_index(drop=True)
    )


def empirical_crps(samples, truth):
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    term1 = np.mean(
        np.abs(samples - truth[None, :])
    )

    pairwise = np.abs(
        samples[:, None, :] - samples[None, :, :]
    )

    term2 = 0.5 * np.mean(pairwise)

    return float(term1 - term2)


def interval_metrics(
    samples,
    truth,
    lower_percentile,
    upper_percentile,
):
    lower = np.percentile(
        samples,
        lower_percentile,
        axis=0
    )

    upper = np.percentile(
        samples,
        upper_percentile,
        axis=0
    )

    coverage = float(
        np.mean(
            (truth >= lower)
            & (truth <= upper)
        )
    )

    width = float(
        np.mean(upper - lower)
    )

    return coverage, width


def build_calibration_table(results):
    rows = []

    gap_values = sorted(
        results["gap_seconds"].unique()
    )

    for nominal in INTERVALS:
        column = f"picp_{nominal}"

        for gap in gap_values:
            group = results[
                np.isclose(
                    results["gap_seconds"],
                    gap
                )
            ]

            observed = float(
                group[column].mean()
            )

            rows.append({
                "scope": f"{gap:g}s",
                "gap_seconds": float(gap),
                "nominal_coverage": nominal / 100.0,
                "observed_coverage": observed,
                "calibration_error": abs(
                    observed - nominal / 100.0
                ),
            })

        overall = float(
            results[column].mean()
        )

        rows.append({
            "scope": "overall",
            "gap_seconds": np.nan,
            "nominal_coverage": nominal / 100.0,
            "observed_coverage": overall,
            "calibration_error": abs(
                overall - nominal / 100.0
            ),
        })

    return pd.DataFrame(rows)


def plot_calibration(calibration, output_path):
    plt.figure(figsize=(8, 6))

    x = np.array([0.5, 0.8, 0.9])

    plt.plot(
        x,
        x,
        linestyle="--",
        label="Ideal calibration"
    )

    scopes = [
        scope
        for scope in calibration["scope"].unique()
        if scope != "overall"
    ]

    for scope in scopes:
        group = (
            calibration[
                calibration["scope"] == scope
            ]
            .sort_values("nominal_coverage")
        )

        plt.plot(
            group["nominal_coverage"],
            group["observed_coverage"],
            marker="o",
            label=scope,
        )

    overall = (
        calibration[
            calibration["scope"] == "overall"
        ]
        .sort_values("nominal_coverage")
    )

    plt.plot(
        overall["nominal_coverage"],
        overall["observed_coverage"],
        marker="s",
        linewidth=2,
        label="Overall",
    )

    plt.xlabel("Nominal interval coverage")
    plt.ylabel("Observed coverage")
    plt.title("Diffusion uncertainty calibration")
    plt.xlim(0.45, 0.95)
    plt.ylim(0.45, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_interval_width(summary, output_path):
    ordered = summary.sort_values("gap_seconds")

    plt.figure(figsize=(8, 6))

    plt.plot(
        ordered["gap_seconds"],
        ordered["mean_mpiw_90"],
        marker="o"
    )

    plt.xlabel("Missing gap length (s)")
    plt.ylabel(
        "Mean 90% prediction interval width (mV)"
    )
    plt.title(
        "Diffusion uncertainty versus missing-gap length"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--cases-per-gap",
        type=int,
        default=500
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--subset-seed",
        type=int,
        default=2026
    )

    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=987654
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    args = parser.parse_args()

    if args.samples < 2:
        raise ValueError(
            "--samples must be at least 2."
        )

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    checkpoint_path = (
        results_dir / "diffusion_best.pt"
    )

    manifest_path = (
        results_dir / "fold10_mask_manifest.csv"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "results/diffusion_best.pt not found."
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "results/fold10_mask_manifest.csv not found."
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
    diffusion_steps = int(
        checkpoint["diffusion_steps"]
    )

    full_manifest = pd.read_csv(
        manifest_path
    )

    subset_manifest = make_stratified_subset(
        manifest=full_manifest,
        cases_per_gap=args.cases_per_gap,
        seed=args.subset_seed,
    )

    subset_path = (
        results_dir
        / "uncertainty_subset_manifest.csv"
    )

    subset_manifest.to_csv(
        subset_path,
        index=False
    )

    dataset = PTBXLManifestDataset(
        dataset_dir=args.dataset_dir,
        manifest_path=subset_path,
        mean=mean,
        std=std,
        lead="II",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = (
        ConditionalDiffusionUNet1D()
        .to(device)
    )

    model.load_state_dict(
        checkpoint["ema_model_state_dict"]
    )
    model.eval()

    schedule = DiffusionSchedule(
        num_steps=diffusion_steps,
        device=device
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.sampling_seed)

    print("Device:", device)
    print("Diffusion steps:", diffusion_steps)
    print("Cases per gap:", args.cases_per_gap)
    print("Total cases:", len(dataset))
    print("Samples per case:", args.samples)

    rows = []
    processed = 0

    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)

        sample_list = []

        for _ in range(args.samples):
            sample_norm = sample_conditional_ddpm(
                model=model,
                observed=observed,
                mask=mask,
                schedule=schedule,
                generator=generator,
            )

            sample_mv = (
                sample_norm * std + mean
            )

            sample_list.append(
                sample_mv
                .detach()
                .cpu()
                .numpy()
            )

        samples_np = np.stack(
            sample_list,
            axis=0
        )

        signal_np = (
            batch["signal_mv"]
            .cpu()
            .numpy()
        )

        mask_np = (
            batch["mask"]
            .cpu()
            .numpy()
            .astype(bool)
        )

        row_indices = (
            batch["row_index"]
            .cpu()
            .numpy()
        )

        for j, row_index in enumerate(row_indices):
            manifest_row = dataset.manifest.iloc[
                int(row_index)
            ]

            truth = signal_np[j, 0]
            observation_mask = mask_np[j, 0]
            missing = ~observation_mask

            missing_samples = samples_np[
                :, j, 0, missing
            ]

            missing_truth = truth[missing]

            predictive_mean = (
                samples_np[:, j, 0, :]
                .mean(axis=0)
            )

            reconstruction = evaluate_reconstruction(
                truth,
                predictive_mean,
                observation_mask,
            )

            sample_std = np.std(
                missing_samples,
                axis=0,
                ddof=1,
            )

            case = {
                "record": manifest_row["record"],
                "gap_seconds": float(
                    manifest_row["gap_seconds"]
                ),
                "trial": int(
                    manifest_row["trial"]
                ),
                "start_sample": int(
                    manifest_row["start_sample"]
                ),
                "end_sample": int(
                    manifest_row["end_sample"]
                ),
                "n_missing_samples": int(
                    missing.sum()
                ),
                "diffusion_samples": int(
                    args.samples
                ),
                "mean_sample_std_mv": float(
                    np.mean(sample_std)
                ),
                "median_sample_std_mv": float(
                    np.median(sample_std)
                ),
                "max_sample_std_mv": float(
                    np.max(sample_std)
                ),
                "crps_mv": empirical_crps(
                    missing_samples,
                    missing_truth,
                ),
                "mean_reconstruction_mae":
                    reconstruction["mae"],
                "mean_reconstruction_rmse":
                    reconstruction["rmse"],
                "mean_reconstruction_correlation":
                    reconstruction["correlation"],
            }

            for nominal, bounds in INTERVALS.items():
                lower_percentile, upper_percentile = bounds

                coverage, width = interval_metrics(
                    samples=missing_samples,
                    truth=missing_truth,
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile,
                )

                case[f"picp_{nominal}"] = coverage
                case[f"mpiw_{nominal}_mv"] = width

            rows.append(case)

        processed += len(row_indices)

        if (
            processed % 100 < len(row_indices)
            or processed == len(dataset)
        ):
            print(
                f"Processed {processed}/{len(dataset)}"
            )

    results = pd.DataFrame(rows)

    results_path = (
        results_dir / "uncertainty_results.csv"
    )
    results.to_csv(
        results_path,
        index=False
    )

    summary = (
        results
        .groupby("gap_seconds")
        .agg(
            n=("record", "count"),
            mean_crps_mv=("crps_mv", "mean"),
            mean_sample_std_mv=(
                "mean_sample_std_mv",
                "mean"
            ),
            mean_reconstruction_mae=(
                "mean_reconstruction_mae",
                "mean"
            ),
            mean_reconstruction_rmse=(
                "mean_reconstruction_rmse",
                "mean"
            ),
            mean_reconstruction_correlation=(
                "mean_reconstruction_correlation",
                "mean"
            ),
            observed_coverage_50=(
                "picp_50",
                "mean"
            ),
            observed_coverage_80=(
                "picp_80",
                "mean"
            ),
            observed_coverage_90=(
                "picp_90",
                "mean"
            ),
            mean_mpiw_50=(
                "mpiw_50_mv",
                "mean"
            ),
            mean_mpiw_80=(
                "mpiw_80_mv",
                "mean"
            ),
            mean_mpiw_90=(
                "mpiw_90_mv",
                "mean"
            ),
        )
        .reset_index()
    )

    for nominal in INTERVALS:
        summary[
            f"calibration_error_{nominal}"
        ] = np.abs(
            summary[
                f"observed_coverage_{nominal}"
            ]
            - nominal / 100.0
        )

    summary_path = (
        results_dir / "uncertainty_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False
    )

    calibration = build_calibration_table(
        results
    )

    calibration_path = (
        results_dir
        / "uncertainty_calibration.csv"
    )

    calibration.to_csv(
        calibration_path,
        index=False
    )

    coverage_plot = (
        results_dir
        / "uncertainty_coverage_plot.png"
    )

    width_plot = (
        results_dir
        / "uncertainty_width_by_gap.png"
    )

    plot_calibration(
        calibration,
        coverage_plot
    )

    plot_interval_width(
        summary,
        width_plot
    )

    print("\nUncertainty summary:\n")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print("Subset:", subset_path)
    print("Per-case results:", results_path)
    print("Summary:", summary_path)
    print("Calibration:", calibration_path)
    print("Coverage plot:", coverage_plot)
    print("Width plot:", width_plot)


if __name__ == "__main__":
    main()
