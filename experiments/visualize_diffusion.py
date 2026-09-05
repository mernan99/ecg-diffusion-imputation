from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.data import (
    load_ptbxl_record,
    get_lead,
)
from src.mask_manifest import (
    load_mask_manifest,
    mask_from_bounds,
)
from src.masking import apply_mask
from src.unet_autoencoder import (
    UNetAutoencoder1D,
    combine_observed_and_prediction,
)
from src.diffusion import (
    ConditionalDiffusionUNet1D,
    DiffusionSchedule,
    sample_conditional_ddpm,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def reconstruct_unet(
    signal_mv,
    mask,
    mean,
    std,
    checkpoint,
    device,
):
    model = (
        UNetAutoencoder1D()
        .to(device)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    normalized = (
        signal_mv - mean
    ) / std

    observed = apply_mask(
        normalized,
        mask,
        fill_value=0.0
    ).astype(np.float32)

    mask_float = (
        mask.astype(np.float32)
    )

    model_input = np.stack(
        [
            observed,
            mask_float
        ],
        axis=0
    )[None, ...]

    with torch.no_grad():
        prediction = model(
            torch.from_numpy(
                model_input
            ).to(device)
        )

        observed_tensor = (
            torch.from_numpy(
                observed
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

        combined = (
            combine_observed_and_prediction(
                observed_tensor,
                prediction,
                mask_tensor,
            )
        )

    return (
        combined
        .squeeze()
        .cpu()
        .numpy()
        * std
        + mean
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help=(
            "Index among unique records in the fixed manifest."
        )
    )

    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--trial",
        type=int,
        default=0
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help=(
            "Number of stochastic diffusion reconstructions."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    args = parser.parse_args()

    results_dir = (
        PROJECT_ROOT
        / "results"
    )

    manifest_path = (
        results_dir
        / "fold10_mask_manifest.csv"
    )

    diffusion_path = (
        results_dir
        / "diffusion_best.pt"
    )

    unet_path = (
        results_dir
        / "unet_autoencoder_best.pt"
    )

    manifest = load_mask_manifest(
        manifest_path
    )

    records = (
        manifest["record"]
        .drop_duplicates()
        .tolist()
    )

    if not (
        0 <= args.record_index
        < len(records)
    ):
        raise IndexError(
            f"record-index must be between "
            f"0 and {len(records) - 1}"
        )

    selected_record = records[
        args.record_index
    ]

    candidates = manifest[
        (
            manifest["record"]
            == selected_record
        )
        & (
            np.isclose(
                manifest["gap_seconds"],
                args.gap_seconds
            )
        )
        & (
            manifest["trial"]
            == args.trial
        )
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not identify exactly one manifest row "
            "for the requested record/gap/trial."
        )

    row = candidates.iloc[0]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    diffusion_checkpoint = (
        torch.load(
            diffusion_path,
            map_location=device
        )
    )

    unet_checkpoint = torch.load(
        unet_path,
        map_location=device
    )

    mean = float(
        diffusion_checkpoint[
            "mean_mv"
        ]
    )

    std = float(
        diffusion_checkpoint[
            "std_mv"
        ]
    )

    record_path = (
        args.dataset_dir
        / str(
            row["filename_lr"]
        )
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

    fs = int(
        metadata["fs"]
    )

    mask = mask_from_bounds(
        length=len(signal_mv),
        start_sample=int(
            row["start_sample"]
        ),
        end_sample=int(
            row["end_sample"]
        ),
    )

    unet_mv = reconstruct_unet(
        signal_mv=signal_mv,
        mask=mask,
        mean=mean,
        std=std,
        checkpoint=unet_checkpoint,
        device=device,
    )

    normalized = (
        signal_mv - mean
    ) / std

    observed_norm = apply_mask(
        normalized,
        mask,
        fill_value=0.0
    ).astype(np.float32)

    observed_tensor = (
        torch.from_numpy(
            observed_norm
        )
        .view(1, 1, -1)
        .to(device)
    )

    mask_tensor = (
        torch.from_numpy(
            mask.astype(
                np.float32
            )
        )
        .view(1, 1, -1)
        .to(device)
    )

    diffusion_model = (
        ConditionalDiffusionUNet1D()
        .to(device)
    )

    diffusion_model.load_state_dict(
        diffusion_checkpoint[
            "ema_model_state_dict"
        ]
    )

    diffusion_model.eval()

    schedule = DiffusionSchedule(
        num_steps=int(
            diffusion_checkpoint[
                "diffusion_steps"
            ]
        ),
        device=device
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        args.seed
    )

    diffusion_samples = []

    for sample_number in range(
        args.samples
    ):
        with torch.no_grad():
            sample_norm = (
                sample_conditional_ddpm(
                    model=diffusion_model,
                    observed=observed_tensor,
                    mask=mask_tensor,
                    schedule=schedule,
                    generator=generator,
                )
            )

        sample_mv = (
            sample_norm
            .squeeze()
            .cpu()
            .numpy()
            * std
            + mean
        )

        diffusion_samples.append(
            sample_mv
        )

        print(
            f"Generated diffusion sample "
            f"{sample_number + 1}"
            f"/{args.samples}"
        )

    diffusion_samples = np.stack(
        diffusion_samples,
        axis=0
    )

    diffusion_mean = (
        diffusion_samples.mean(
            axis=0
        )
    )

    lower = np.percentile(
        diffusion_samples,
        5,
        axis=0
    )

    upper = np.percentile(
        diffusion_samples,
        95,
        axis=0
    )

    observed_plot = (
        signal_mv.copy()
    )

    observed_plot[
        ~mask
    ] = np.nan

    t = (
        np.arange(
            len(signal_mv)
        )
        / fs
    )

    missing = np.flatnonzero(
        ~mask
    )

    start = int(
        missing[0]
    )

    end = int(
        missing[-1]
    )

    context = int(
        round(
            0.5
            * fs
        )
    )

    zoom_start = max(
        0,
        start - context
    )

    zoom_end = min(
        len(signal_mv),
        end + context + 1
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9)
    )

    ax = axes[0]

    ax.plot(
        t,
        signal_mv,
        label="Ground truth",
        linewidth=2
    )

    ax.plot(
        t,
        observed_plot,
        label="Observed signal",
        linewidth=1.2
    )

    ax.plot(
        t,
        unet_mv,
        label="U-Net AE",
        linewidth=1.4
    )

    ax.plot(
        t,
        diffusion_mean,
        label="Diffusion mean",
        linewidth=1.6
    )

    ax.fill_between(
        t,
        lower,
        upper,
        alpha=0.18,
        label="Diffusion 5–95% interval"
    )

    ax.axvspan(
        t[start],
        t[end],
        alpha=0.12,
        label="Missing region"
    )

    ax.set_title(
        f"{selected_record} — Lead II — full signal"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Amplitude (mV)"
    )

    ax.legend()

    ax = axes[1]

    sl = slice(
        zoom_start,
        zoom_end
    )

    ax.plot(
        t[sl],
        signal_mv[sl],
        label="Ground truth",
        linewidth=2
    )

    ax.plot(
        t[sl],
        observed_plot[sl],
        label="Observed signal",
        linewidth=1.2
    )

    ax.plot(
        t[sl],
        unet_mv[sl],
        label="U-Net AE",
        linewidth=1.5
    )

    ax.plot(
        t[sl],
        diffusion_mean[sl],
        label="Diffusion mean",
        linewidth=1.7
    )

    ax.fill_between(
        t[sl],
        lower[sl],
        upper[sl],
        alpha=0.22,
        label="Diffusion 5–95% interval"
    )

    ax.axvspan(
        t[start],
        t[end],
        alpha=0.12,
        label="Missing region"
    )

    ax.set_title(
        f"{selected_record} — "
        f"{args.gap_seconds:g}s fixed gap — zoom"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Amplitude (mV)"
    )

    ax.legend()

    plt.tight_layout()

    output_path = (
        results_dir
        / "diffusion_reconstruction_visualization.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    print(
        "\nDevice:",
        device
    )

    print(
        "Record:",
        selected_record
    )

    print(
        "Gap:",
        args.gap_seconds
    )

    print(
        "Trial:",
        args.trial
    )

    print(
        "Missing samples:",
        start,
        "to",
        end
    )

    print(
        "Saved:",
        output_path
    )

    plt.show()


if __name__ == "__main__":
    main()
