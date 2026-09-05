from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.data import load_ptbxl_record, get_lead
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


FACTOR_NAMES = [
    "gap_seconds",
    "gap_position",
    "noise_std_mv",
    "baseline_wander_mv",
]


def empirical_crps(samples, truth):
    """
    Empirical CRPS for a finite ensemble.

    samples: (K, N)
    truth:   (N,)
    """
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    term1 = np.mean(
        np.abs(samples - truth[None, :])
    )

    pairwise = np.abs(
        samples[:, None, :]
        - samples[None, :, :]
    )

    term2 = 0.5 * np.mean(pairwise)

    return float(term1 - term2)


def make_gap_mask(
    length,
    fs,
    gap_seconds,
    gap_position,
    margin_seconds=1.0,
):
    """
    Convert continuous gap length and relative gap position into a
    reproducible contiguous missing block.

    gap_position:
        0.0 = earliest allowed location
        1.0 = latest allowed location
    """
    gap_samples = int(
        round(float(gap_seconds) * fs)
    )

    margin_samples = int(
        round(float(margin_seconds) * fs)
    )

    min_start = margin_samples

    max_start = (
        length
        - margin_samples
        - gap_samples
    )

    if max_start < min_start:
        raise ValueError(
            "Gap plus margins do not fit inside the ECG."
        )

    gap_position = float(
        np.clip(
            gap_position,
            0.0,
            1.0
        )
    )

    start_sample = int(
        round(
            min_start
            + gap_position
            * (max_start - min_start)
        )
    )

    end_sample = (
        start_sample
        + gap_samples
    )

    mask = np.ones(
        length,
        dtype=bool
    )

    mask[
        start_sample:end_sample
    ] = False

    return (
        mask,
        start_sample,
        end_sample,
    )


def create_lhs_design(
    n_points,
    seed,
    gap_min,
    gap_max,
    noise_max,
    baseline_max,
):
    """
    Latin-hypercube design over four independent uniform factors.

    Both physical factor values and standardised [-1, 1] values are
    returned. The latter are used directly by the Legendre PCE.
    """
    sampler = qmc.LatinHypercube(
        d=4,
        seed=seed
    )

    unit = sampler.random(
        n=int(n_points)
    )

    z = (
        2.0 * unit
        - 1.0
    )

    physical = np.empty_like(
        unit
    )

    physical[:, 0] = (
        gap_min
        + unit[:, 0]
        * (gap_max - gap_min)
    )

    physical[:, 1] = unit[:, 1]

    physical[:, 2] = (
        unit[:, 2]
        * noise_max
    )

    physical[:, 3] = (
        unit[:, 3]
        * baseline_max
    )

    design = pd.DataFrame({
        "design_id":
            np.arange(
                n_points,
                dtype=int
            ),

        "gap_seconds":
            physical[:, 0],

        "gap_position":
            physical[:, 1],

        "noise_std_mv":
            physical[:, 2],

        "baseline_wander_mv":
            physical[:, 3],

        "z_gap_seconds":
            z[:, 0],

        "z_gap_position":
            z[:, 1],

        "z_noise_std_mv":
            z[:, 2],

        "z_baseline_wander_mv":
            z[:, 3],
    })

    return design


def select_test_records(
    dataset_dir,
    n_records,
    seed,
):
    """
    Select a fixed held-out record panel from PTB-XL fold 10.

    This does not retrain or tune the diffusion model; it is only used
    as a common evaluation panel for each UQ design point.
    """
    metadata_path = (
        dataset_dir
        / "ptbxl_database.csv"
    )

    metadata = pd.read_csv(
        metadata_path
    )

    test = (
        metadata[
            metadata["strat_fold"]
            == 10
        ]
        .reset_index(drop=True)
    )

    if n_records > len(test):
        n_records = len(test)

    selected = test.sample(
        n=int(n_records),
        replace=False,
        random_state=int(seed),
    )

    selected = (
        selected[
            [
                "filename_lr",
                "strat_fold",
            ]
        ]
        .reset_index(drop=True)
    )

    return selected


def load_record_panel(
    dataset_dir,
    record_table,
    lead,
):
    """
    Load selected ECGs into memory.

    PTB-XL records100 traces are short, so a small evaluation panel is
    inexpensive to keep in memory.
    """
    signals = []
    names = []
    fs_value = None

    for _, row in (
        record_table.iterrows()
    ):
        record_path = (
            dataset_dir
            / str(
                row["filename_lr"]
            )
        )

        ecg, metadata = (
            load_ptbxl_record(
                record_path
            )
        )

        signal = get_lead(
            ecg,
            metadata,
            lead
        ).astype(np.float32)

        fs = int(metadata["fs"])

        if fs_value is None:
            fs_value = fs
        elif fs != fs_value:
            raise ValueError(
                "Selected records do not have a common sampling rate."
            )

        signals.append(signal)

        names.append(
            Path(
                str(
                    row["filename_lr"]
                )
            ).name
        )

    lengths = {
        len(signal)
        for signal in signals
    }

    if len(lengths) != 1:
        raise ValueError(
            "Selected records do not have a common signal length."
        )

    return (
        np.stack(
            signals,
            axis=0
        ),
        names,
        fs_value,
    )


def build_common_noise(
    n_records,
    signal_length,
    seed,
):
    """
    Common-random-number measurement-noise templates.

    The same standard-normal shape is reused at every design point and
    scaled by noise_std_mv. This reduces nuisance Monte Carlo variance
    in the sensitivity experiment.
    """
    rng = np.random.default_rng(
        seed
    )

    return rng.standard_normal(
        size=(
            n_records,
            signal_length
        )
    ).astype(np.float32)


def evaluate_design_point(
    model,
    schedule,
    clean_signals_mv,
    fs,
    mean_mv,
    std_mv,
    gap_seconds,
    gap_position,
    noise_std_mv,
    baseline_wander_mv,
    measurement_noise_template,
    baseline_frequency_hz,
    diffusion_samples,
    diffusion_seed,
    device,
):
    """
    Evaluate one UQ design point across the fixed record panel.

    The scalar outputs are averaged across records and become the
    expensive model outputs that the PCE surrogate approximates.
    """
    n_records, signal_length = (
        clean_signals_mv.shape
    )

    (
        mask_1d,
        start_sample,
        end_sample,
    ) = make_gap_mask(
        length=signal_length,
        fs=fs,
        gap_seconds=gap_seconds,
        gap_position=gap_position,
    )

    # Fixed low-frequency baseline-wander shape.
    t = (
        np.arange(
            signal_length,
            dtype=np.float32
        )
        / float(fs)
    )

    wander_shape = np.sin(
        2.0
        * math.pi
        * float(
            baseline_frequency_hz
        )
        * t
    ).astype(np.float32)

    # Perturb only what the model can observe. The ground truth remains
    # the original clean ECG.
    perturbed_mv = (
        clean_signals_mv
        + float(noise_std_mv)
        * measurement_noise_template
        + float(baseline_wander_mv)
        * wander_shape[None, :]
    )

    observed_norm = (
        perturbed_mv
        - float(mean_mv)
    ) / float(std_mv)

    observed_norm = observed_norm.astype(
        np.float32
    )

    observed_norm[
        :,
        ~mask_1d
    ] = 0.0

    observed_tensor = torch.from_numpy(
        observed_norm[:, None, :]
    ).to(device)

    mask_tensor = torch.from_numpy(
        np.broadcast_to(
            mask_1d[
                None,
                None,
                :
            ],
            (
                n_records,
                1,
                signal_length
            )
        ).astype(np.float32)
    ).to(device)

    generator = torch.Generator(
        device=device
    )

    # Resetting the same seed at every design point gives common random
    # numbers across the UQ design. This makes G(X) less noisy and is
    # useful when fitting the PCE surrogate.
    generator.manual_seed(
        int(diffusion_seed)
    )

    generated = []

    for _ in range(
        int(diffusion_samples)
    ):
        sample_norm = (
            sample_conditional_ddpm(
                model=model,
                observed=observed_tensor,
                mask=mask_tensor,
                schedule=schedule,
                generator=generator,
            )
        )

        sample_mv = (
            sample_norm
            * float(std_mv)
            + float(mean_mv)
        )

        generated.append(
            sample_mv[
                :,
                0,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

    # K x R x L
    generated = np.stack(
        generated,
        axis=0
    )

    predictive_mean = (
        generated.mean(
            axis=0
        )
    )

    missing = (
        ~mask_1d
    )

    per_record_rmse = []
    per_record_crps = []
    per_record_picp90 = []
    per_record_mpiw90 = []
    per_record_sample_std = []
    per_record_correlation = []

    for record_index in range(
        n_records
    ):
        truth = clean_signals_mv[
            record_index,
            missing
        ]

        samples = generated[
            :,
            record_index,
            missing
        ]

        mean_prediction = (
            predictive_mean[
                record_index,
                missing
            ]
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        mean_prediction
                        - truth
                    ) ** 2
                )
            )
        )

        if (
            np.std(truth) > 0
            and np.std(
                mean_prediction
            ) > 0
        ):
            correlation = float(
                np.corrcoef(
                    truth,
                    mean_prediction
                )[0, 1]
            )
        else:
            correlation = np.nan

        lower = np.percentile(
            samples,
            5.0,
            axis=0
        )

        upper = np.percentile(
            samples,
            95.0,
            axis=0
        )

        picp90 = float(
            np.mean(
                (
                    truth >= lower
                )
                & (
                    truth <= upper
                )
            )
        )

        mpiw90 = float(
            np.mean(
                upper - lower
            )
        )

        sample_std = float(
            np.mean(
                np.std(
                    samples,
                    axis=0,
                    ddof=1
                )
            )
        )

        crps = empirical_crps(
            samples,
            truth
        )

        per_record_rmse.append(
            rmse
        )

        per_record_crps.append(
            crps
        )

        per_record_picp90.append(
            picp90
        )

        per_record_mpiw90.append(
            mpiw90
        )

        per_record_sample_std.append(
            sample_std
        )

        per_record_correlation.append(
            correlation
        )

    return {
        "mean_rmse_mv":
            float(
                np.mean(
                    per_record_rmse
                )
            ),

        "mean_crps_mv":
            float(
                np.mean(
                    per_record_crps
                )
            ),

        "mean_picp90":
            float(
                np.mean(
                    per_record_picp90
                )
            ),

        "mean_mpiw90_mv":
            float(
                np.mean(
                    per_record_mpiw90
                )
            ),

        "mean_sample_std_mv":
            float(
                np.mean(
                    per_record_sample_std
                )
            ),

        "mean_correlation":
            float(
                np.nanmean(
                    per_record_correlation
                )
            ),

        "start_sample":
            int(
                start_sample
            ),

        "end_sample":
            int(
                end_sample
            ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate expensive diffusion-model evaluations for "
            "PCE/Sobol uncertainty quantification."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--design-points",
        type=int,
        default=128
    )

    parser.add_argument(
        "--records",
        type=int,
        default=8
    )

    parser.add_argument(
        "--diffusion-samples",
        type=int,
        default=5
    )

    parser.add_argument(
        "--design-seed",
        type=int,
        default=314159
    )

    parser.add_argument(
        "--record-seed",
        type=int,
        default=271828
    )

    parser.add_argument(
        "--measurement-noise-seed",
        type=int,
        default=161803
    )

    parser.add_argument(
        "--diffusion-seed",
        type=int,
        default=20260902
    )

    parser.add_argument(
        "--gap-min",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--gap-max",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--noise-max-mv",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--baseline-max-mv",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--baseline-frequency-hz",
        type=float,
        default=0.33
    )

    args = parser.parse_args()

    if args.design_points < 16:
        raise ValueError(
            "--design-points should be at least 16."
        )

    if args.records < 1:
        raise ValueError(
            "--records must be >= 1."
        )

    if args.diffusion_samples < 2:
        raise ValueError(
            "--diffusion-samples must be >= 2."
        )

    if not (
        0 < args.gap_min
        < args.gap_max
    ):
        raise ValueError(
            "Require 0 < gap-min < gap-max."
        )

    results_dir = (
        PROJECT_ROOT
        / "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    checkpoint_path = (
        results_dir
        / "diffusion_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "results/diffusion_best.pt not found."
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

    mean_mv = float(
        checkpoint["mean_mv"]
    )

    std_mv = float(
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

    record_table = select_test_records(
        dataset_dir=args.dataset_dir,
        n_records=args.records,
        seed=args.record_seed,
    )

    record_path = (
        results_dir
        / "uq_record_manifest.csv"
    )

    record_table.to_csv(
        record_path,
        index=False
    )

    (
        clean_signals_mv,
        record_names,
        fs,
    ) = load_record_panel(
        dataset_dir=args.dataset_dir,
        record_table=record_table,
        lead="II",
    )

    measurement_noise_template = (
        build_common_noise(
            n_records=len(
                clean_signals_mv
            ),
            signal_length=clean_signals_mv.shape[
                1
            ],
            seed=args.measurement_noise_seed,
        )
    )

    design = create_lhs_design(
        n_points=args.design_points,
        seed=args.design_seed,
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        noise_max=args.noise_max_mv,
        baseline_max=args.baseline_max_mv,
    )

    config = {
        "factor_names":
            FACTOR_NAMES,

        "factor_distributions":
            "independent_uniform",

        "gap_seconds":
            [
                args.gap_min,
                args.gap_max
            ],

        "gap_position":
            [
                0.0,
                1.0
            ],

        "noise_std_mv":
            [
                0.0,
                args.noise_max_mv
            ],

        "baseline_wander_mv":
            [
                0.0,
                args.baseline_max_mv
            ],

        "baseline_frequency_hz":
            args.baseline_frequency_hz,

        "design_points":
            args.design_points,

        "records":
            args.records,

        "diffusion_samples":
            args.diffusion_samples,

        "design_seed":
            args.design_seed,

        "record_seed":
            args.record_seed,

        "measurement_noise_seed":
            args.measurement_noise_seed,

        "diffusion_seed":
            args.diffusion_seed,

        "ptbxl_fold":
            10,

        "lead":
            "II",

        "diffusion_steps":
            diffusion_steps,
    }

    config_path = (
        results_dir
        / "uq_config.json"
    )

    with config_path.open(
        "w"
    ) as f:
        json.dump(
            config,
            f,
            indent=2
        )

    print("Device:", device)
    print(
        "Design points:",
        args.design_points
    )
    print(
        "Held-out records:",
        len(record_names)
    )
    print(
        "Diffusion samples per record/design:",
        args.diffusion_samples
    )
    print(
        "Diffusion steps:",
        diffusion_steps
    )
    print(
        "Factors:",
        ", ".join(
            FACTOR_NAMES
        )
    )

    output_rows = []

    for design_index, row in (
        design.iterrows()
    ):
        metrics = evaluate_design_point(
            model=model,
            schedule=schedule,
            clean_signals_mv=clean_signals_mv,
            fs=fs,
            mean_mv=mean_mv,
            std_mv=std_mv,
            gap_seconds=float(
                row["gap_seconds"]
            ),
            gap_position=float(
                row["gap_position"]
            ),
            noise_std_mv=float(
                row["noise_std_mv"]
            ),
            baseline_wander_mv=float(
                row["baseline_wander_mv"]
            ),
            measurement_noise_template=measurement_noise_template,
            baseline_frequency_hz=args.baseline_frequency_hz,
            diffusion_samples=args.diffusion_samples,
            diffusion_seed=args.diffusion_seed,
            device=device,
        )

        output_rows.append({
            **row.to_dict(),
            **metrics,
        })

        print(
            f"Design "
            f"{design_index + 1}"
            f"/{len(design)} "
            f"| gap="
            f"{row['gap_seconds']:.3f}s "
            f"| noise="
            f"{row['noise_std_mv']:.4f}mV "
            f"| baseline="
            f"{row['baseline_wander_mv']:.4f}mV "
            f"| RMSE="
            f"{metrics['mean_rmse_mv']:.5f} "
            f"| CRPS="
            f"{metrics['mean_crps_mv']:.5f} "
            f"| PICP90="
            f"{metrics['mean_picp90']:.3f}"
        )

    results = pd.DataFrame(
        output_rows
    )

    output_path = (
        results_dir
        / "uq_design_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(
        "UQ design results:",
        output_path
    )
    print(
        "UQ configuration:",
        config_path
    )
    print(
        "UQ record panel:",
        record_path
    )


if __name__ == "__main__":
    main()
