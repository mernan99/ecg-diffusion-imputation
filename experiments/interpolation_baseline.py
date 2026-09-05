from pathlib import Path
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------
# Allow imports from project/src
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from src.data import (
    find_records,
    load_ptbxl_record,
    get_lead
)

from src.masking import (
    make_block_mask,
    apply_mask
)

from src.interpolation import (
    linear_interpolate,
    cubic_spline_interpolate
)

from src.metrics import (
    evaluate_reconstruction
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

RESULTS_DIR = PROJECT_ROOT / "results"

LEAD = "II"

GAP_SECONDS = [
    0.25,
    0.5,
    1.0,
    2.0
]

TRIALS_PER_RECORD = 5

SEED = 42


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find every low-resolution ECG
    records = find_records(DATA_DIR)

    print(
        f"Found {len(records)} ECG records."
    )

    if len(records) == 0:
        raise RuntimeError(
            f"No ECG records found under {DATA_DIR}"
        )

    rng = np.random.default_rng(SEED)

    rows = []


    # --------------------------------------------------
    # Loop through ECGs
    # --------------------------------------------------

    for record_number, record_path in enumerate(records):

        try:

            ecg, metadata = load_ptbxl_record(
                record_path
            )

            signal = get_lead(
                ecg,
                metadata,
                LEAD
            )

            fs = metadata["fs"]

        except Exception as error:

            print(
                f"\nCould not load {record_path}: "
                f"{error}"
            )

            continue


        # --------------------------------------------------
        # Different missing-gap lengths
        # --------------------------------------------------

        for gap_seconds in GAP_SECONDS:

            gap_samples = int(
                round(gap_seconds * fs)
            )

            for trial in range(
                TRIALS_PER_RECORD
            ):

                mask = make_block_mask(
                    length=len(signal),
                    gap_samples=gap_samples,
                    rng=rng,
                    margin_samples=fs
                )

                corrupted = apply_mask(
                    signal,
                    mask,
                    fill_value=0.0
                )


                # ------------------------------------------
                # Linear interpolation
                # ------------------------------------------

                linear = linear_interpolate(
                    corrupted,
                    mask
                )

                linear_metrics = (
                    evaluate_reconstruction(
                        signal,
                        linear,
                        mask
                    )
                )

                rows.append({
                    "record":
                        record_path.name,

                    "lead":
                        LEAD,

                    "gap_seconds":
                        gap_seconds,

                    "trial":
                        trial,

                    "method":
                        "linear",

                    "mae":
                        linear_metrics["mae"],

                    "rmse":
                        linear_metrics["rmse"],

                    "correlation":
                        linear_metrics[
                            "correlation"
                        ],
                })


                # ------------------------------------------
                # Cubic spline interpolation
                # ------------------------------------------

                try:

                    cubic = (
                        cubic_spline_interpolate(
                            corrupted,
                            mask
                        )
                    )

                    cubic_metrics = (
                        evaluate_reconstruction(
                            signal,
                            cubic,
                            mask
                        )
                    )

                    rows.append({
                        "record":
                            record_path.name,

                        "lead":
                            LEAD,

                        "gap_seconds":
                            gap_seconds,

                        "trial":
                            trial,

                        "method":
                            "cubic",

                        "mae":
                            cubic_metrics["mae"],

                        "rmse":
                            cubic_metrics["rmse"],

                        "correlation":
                            cubic_metrics[
                                "correlation"
                            ],
                    })

                except Exception as error:

                    print(
                        f"\nCubic interpolation failed for "
                        f"{record_path.name}: {error}"
                    )


        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            (record_number + 1) % 100 == 0
            or record_number == 0
        ):

            print(
                f"Processed "
                f"{record_number + 1}"
                f"/{len(records)} records"
            )


    # --------------------------------------------------
    # Save every individual result
    # --------------------------------------------------

    results = pd.DataFrame(rows)

    raw_output = (
        RESULTS_DIR
        / "interpolation_results.csv"
    )

    results.to_csv(
        raw_output,
        index=False
    )


    # --------------------------------------------------
    # Calculate summary statistics
    # --------------------------------------------------

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

            n=(
                "rmse",
                "count"
            )
        )
        .reset_index()
    )


    summary_output = (
        RESULTS_DIR
        / "interpolation_summary.csv"
    )

    summary.to_csv(
        summary_output,
        index=False
    )


    print("\nFinished.\n")

    print(summary.to_string(index=False))

    print(
        "\nIndividual results saved to:"
    )

    print(raw_output)

    print(
        "\nSummary saved to:"
    )

    print(summary_output)


if __name__ == "__main__":
    main()
