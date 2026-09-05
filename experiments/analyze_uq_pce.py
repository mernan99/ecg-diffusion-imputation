from pathlib import Path
import argparse
import itertools
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legval


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


FACTOR_COLUMNS = [
    "z_gap_seconds",
    "z_gap_position",
    "z_noise_std_mv",
    "z_baseline_wander_mv",
]

FACTOR_LABELS = [
    "Gap length",
    "Gap position",
    "Measurement noise",
    "Baseline wander",
]

OUTPUT_COLUMNS = {
    "mean_rmse_mv":
        "RMSE",
    "mean_crps_mv":
        "CRPS",
    "mean_picp90":
        "PICP90",
    "mean_mpiw90_mv":
        "MPIW90",
    "mean_sample_std_mv":
        "Sample uncertainty",
    "mean_correlation":
        "Correlation",
}


def total_degree_indices(
    dimension,
    degree,
):
    """
    Generate all multi-indices alpha with sum(alpha) <= degree.
    """
    indices = []

    for alpha in itertools.product(
        range(
            degree + 1
        ),
        repeat=dimension
    ):
        if sum(alpha) <= degree:
            indices.append(
                tuple(
                    int(v)
                    for v in alpha
                )
            )

    return indices


def orthonormal_legendre_values(
    x,
    max_degree,
):
    """
    Evaluate orthonormal Legendre polynomials for Uniform(-1, 1).

    Standard Legendre polynomials satisfy:
        E[P_n(X)^2] = 1 / (2n + 1)

    for X ~ Uniform(-1,1), therefore:
        phi_n(x) = sqrt(2n + 1) P_n(x)
    """
    x = np.asarray(
        x,
        dtype=np.float64
    )

    values = np.empty(
        (
            len(x),
            max_degree + 1
        ),
        dtype=np.float64
    )

    for degree in range(
        max_degree + 1
    ):
        coefficients = np.zeros(
            degree + 1,
            dtype=np.float64
        )

        coefficients[
            degree
        ] = 1.0

        values[
            :,
            degree
        ] = (
            math.sqrt(
                2 * degree + 1
            )
            * legval(
                x,
                coefficients
            )
        )

    return values


def build_pce_matrix(
    x,
    multi_indices,
):
    """
    Build the multivariate orthonormal Legendre PCE design matrix.
    """
    x = np.asarray(
        x,
        dtype=np.float64
    )

    n_samples, dimension = (
        x.shape
    )

    max_degree = max(
        max(alpha)
        for alpha in multi_indices
    )

    one_dimensional = [
        orthonormal_legendre_values(
            x[:, factor],
            max_degree=max_degree
        )
        for factor in range(
            dimension
        )
    ]

    matrix = np.ones(
        (
            n_samples,
            len(
                multi_indices
            )
        ),
        dtype=np.float64
    )

    for column, alpha in enumerate(
        multi_indices
    ):
        term = np.ones(
            n_samples,
            dtype=np.float64
        )

        for factor in range(
            dimension
        ):
            term *= (
                one_dimensional[
                    factor
                ][
                    :,
                    alpha[factor]
                ]
            )

        matrix[
            :,
            column
        ] = term

    return matrix


def fit_pce(
    x,
    y,
    multi_indices,
):
    matrix = build_pce_matrix(
        x,
        multi_indices
    )

    coefficients, _, _, _ = (
        np.linalg.lstsq(
            matrix,
            y,
            rcond=None
        )
    )

    return coefficients


def predict_pce(
    x,
    coefficients,
    multi_indices,
):
    matrix = build_pce_matrix(
        x,
        multi_indices
    )

    return matrix @ coefficients


def regression_metrics(
    truth,
    prediction,
):
    truth = np.asarray(
        truth,
        dtype=np.float64
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64
    )

    residual = (
        truth - prediction
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                residual
            )
        )
    )

    denominator = np.sum(
        (
            truth
            - np.mean(
                truth
            )
        ) ** 2
    )

    if denominator > 0:
        r2 = float(
            1.0
            - np.sum(
                residual ** 2
            )
            / denominator
        )
    else:
        r2 = np.nan

    return {
        "rmse":
            rmse,
        "mae":
            mae,
        "r2":
            r2,
    }


def cross_validate_pce(
    x,
    y,
    multi_indices,
    folds=5,
    seed=12345,
):
    """
    Deterministic K-fold validation of the PCE approximation.
    """
    n = len(y)

    if n < folds:
        folds = n

    rng = np.random.default_rng(
        seed
    )

    permutation = rng.permutation(
        n
    )

    fold_indices = np.array_split(
        permutation,
        folds
    )

    prediction = np.full(
        n,
        np.nan,
        dtype=np.float64
    )

    all_indices = np.arange(
        n
    )

    for test_indices in fold_indices:
        train_mask = np.ones(
            n,
            dtype=bool
        )

        train_mask[
            test_indices
        ] = False

        train_indices = all_indices[
            train_mask
        ]

        coefficients = fit_pce(
            x[
                train_indices
            ],
            y[
                train_indices
            ],
            multi_indices,
        )

        prediction[
            test_indices
        ] = predict_pce(
            x[
                test_indices
            ],
            coefficients,
            multi_indices,
        )

    return regression_metrics(
        y,
        prediction
    )


def sobol_from_pce(
    coefficients,
    multi_indices,
    dimension,
):
    """
    Compute first-order and total-effect Sobol indices analytically
    from an orthonormal PCE.

    With an orthonormal basis, the output variance is the sum of the
    squared non-constant coefficients.
    """
    coefficients = np.asarray(
        coefficients,
        dtype=np.float64
    )

    variance = 0.0

    for coefficient, alpha in zip(
        coefficients,
        multi_indices
    ):
        if any(
            degree > 0
            for degree in alpha
        ):
            variance += (
                coefficient ** 2
            )

    if variance <= 0:
        return (
            np.full(
                dimension,
                np.nan
            ),
            np.full(
                dimension,
                np.nan
            ),
            0.0,
        )

    first_order = np.zeros(
        dimension,
        dtype=np.float64
    )

    total_effect = np.zeros(
        dimension,
        dtype=np.float64
    )

    for coefficient, alpha in zip(
        coefficients,
        multi_indices
    ):
        if not any(
            degree > 0
            for degree in alpha
        ):
            continue

        contribution = (
            coefficient ** 2
        )

        active = [
            index
            for index, degree in enumerate(
                alpha
            )
            if degree > 0
        ]

        for factor in active:
            total_effect[
                factor
            ] += contribution

        if len(active) == 1:
            first_order[
                active[0]
            ] += contribution

    first_order /= variance
    total_effect /= variance

    return (
        first_order,
        total_effect,
        variance,
    )


def save_sobol_plot(
    sobol_table,
    output_name,
    output_dir,
):
    group = sobol_table[
        sobol_table[
            "output"
        ] == output_name
    ].copy()

    positions = np.arange(
        len(group)
    )

    width = 0.36

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        positions - width / 2,
        group["S1"],
        width=width,
        label="First-order S1",
    )

    plt.bar(
        positions + width / 2,
        group["ST"],
        width=width,
        label="Total-effect ST",
    )

    plt.xticks(
        positions,
        group["factor_label"],
        rotation=20,
        ha="right",
    )

    plt.ylabel(
        "Sobol index"
    )

    plt.ylim(
        bottom=0
    )

    plt.title(
        f"PCE Sobol sensitivity — {output_name}"
    )

    plt.legend()
    plt.tight_layout()

    safe_name = (
        output_name
        .lower()
        .replace(
            " ",
            "_"
        )
    )

    output_path = (
        output_dir
        / f"uq_sobol_{safe_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit Legendre polynomial-chaos surrogates to diffusion UQ "
            "outputs and derive Sobol first-order/total-effect indices."
        )
    )

    parser.add_argument(
        "--degree",
        type=int,
        default=3
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5
    )

    parser.add_argument(
        "--cv-seed",
        type=int,
        default=424242
    )

    args = parser.parse_args()

    if args.degree < 1:
        raise ValueError(
            "--degree must be >= 1."
        )

    results_dir = (
        PROJECT_ROOT
        / "results"
    )

    design_path = (
        results_dir
        / "uq_design_results.csv"
    )

    if not design_path.exists():
        raise FileNotFoundError(
            "results/uq_design_results.csv not found. "
            "Run experiments/run_uq_design.py first."
        )

    data = pd.read_csv(
        design_path
    )

    missing_factor_columns = [
        column
        for column in FACTOR_COLUMNS
        if column not in data.columns
    ]

    if missing_factor_columns:
        raise ValueError(
            "Missing standardised factor columns: "
            f"{missing_factor_columns}"
        )

    available_outputs = [
        column
        for column in OUTPUT_COLUMNS
        if column in data.columns
    ]

    if not available_outputs:
        raise ValueError(
            "No recognised UQ output columns were found."
        )

    x = data[
        FACTOR_COLUMNS
    ].to_numpy(
        dtype=np.float64
    )

    dimension = x.shape[
        1
    ]

    multi_indices = (
        total_degree_indices(
            dimension=dimension,
            degree=args.degree,
        )
    )

    n_terms = len(
        multi_indices
    )

    if len(data) <= n_terms:
        raise ValueError(
            f"Degree-{args.degree} PCE with {dimension} factors "
            f"contains {n_terms} terms, but only {len(data)} design "
            "points are available. Increase --design-points or reduce "
            "--degree."
        )

    print(
        "UQ design points:",
        len(data)
    )

    print(
        "PCE factors:",
        dimension
    )

    print(
        "PCE degree:",
        args.degree
    )

    print(
        "PCE basis terms:",
        n_terms
    )

    validation_rows = []
    sobol_rows = []
    coefficient_rows = []

    for output_column in (
        available_outputs
    ):
        output_label = (
            OUTPUT_COLUMNS[
                output_column
            ]
        )

        y = data[
            output_column
        ].to_numpy(
            dtype=np.float64
        )

        finite = np.isfinite(
            y
        )

        x_valid = x[
            finite
        ]

        y_valid = y[
            finite
        ]

        if len(y_valid) <= n_terms:
            print(
                f"Skipping {output_label}: "
                "not enough finite observations."
            )
            continue

        cv = cross_validate_pce(
            x=x_valid,
            y=y_valid,
            multi_indices=multi_indices,
            folds=args.cv_folds,
            seed=args.cv_seed,
        )

        coefficients = fit_pce(
            x=x_valid,
            y=y_valid,
            multi_indices=multi_indices,
        )

        fitted = predict_pce(
            x=x_valid,
            coefficients=coefficients,
            multi_indices=multi_indices,
        )

        training_metrics = (
            regression_metrics(
                y_valid,
                fitted
            )
        )

        (
            first_order,
            total_effect,
            pce_variance,
        ) = sobol_from_pce(
            coefficients=coefficients,
            multi_indices=multi_indices,
            dimension=dimension,
        )

        pce_mean = float(
            coefficients[
                multi_indices.index(
                    tuple(
                        [0]
                        * dimension
                    )
                )
            ]
        )

        validation_rows.append({
            "output":
                output_label,

            "output_column":
                output_column,

            "n":
                len(
                    y_valid
                ),

            "pce_degree":
                args.degree,

            "basis_terms":
                n_terms,

            "cv_rmse":
                cv["rmse"],

            "cv_mae":
                cv["mae"],

            "cv_r2":
                cv["r2"],

            "fit_rmse":
                training_metrics[
                    "rmse"
                ],

            "fit_r2":
                training_metrics[
                    "r2"
                ],

            "pce_mean":
                pce_mean,

            "pce_variance":
                pce_variance,
        })

        for factor_index, (
            factor_column,
            factor_label,
        ) in enumerate(
            zip(
                FACTOR_COLUMNS,
                FACTOR_LABELS,
            )
        ):
            sobol_rows.append({
                "output":
                    output_label,

                "output_column":
                    output_column,

                "factor":
                    factor_column.replace(
                        "z_",
                        ""
                    ),

                "factor_label":
                    factor_label,

                "S1":
                    float(
                        first_order[
                            factor_index
                        ]
                    ),

                "ST":
                    float(
                        total_effect[
                            factor_index
                        ]
                    ),

                "interaction_share":
                    float(
                        total_effect[
                            factor_index
                        ]
                        - first_order[
                            factor_index
                        ]
                    ),
            })

        for coefficient, alpha in zip(
            coefficients,
            multi_indices
        ):
            row = {
                "output":
                    output_label,

                "coefficient":
                    float(
                        coefficient
                    ),

                "coefficient_squared":
                    float(
                        coefficient
                        ** 2
                    ),
            }

            for factor_index, (
                factor_label
            ) in enumerate(
                FACTOR_LABELS
            ):
                key = (
                    "degree_"
                    + factor_label
                    .lower()
                    .replace(
                        " ",
                        "_"
                    )
                )

                row[
                    key
                ] = int(
                    alpha[
                        factor_index
                    ]
                )

            coefficient_rows.append(
                row
            )

        print(
            f"{output_label}: "
            f"CV R2={cv['r2']:.4f}, "
            f"CV RMSE={cv['rmse']:.6f}"
        )

    validation = pd.DataFrame(
        validation_rows
    )

    sobol = pd.DataFrame(
        sobol_rows
    )

    coefficients = pd.DataFrame(
        coefficient_rows
    )

    validation_path = (
        results_dir
        / "uq_pce_validation.csv"
    )

    sobol_path = (
        results_dir
        / "uq_sobol_indices.csv"
    )

    coefficient_path = (
        results_dir
        / "uq_pce_coefficients.csv"
    )

    validation.to_csv(
        validation_path,
        index=False
    )

    sobol.to_csv(
        sobol_path,
        index=False
    )

    coefficients.to_csv(
        coefficient_path,
        index=False
    )

    plot_paths = []

    if not sobol.empty:
        for output_name in (
            sobol["output"]
            .drop_duplicates()
            .tolist()
        ):
            plot_paths.append(
                save_sobol_plot(
                    sobol_table=sobol,
                    output_name=output_name,
                    output_dir=results_dir,
                )
            )

    print(
        "\nPCE validation:\n"
    )

    if not validation.empty:
        print(
            validation.to_string(
                index=False
            )
        )

    print(
        "\nSobol indices:\n"
    )

    if not sobol.empty:
        print(
            sobol.to_string(
                index=False
            )
        )

    print(
        "\nSaved:"
    )

    print(
        "Validation:",
        validation_path
    )

    print(
        "Sobol indices:",
        sobol_path
    )

    print(
        "PCE coefficients:",
        coefficient_path
    )

    for path in plot_paths:
        print(
            "Plot:",
            path
        )


if __name__ == "__main__":
    main()
