from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.data import find_records, load_ptbxl_record, get_lead
from src.masking import make_block_mask, apply_mask
from src.interpolation import linear_interpolate, cubic_spline_interpolate
from src.metrics import evaluate_reconstruction


def main():
    parser = argparse.ArgumentParser(
        description="Test ECG missing-segment reconstruction on one PTB-XL record."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Folder containing PTB-XL .hea/.dat files. Searched recursively."
    )
    parser.add_argument(
        "--record",
        default=None,
        help=(
            "Optional record path without extension, for example "
            "'data/records100/00000/00001_lr'. "
            "If omitted, the first complete record under --data-dir is used."
        ),
    )
    parser.add_argument(
        "--lead",
        default="II",
        help="ECG lead to reconstruct. Default: II"
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=1.0,
        help="Length of the simulated missing block in seconds. Default: 1.0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to choose the gap position. Default: 42"
    )
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. Choose a record
    # --------------------------------------------------
    if args.record is not None:
        record_path = Path(args.record)
    else:
        records = find_records(args.data_dir)

        if not records:
            raise FileNotFoundError(
                f"No complete '*_lr.hea' + '*_lr.dat' pairs were found under "
                f"'{args.data_dir}'. Put the PTB-XL files in that folder, "
                f"or pass --data-dir with the correct location."
            )

        record_path = records[0]

    print(f"Using record: {record_path}")

    # --------------------------------------------------
    # 2. Load ECG and extract one lead
    # --------------------------------------------------
    ecg, metadata = load_ptbxl_record(record_path)
    lead_signal = get_lead(ecg, metadata, args.lead)

    fs = metadata["fs"]
    gap_samples = int(round(args.gap_seconds * fs))

    if gap_samples <= 0:
        raise ValueError("--gap-seconds must produce at least one sample.")

    if gap_samples >= len(lead_signal):
        raise ValueError(
            f"Gap is too long. Signal has {len(lead_signal)} samples "
            f"({len(lead_signal) / fs:.2f} seconds)."
        )

    print("ECG shape:", ecg.shape)
    print("Sampling frequency:", fs, "Hz")
    print("Duration:", len(lead_signal) / fs, "seconds")
    print("Lead:", args.lead)
    print("Lead shape:", lead_signal.shape)

    # --------------------------------------------------
    # 3. Simulate one missing block
    # --------------------------------------------------
    rng = np.random.default_rng(args.seed)

    mask = make_block_mask(
        length=len(lead_signal),
        gap_samples=gap_samples,
        rng=rng
    )

    corrupted = apply_mask(
        lead_signal,
        mask,
        fill_value=0.0
    )

    # --------------------------------------------------
    # 4. Reconstruct
    # --------------------------------------------------
    linear = linear_interpolate(corrupted, mask)
    cubic = cubic_spline_interpolate(corrupted, mask)

    # --------------------------------------------------
    # 5. Evaluate only the hidden samples
    # --------------------------------------------------
    linear_metrics = evaluate_reconstruction(
        lead_signal, linear, mask
    )
    cubic_metrics = evaluate_reconstruction(
        lead_signal, cubic, mask
    )

    print("\nLinear interpolation")
    for name, value in linear_metrics.items():
        print(f"  {name}: {value:.6f}")

    print("\nCubic spline interpolation")
    for name, value in cubic_metrics.items():
        print(f"  {name}: {value:.6f}")

    # --------------------------------------------------
    # 6. Plot
    # --------------------------------------------------
    t = np.arange(len(lead_signal)) / fs
    missing_indices = np.flatnonzero(~mask)

    # NaN is used only for plotting so the missing section appears blank.
    corrupted_for_plot = corrupted.copy()
    corrupted_for_plot[~mask] = np.nan

    plt.figure(figsize=(13, 5))
    plt.plot(t, lead_signal, label="Ground truth", linewidth=2)
    plt.plot(t, corrupted_for_plot, label="Observed signal", linewidth=1.5)
    plt.plot(t, linear, label="Linear interpolation", linewidth=1.2)
    plt.plot(t, cubic, label="Cubic spline", linewidth=1.2)

    plt.axvspan(
        t[missing_indices[0]],
        t[missing_indices[-1]],
        alpha=0.15,
        label="Simulated missing region"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title(
        f"{record_path.name} — Lead {args.lead} — "
        f"{args.gap_seconds:g}s missing block"
    )
    plt.legend()
    plt.tight_layout()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "test_pipeline.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
