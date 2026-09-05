from pathlib import Path
import re
import numpy as np


DEFAULT_LEAD_NAMES = [
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


def read_header(header_path):
    """
    Read a WFDB-style .hea header file.

    Parameters
    ----------
    header_path : str or pathlib.Path
        Path to a PTB-XL .hea file.

    Returns
    -------
    dict
        Metadata containing:
        - record_name
        - n_leads
        - fs
        - n_samples
        - lead_names
        - gains
    """
    header_path = Path(header_path)

    with header_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    first = lines[0].split()

    record_name = first[0]
    n_leads = int(first[1])
    fs = int(float(first[2]))
    n_samples = int(first[3])

    signal_lines = lines[1:1 + n_leads]

    lead_names = []
    gains = []

    for line in signal_lines:
        parts = line.split()

        # In PTB-XL, the final field is the lead name.
        lead_names.append(parts[-1])

        # PTB-XL gain fields may look like:
        # "1000.0/mV" or "1000.0(0)/mV".
        gain_field = parts[2]
        gain_match = re.match(r"([-+]?\d*\.?\d+)", gain_field)

        if gain_match is None:
            raise ValueError(
                f"Could not parse gain from header field: {gain_field}"
            )

        gain_value = float(gain_match.group(1))
        gains.append(gain_value)

    return {
        "record_name": record_name,
        "n_leads": n_leads,
        "fs": fs,
        "n_samples": n_samples,
        "lead_names": lead_names,
        "gains": np.asarray(gains, dtype=np.float32),
    }


def load_ptbxl_record(record_path):
    """
    Load one PTB-XL record from matching .dat and .hea files.

    Parameters
    ----------
    record_path : str or pathlib.Path
        Record path without extension, e.g.
        'records100/00000/00001_lr'

    Returns
    -------
    ecg : np.ndarray
        ECG signal in millivolts with shape (n_samples, n_leads).
    metadata : dict
        Parsed header metadata.
    """
    record_path = Path(record_path)

    hea_path = record_path.with_suffix(".hea")
    dat_path = record_path.with_suffix(".dat")

    if not hea_path.exists():
        raise FileNotFoundError(f"Header file not found: {hea_path}")

    if not dat_path.exists():
        raise FileNotFoundError(f"Data file not found: {dat_path}")

    metadata = read_header(hea_path)

    raw = np.fromfile(dat_path, dtype="<i2")

    expected = metadata["n_samples"] * metadata["n_leads"]

    if raw.size != expected:
        raise ValueError(
            f"Unexpected number of samples in {dat_path}. "
            f"Expected {expected}, found {raw.size}."
        )

    raw = raw.reshape(
        metadata["n_samples"],
        metadata["n_leads"]
    )

    gains = metadata["gains"]

    # Convert digital values to mV.
    ecg = raw.astype(np.float32) / gains[None, :]

    return ecg, metadata


def get_lead(ecg, metadata, lead="II"):
    """
    Extract one ECG lead by name.

    Parameters
    ----------
    ecg : np.ndarray
        ECG array with shape (n_samples, n_leads).
    metadata : dict
        Metadata returned by load_ptbxl_record().
    lead : str
        Lead name, e.g. 'II', 'V1', 'V5'.

    Returns
    -------
    np.ndarray
        One-dimensional ECG signal.
    """
    lead_names = metadata["lead_names"]

    if lead not in lead_names:
        raise ValueError(
            f"Lead '{lead}' not found. Available leads: {lead_names}"
        )

    lead_index = lead_names.index(lead)
    return ecg[:, lead_index].copy()


def find_records(data_dir, pattern="*_lr.hea"):
    """
    Recursively find PTB-XL records with matching .hea and .dat files.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Root directory containing PTB-XL files.
    pattern : str
        Header filename pattern.

    Returns
    -------
    list[pathlib.Path]
        Record paths without file extensions.
    """
    data_dir = Path(data_dir)

    records = []

    for hea_path in sorted(data_dir.rglob(pattern)):
        record_path = hea_path.with_suffix("")

        if record_path.with_suffix(".dat").exists():
            records.append(record_path)

    return records
