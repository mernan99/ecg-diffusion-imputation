import numpy as np
from scipy.signal import find_peaks


def mae(original, reconstructed, mask=None):
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if mask is not None:
        original = original[~mask]
        reconstructed = reconstructed[~mask]

    return float(np.mean(np.abs(original - reconstructed)))


def rmse(original, reconstructed, mask=None):
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if mask is not None:
        original = original[~mask]
        reconstructed = reconstructed[~mask]

    return float(
        np.sqrt(
            np.mean((original - reconstructed) ** 2)
        )
    )


def pearson_correlation(original, reconstructed, mask=None):
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if mask is not None:
        original = original[~mask]
        reconstructed = reconstructed[~mask]

    if original.size < 2:
        return np.nan

    if np.std(original) == 0 or np.std(reconstructed) == 0:
        return np.nan

    return float(
        np.corrcoef(original, reconstructed)[0, 1]
    )


def evaluate_reconstruction(original, reconstructed, mask):
    return {
        "mae": mae(original, reconstructed, mask),
        "rmse": rmse(original, reconstructed, mask),
        "correlation": pearson_correlation(
            original, reconstructed, mask
        ),
    }


def detect_r_peaks(
    signal,
    fs,
    min_distance_seconds=0.30,
    prominence=None,
):
    """
    Simple R-peak-like detector for research evaluation.

    This is a heuristic, not a clinical ECG interpretation algorithm.
    """
    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - np.median(signal)
    magnitude = np.abs(centered)

    if prominence is None:
        prominence = max(
            0.5 * np.std(signal),
            0.05
        )

    distance = max(
        1,
        int(round(min_distance_seconds * fs))
    )

    peaks, _ = find_peaks(
        magnitude,
        distance=distance,
        prominence=prominence
    )

    return peaks.astype(int)


def _match_peaks(true_peaks, predicted_peaks, tolerance_samples):
    true_peaks = sorted(int(x) for x in true_peaks)
    predicted_peaks = sorted(int(x) for x in predicted_peaks)

    used = set()
    matches = []

    for true_peak in true_peaks:
        best_j = None
        best_distance = None

        for j, pred_peak in enumerate(predicted_peaks):
            if j in used:
                continue

            distance = abs(pred_peak - true_peak)

            if distance <= tolerance_samples:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_j = j

        if best_j is not None:
            used.add(best_j)
            matches.append(
                (
                    true_peak,
                    predicted_peaks[best_j],
                    best_distance
                )
            )

    return matches


def evaluate_r_peaks(
    original,
    reconstructed,
    mask,
    fs,
    tolerance_seconds=0.10,
):
    """
    Evaluate R-peak-like events inside the missing region.

    Returns recall, precision, F1 and mean matched timing error.
    """
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    mask = np.asarray(mask, dtype=bool)

    true_all = detect_r_peaks(original, fs)
    pred_all = detect_r_peaks(reconstructed, fs)

    true_peaks = np.asarray(
        [p for p in true_all if not mask[p]],
        dtype=int
    )

    predicted_peaks = np.asarray(
        [p for p in pred_all if not mask[p]],
        dtype=int
    )

    tolerance_samples = max(
        1,
        int(round(tolerance_seconds * fs))
    )

    matches = _match_peaks(
        true_peaks,
        predicted_peaks,
        tolerance_samples
    )

    tp = len(matches)

    if len(true_peaks) > 0:
        recall = tp / len(true_peaks)
    else:
        recall = np.nan

    if len(predicted_peaks) > 0:
        precision = tp / len(predicted_peaks)
    else:
        precision = 0.0 if len(true_peaks) > 0 else np.nan

    if (
        np.isfinite(recall)
        and np.isfinite(precision)
        and (recall + precision) > 0
    ):
        f1 = (
            2 * recall * precision
            / (recall + precision)
        )
    elif np.isfinite(recall) and np.isfinite(precision):
        f1 = 0.0
    else:
        f1 = np.nan

    if matches:
        timing_error_ms = float(
            np.mean([
                distance / fs * 1000.0
                for _, _, distance in matches
            ])
        )
    else:
        timing_error_ms = np.nan

    return {
        "r_peak_recall":
            float(recall)
            if np.isfinite(recall)
            else np.nan,
        "r_peak_precision":
            float(precision)
            if np.isfinite(precision)
            else np.nan,
        "r_peak_f1":
            float(f1)
            if np.isfinite(f1)
            else np.nan,
        "r_peak_timing_error_ms":
            timing_error_ms,
        "true_r_peaks":
            int(len(true_peaks)),
        "predicted_r_peaks":
            int(len(predicted_peaks)),
    }
