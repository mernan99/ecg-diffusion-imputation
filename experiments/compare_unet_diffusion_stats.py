
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "rmse": ("RMSE", False),
    "mae": ("MAE", False),
    "correlation": ("Correlation", True),
    "r_peak_f1": ("Peak-event F1", True),
}

def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for j, idx in enumerate(order):
        running = max(running, min(1.0, p[idx] * (m - j)))
        out[idx] = running
    return out

def bootstrap_ci(x, stat, n_boot, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        vals[i] = stat(x[rng.integers(0, n, n)])
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))

def rank_biserial(adv):
    adv = np.asarray(adv, float)
    adv = adv[np.isfinite(adv) & (adv != 0)]
    if len(adv) == 0:
        return 0.0
    r = rankdata(np.abs(adv))
    pos = r[adv > 0].sum()
    neg = r[adv < 0].sum()
    return float((pos - neg) / (pos + neg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path,
                    default=ROOT / "results" / "fixed_test_results.csv")
    ap.add_argument("--diffusion", type=Path,
                    default=ROOT / "results" / "diffusion_test_results.csv")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "results" / "fold10_mask_manifest.csv")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    base = pd.read_csv(args.baseline)
    diff = pd.read_csv(args.diffusion)

    u = base[base["method"] == "unet_autoencoder"].copy()
    d = diff[diff["method"] == "diffusion"].copy()

    keys = ["record", "gap_seconds", "trial"]
    for k in keys:
        if k not in u or k not in d:
            raise ValueError(f"Missing pairing key: {k}")

    metrics = [m for m in METRICS if m in u.columns and m in d.columns]
    if not metrics:
        raise ValueError("No common metrics found.")

    ucols = keys + metrics
    dcols = keys + metrics
    if "samples_per_input" in d.columns:
        dcols.append("samples_per_input")

    paired = u[ucols].merge(
        d[dcols], on=keys, suffixes=("_unet", "_diffusion"),
        validate="one_to_one"
    )

    if args.manifest.exists():
        man = pd.read_csv(args.manifest)
        keep = keys + [c for c in ["start_sample", "end_sample"] if c in man.columns]
        man = man[keep]
        if man.duplicated(keys).any():
            raise ValueError("Duplicate cases in mask manifest.")
        paired = paired.merge(man, on=keys, how="left", validate="one_to_one")
        if any(c in paired and paired[c].isna().any() for c in ["start_sample", "end_sample"]):
            raise ValueError("Some paired cases are missing from the frozen manifest.")

    for m in metrics:
        label, higher = METRICS[m]
        if higher:
            adv = paired[f"{m}_diffusion"] - paired[f"{m}_unet"]
        else:
            adv = paired[f"{m}_unet"] - paired[f"{m}_diffusion"]
        paired[f"{m}_diffusion_advantage"] = adv

    rows = []
    for m in metrics:
        label, higher = METRICS[m]
        for gap, g in paired.groupby("gap_seconds"):
            a = g[f"{m}_unet"].to_numpy(float)
            b = g[f"{m}_diffusion"].to_numpy(float)
            adv = g[f"{m}_diffusion_advantage"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(adv)
            a, b, adv = a[ok], b[ok], adv[ok]
            if len(adv) == 0:
                continue

            mean_ci = bootstrap_ci(
                adv, np.mean, args.bootstrap,
                args.seed + int(round(gap * 1000))
            )
            median_ci = bootstrap_ci(
                adv, np.median, args.bootstrap,
                args.seed + 100000 + int(round(gap * 1000))
            )

            if np.allclose(adv, 0):
                wstat, pval = 0.0, 1.0
            else:
                res = wilcoxon(
                    adv, zero_method="wilcox",
                    alternative="two-sided", method="approx"
                )
                wstat, pval = float(res.statistic), float(res.pvalue)

            tol = 1e-12
            dw = int(np.sum(adv > tol))
            uw = int(np.sum(adv < -tol))
            ties = len(adv) - dw - uw

            rows.append({
                "metric": m,
                "metric_label": label,
                "gap_seconds": float(gap),
                "n_pairs": len(adv),
                "mean_unet": float(np.mean(a)),
                "mean_diffusion": float(np.mean(b)),
                "median_unet": float(np.median(a)),
                "median_diffusion": float(np.median(b)),
                "mean_diffusion_advantage": float(np.mean(adv)),
                "mean_advantage_ci95_low": mean_ci[0],
                "mean_advantage_ci95_high": mean_ci[1],
                "median_diffusion_advantage": float(np.median(adv)),
                "median_advantage_ci95_low": median_ci[0],
                "median_advantage_ci95_high": median_ci[1],
                "diffusion_wins": dw,
                "unet_wins": uw,
                "ties": ties,
                "diffusion_win_rate": dw / len(adv),
                "unet_win_rate": uw / len(adv),
                "rank_biserial_effect": rank_biserial(adv),
                "wilcoxon_statistic": wstat,
                "wilcoxon_p": pval,
            })

    summary = pd.DataFrame(rows)
    summary["wilcoxon_p_holm"] = np.nan
    for m, idx in summary.groupby("metric").groups.items():
        idx = list(idx)
        summary.loc[idx, "wilcoxon_p_holm"] = holm(
            summary.loc[idx, "wilcoxon_p"].to_numpy(float)
        )
    summary["significant_holm_0_05"] = summary["wilcoxon_p_holm"] < .05

    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    paired.to_csv(outdir / "paired_unet_diffusion_results.csv", index=False)
    summary.to_csv(outdir / "paired_model_statistics.csv", index=False)

    # RMSE win-rate plot
    if "rmse" in metrics:
        r = summary[summary["metric"] == "rmse"].sort_values("gap_seconds")
        x = np.arange(len(r))
        plt.figure(figsize=(8, 6))
        plt.bar(x, r["diffusion_win_rate"], label="Diffusion lower RMSE")
        plt.bar(x, r["unet_win_rate"], bottom=r["diffusion_win_rate"],
                label="U-Net lower RMSE")
        plt.xticks(x, [f"{v:g}s" for v in r["gap_seconds"]])
        plt.ylim(0, 1)
        plt.xlabel("Missing gap length")
        plt.ylabel("Fraction of paired cases")
        plt.title("Paired U-Net vs diffusion RMSE win rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "paired_rmse_win_rate.png", dpi=200)
        plt.close()

    # Separate paired-advantage plots
    for m in [x for x in ["rmse", "correlation"] if x in metrics]:
        groups, labels = [], []
        for gap in sorted(paired["gap_seconds"].unique()):
            vals = paired.loc[
                np.isclose(paired["gap_seconds"], gap),
                f"{m}_diffusion_advantage"
            ].to_numpy(float)
            groups.append(vals[np.isfinite(vals)])
            labels.append(f"{gap:g}s")
        plt.figure(figsize=(8, 6))
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Missing gap length")
        plt.ylabel(f"Diffusion advantage in {METRICS[m][0]}\n(positive = diffusion better)")
        plt.title(f"Paired diffusion advantage — {METRICS[m][0]}")
        plt.tight_layout()
        plt.savefig(outdir / f"paired_{m}_advantage.png", dpi=200)
        plt.close()

    print("Matched test cases:", len(paired))
    if "samples_per_input" in paired.columns:
        print("Diffusion samples per input:",
              sorted(paired["samples_per_input"].dropna().unique().tolist()))
    print("\nPositive advantage/effect size always means diffusion is better.\n")
    cols = [
        "metric_label", "gap_seconds", "n_pairs",
        "mean_unet", "mean_diffusion",
        "mean_diffusion_advantage",
        "mean_advantage_ci95_low", "mean_advantage_ci95_high",
        "diffusion_win_rate", "rank_biserial_effect",
        "wilcoxon_p_holm", "significant_holm_0_05",
    ]
    print(summary[cols].to_string(index=False))
    print("\nSaved:")
    print(outdir / "paired_unet_diffusion_results.csv")
    print(outdir / "paired_model_statistics.csv")
    print(outdir / "paired_rmse_win_rate.png")
    if "rmse" in metrics:
        print(outdir / "paired_rmse_advantage.png")
    if "correlation" in metrics:
        print(outdir / "paired_correlation_advantage.png")

if __name__ == "__main__":
    main()
