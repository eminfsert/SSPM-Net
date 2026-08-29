"""Head-to-head: amplitude-only SSPM-Net vs. SSPM-Net + complex (RI) auxiliaries.

Runs the zero-shot pipeline twice on the same quad-pol patch with identical
seeds and budgets:

    baseline : amplitude only (the thesis pipeline)
    +RI      : |Re|/|Im| Noise2Noise targets + multi-look guided TV / NLM
               (see sspmnet/complex_data.py)

and prints the fair metrics table (ROIs chosen once on the noisy image) plus
a side-by-side PNG.

Usage:
    python scripts/compare_ri.py [--tiff-dir data/tiff] [--iters 700]
                                 [--out results/ri_compare]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sspmnet import denoise, TrainConfig, load_quadpol_tiffs           # noqa: E402
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)          # noqa: E402


def metric_row(name, out, noisy, rois, rs):
    hv_n, vh_n = noisy[1], noisy[2]
    hv, vh = out[1], out[2]
    rec = reciprocity_metrics(hv, vh)
    return {
        "method": name,
        "corr(HV,VH)": rec["corr"],
        "MAD": rec["mad"],
        "RMSE": rec["rmse"],
        "ENL-ROI(HV)": enl_roi_multi(hv, rois, rs),
        "ENL-ROI(HH)": enl_roi_multi(out[0], rois, rs),
        "EPI(HV)": epi_metric(hv_n, hv),
        "EPI(HH)": epi_metric(noisy[0], out[0]),
        "SSIM(HV)": ssim_metric(hv_n, hv),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiff-dir", default="data/tiff")
    ap.add_argument("--iters", type=int, default=700)
    ap.add_argument("--out", default="results/ri_compare")
    ap.add_argument("--ri-weight", type=float, default=0.6)
    ap.add_argument("--guide-alpha", type=float, default=3.0)
    ap.add_argument("--merlin", action="store_true",
                    help="also run the MERLIN input-separation variant")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    amp, ri = load_quadpol_tiffs(args.tiff_dir)
    print(f"Loaded {args.tiff_dir}: amp {amp.shape}, ri {ri.shape}")

    variants = [("baseline", None, "targets"), ("+RI", ri, "targets")]
    if args.merlin:
        variants.append(("MERLIN", ri, "merlin"))

    runs = {}
    for name, ri_pair, mode in variants:
        print(f"\n=== {name} ===")
        cfg = TrainConfig(iters=args.iters, ri_weight=args.ri_weight,
                          guide_alpha=args.guide_alpha, ri_mode=mode)
        res = denoise(amp, cfg, ri_pair=ri_pair)
        runs[name] = res["denoised"]
        np.save(os.path.join(args.out, f"denoised_{name.strip('+')}.npy"),
                res["denoised"])

    # ── Fair metrics: ROIs picked once on the noisy HV channel ──
    rois, rs = find_top_k_rois(amp[1])
    rows = [metric_row("Noisy", amp, amp, rois, rs)]
    for name, out in runs.items():
        rows.append(metric_row(name, out, amp, rois, rs))

    cols = list(rows[0].keys())[1:]
    hdr = "  {:<10}".format("Method") + "".join(f"{c:>13}" for c in cols)
    print("\n" + hdr)
    lines = [hdr]
    for r in rows:
        line = "  {:<10}".format(r["method"]) + "".join(
            f"{r[c]:>13.4f}" for c in cols)
        print(line)
        lines.append(line)
    with open(os.path.join(args.out, "metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ── Side-by-side PNG (HH and HV) ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["Noisy"] + list(runs.keys())
        imgs = [amp] + list(runs.values())
        fig, axes = plt.subplots(2, len(names), figsize=(4.7 * len(names), 9.5))
        for col, (nm, im) in enumerate(zip(names, imgs)):
            for row, (ch, ch_name) in enumerate([(0, "HH"), (1, "HV")]):
                v = np.clip(im[ch] / np.quantile(amp[ch], 0.99), 0, 1)
                axes[row, col].imshow(v, cmap="gray", vmin=0, vmax=1)
                axes[row, col].set_title(f"{nm} — {ch_name}")
                axes[row, col].axis("off")
        fig.tight_layout()
        png = os.path.join(args.out, "compare.png")
        fig.savefig(png, dpi=130)
        print(f"\nSaved {png}")
    except Exception as e:                                    # headless-safe
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
