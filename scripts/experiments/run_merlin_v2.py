"""MERLIN v2: conditional-median (L1) calibration + lighter TV
(dense full-pixel supervision needs less regularization)."""
import os
import sys

import numpy as np

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")

from sspmnet import denoise, TrainConfig, load_quadpol_tiffs
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)

OUT = "results/ri_compare"
amp, ri = load_quadpol_tiffs("data/tiff")          # now l1-calibrated

cfg = TrainConfig(iters=700, ri_mode="merlin", tv_mult=5.0)
res = denoise(amp, cfg, ri_pair=ri)
np.save(os.path.join(OUT, "denoised_MERLIN_v2.npy"), res["denoised"])

runs = {
    "baseline": np.load(os.path.join(OUT, "denoised_baseline.npy")),
    "MERLIN": np.load(os.path.join(OUT, "denoised_MERLIN.npy")),
    "MERLIN-v2": res["denoised"],
}

rois, rs = find_top_k_rois(amp[1])


def row(name, out):
    rec = reciprocity_metrics(out[1], out[2])
    return (name, rec["corr"], rec["mad"], rec["rmse"],
            enl_roi_multi(out[1], rois, rs), enl_roi_multi(out[0], rois, rs),
            epi_metric(amp[1], out[1]), epi_metric(amp[0], out[0]),
            ssim_metric(amp[1], out[1]))


rows = [row("Noisy", amp)] + [row(k, v) for k, v in runs.items()]
hdr = ("  {:<10}".format("Method")
       + "".join(f"{c:>13}" for c in ["corr(HV,VH)", "MAD", "RMSE",
                                      "ENL-ROI(HV)", "ENL-ROI(HH)",
                                      "EPI(HV)", "EPI(HH)", "SSIM(HV)"]))
print(hdr)
lines = [hdr]
for r in rows:
    line = "  {:<10}".format(r[0]) + "".join(f"{v:>13.4f}" for v in r[1:])
    print(line)
    lines.append(line)
with open(os.path.join(OUT, "metrics_merlin_v2.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("ch means:", {k: [round(float(v[c].mean()), 1) for c in range(4)]
                    for k, v in runs.items()})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = ["Noisy"] + list(runs.keys())
imgs = [amp] + list(runs.values())
fig, axes = plt.subplots(2, len(names), figsize=(4.7 * len(names), 9.5))
for col, (nm, im) in enumerate(zip(names, imgs)):
    for r_i, (ch, ch_name) in enumerate([(0, "HH"), (1, "HV")]):
        v = np.clip(im[ch] / np.quantile(amp[ch], 0.99), 0, 1)
        axes[r_i, col].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[r_i, col].set_title(f"{nm} — {ch_name}")
        axes[r_i, col].axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_merlin_v2.png"), dpi=130)
print("Saved", os.path.join(OUT, "compare_merlin_v2.png"))
