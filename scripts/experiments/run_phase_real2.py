"""Final real-data validation of the tuned phase feedback: winner+PH at
b=3 and b=5 (f=0.5), tabulated against the saved baseline / winner /
winner+PH(b1.5) outputs; full table + side-by-side figure."""
import os, sys
import numpy as np
sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs, load_quadpol_phase
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)

OUT = "results/ri_compare"
amp, ri = load_quadpol_tiffs("data/tiff")
pha = load_quadpol_phase("data/tiff")
W = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
         guide_cv_protect=0.3)

runs = {
    "baseline":   np.load(f"{OUT}/denoised_baseline.npy"),
    "winner":     np.load(f"{OUT}/denoised_winner.npy"),
    "PH-b1.5":    np.load(f"{OUT}/denoised_winner_PH.npy"),
}
for name, b in [("PH-b3", 3.0), ("PH-b5", 5.0)]:
    print(f"\n=== {name} ===", flush=True)
    cfg = TrainConfig(**W, phase_smooth_boost=b, phase_fidelity=0.5)
    res = denoise(amp, cfg, ri_pair=ri, pha=pha)
    runs[name] = res["denoised"]
    np.save(f"{OUT}/denoised_{name.replace('.','p').replace('-','_')}.npy",
            res["denoised"])

rois, rs = find_top_k_rois(amp[1])
def ratio_enl(d, c):
    eps = 1e-3
    m = (d[c] > 2) & (amp[c] > 0)
    rI = (amp[c].astype(np.float64)**2 + eps) / (d[c].astype(np.float64)**2 + eps)
    v = rI[m]
    return (v.mean()/v.std())**2

cols = ["corr(HV,VH)", "ENL-ROI(HV)", "ENL-ROI(HH)", "EPI(HV)", "SSIM(HV)",
        "ENLr(HH)", "ENLr(HV)"]
hdr = "  {:<12}".format("Method") + "".join(f"{c:>13}" for c in cols)
print("\nReal-data metrics (noisy-reference; ratio-ENL ideal ~= 1)")
print(hdr); lines=[hdr]
for name, d in runs.items():
    rec = reciprocity_metrics(d[1], d[2])
    vals = [rec["corr"], enl_roi_multi(d[1], rois, rs),
            enl_roi_multi(d[0], rois, rs), epi_metric(amp[1], d[1]),
            ssim_metric(amp[1], d[1]), ratio_enl(d, 0), ratio_enl(d, 1)]
    line = "  {:<12}".format(name) + "".join(f"{v:>13.4f}" for v in vals)
    print(line); lines.append(line)
with open(f"{OUT}/metrics_real_phase2.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = ["Noisy", "winner", "PH-b3", "PH-b5"]
imgs = [amp, runs["winner"], runs["PH-b3"], runs["PH-b5"]]
fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
for col, (nm, im_) in enumerate(zip(names, imgs)):
    for r_i, (ch, ch_name) in enumerate([(0, "HH"), (1, "HV")]):
        v = np.clip(im_[ch] / np.quantile(amp[ch], 0.99), 0, 1)
        axes[r_i, col].imshow(v, cmap="gray", vmin=0, vmax=1)
        axes[r_i, col].set_title(f"{nm} — {ch_name}"); axes[r_i, col].axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/compare_real_phase2.png", dpi=130)
print("Saved", f"{OUT}/compare_real_phase2.png")
