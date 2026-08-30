"""Real-data validation of the residual-speckle refinements.

Winner combo from the GT protocol: PH-b3-f0.5 + whiteness loss (C) +
final non-local polish (B, s=0.5). On real (oversampled) data the
whiteness lags move to (3,4,5) — lag-1/2 autocorrelation (~0.5/0.12) is
inherent speckle correlation, not residue. Fallback row without C is
produced post-hoc from the saved PH-b3 output. Zoom crops show the
visual grain reduction (the user-facing goal)."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import (denoise, TrainConfig, load_quadpol_tiffs,
                     load_quadpol_phase)
from sspmnet.phase_data import phase_feedback_maps
from sspmnet.losses import nl_polish, _box_blur
from sspmnet.metrics import (find_top_k_rois, enl_roi_multi, epi_metric,
                             ssim_metric, reciprocity_metrics)

OUT = "results/ri_compare"
amp, ri = load_quadpol_tiffs("data/tiff")
pha = load_quadpol_phase("data/tiff")
PH = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
          guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5)

print("\n=== PH-b3 + C(white, lags 3-5) ===", flush=True)
res = denoise(amp, TrainConfig(**PH, whiteness_lambda=0.05,
                               whiteness_lags=(3, 4, 5)),
              ri_pair=ri, pha=pha)
dC = res["denoised"]
np.save(f"{OUT}/denoised_PH_b3_C.npy", dC)

# ── post-hoc polish (B), normalized scale, protected ──
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q99 = np.quantile(amp, 0.99, axis=(1, 2), keepdims=True)
ri_t = torch.from_numpy(np.clip(ri/np.maximum(q99[None],1e-9),0,5)).float().to(dev)
ga = (ri_t**2).mean(dim=(0,1)).sqrt()[None,None]
mu = _box_blur(ga,9,1); m2 = _box_blur(ga**2,9,1)
cv = torch.sqrt((m2-mu**2).clamp(min=0))/(mu+1e-6)
pm = phase_feedback_maps(pha=pha)
prot = torch.maximum(torch.sigmoid((cv-0.3)/0.075),
                     torch.from_numpy(pm["det"]).float()[None,None].to(dev))
def polish(d_np, s_):
    dn = torch.from_numpy(np.clip(d_np/np.maximum(q99,1e-9),0,1)).float()[None].to(dev)
    out = nl_polish(dn, window=9, sigma=0.1, strength=s_, protect=prot).clamp(0,1)
    return out[0].cpu().numpy().astype(np.float32)*q99.astype(np.float32)

runs = {
    "winner":     np.load(f"{OUT}/denoised_winner.npy"),
    "PH-b3":      np.load(f"{OUT}/denoised_PH_b3.npy"),
    "PH-b3+B0.5": polish(np.load(f"{OUT}/denoised_PH_b3.npy"), 0.5),
    "PH-b3+C":    dC,
    "PH-b3+C+B0.5": polish(dC, 0.5),
}
np.save(f"{OUT}/denoised_PH_b3_C_B05.npy", runs["PH-b3+C+B0.5"])

rois, rs = find_top_k_rois(amp[1])
def ratio_enl(d, c):
    eps=1e-3; m=(d[c]>2)&(amp[c]>0)
    rI=(amp[c].astype(np.float64)**2+eps)/(d[c].astype(np.float64)**2+eps)
    v=rI[m]; return (v.mean()/v.std())**2

cols = ["corr(HV,VH)","ENL-ROI(HV)","ENL-ROI(HH)","EPI(HV)","SSIM(HV)",
        "ENLr(HH)","ENLr(HV)"]
hdr = "  {:<14}".format("Method")+"".join(f"{c:>13}" for c in cols)
print("\nReal-data metrics"); print(hdr); lines=[hdr]
for name, d in runs.items():
    rec = reciprocity_metrics(d[1], d[2])
    vals=[rec["corr"], enl_roi_multi(d[1],rois,rs), enl_roi_multi(d[0],rois,rs),
          epi_metric(amp[1],d[1]), ssim_metric(amp[1],d[1]),
          ratio_enl(d,0), ratio_enl(d,1)]
    line = "  {:<14}".format(name)+"".join(f"{v:>13.4f}" for v in vals)
    print(line); lines.append(line)
with open(f"{OUT}/metrics_polish_real.txt","w") as f:
    f.write("\n".join(lines)+"\n")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sel = ["Noisy","winner","PH-b3","PH-b3+C+B0.5"]
imgs = {"Noisy": amp, **runs}
fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
for col, nm in enumerate(sel):
    for r_i,(ch,cn) in enumerate([(0,"HH"),(1,"HV")]):
        v = np.clip(imgs[nm][ch]/np.quantile(amp[ch],0.99),0,1)
        axes[r_i,col].imshow(v,cmap="gray",vmin=0,vmax=1)
        axes[r_i,col].set_title(f"{nm} — {cn}"); axes[r_i,col].axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/compare_polish_real.png", dpi=130)

# zoom crops: river bank / building grid / vegetation (HH)
crops = {"river bank": (100, 220, 280, 400),
         "building grid": (220, 340, 110, 230),
         "vegetation": (380, 500, 300, 420)}
fig, axes = plt.subplots(3, 4, figsize=(17, 12.5))
for r_i,(cnm,(y0,y1,x0,x1)) in enumerate(crops.items()):
    for col, nm in enumerate(sel):
        v = np.clip(imgs[nm][0]/np.quantile(amp[0],0.99),0,1)[y0:y1, x0:x1]
        axes[r_i,col].imshow(v,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        axes[r_i,col].set_title(f"{nm} — {cnm}", fontsize=10)
        axes[r_i,col].axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/compare_polish_zoom.png", dpi=130)
print("Saved figures")
