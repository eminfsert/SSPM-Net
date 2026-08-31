"""Diagnostic: do the 'surface' (HH-VV) and 'det' (spatial phase) maps carry
real structure on this patch, or are they estimator noise?"""
import os, sys
import numpy as np
sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import load_quadpol_tiffs, load_quadpol_phase
from sspmnet.phase_data import phase_feedback_maps, _local_mean

amp, ri = load_quadpol_tiffs("data/tiff")
pha = load_quadpol_phase("data/tiff")
pm = phase_feedback_maps(pha=pha, win=7)
snr, surf, det = pm["snr"], pm["surface"], pm["det"]

span = amp[0]**2 + 2*amp[1]**2 + amp[3]**2
logspan = np.log(span + 1e-6)
def local_cv(x, w=7):
    m = _local_mean(x.astype(np.float64), w)
    s = np.sqrt(np.maximum(_local_mean(x.astype(np.float64)**2, w) - m**2, 0))
    return s / (m + 1e-6)
cv_span = local_cv(amp[0])
depol = amp[1]**2 / (span + 1e-6)
copol_ratio = np.log((amp[0]**2 + 1e-6) / (amp[3]**2 + 1e-6))

def c(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

print("map stats (robust-normalized to [0,1]):")
for n, g in [("snr", snr), ("surface", surf), ("det", det)]:
    print(f"  {n:<8} mean {g.mean():.3f}  std {g.std():.3f}  "
          f"q05 {np.quantile(g,.05):.3f} q50 {np.quantile(g,.5):.3f} "
          f"q95 {np.quantile(g,.95):.3f}")

print("\ncorrelations with scene descriptors:")
for n, g in [("snr", snr), ("surface", surf), ("det", det)]:
    print(f"  {n:<8} logspan {c(g,logspan):+.3f}  CV(HH) {c(g,cv_span):+.3f}  "
          f"depol {c(g,depol):+.3f}  log(HH/VV) {c(g,copol_ratio):+.3f}")
print(f"\n  corr(surface, snr) {c(surf,snr):+.3f}   corr(det, snr) {c(det,snr):+.3f}"
      f"   corr(det, surface) {c(det,surf):+.3f}")

thr = np.quantile(span, 0.99)
bright = span >= thr
print(f"\n  det   on brightest 1%: {det[bright].mean():.3f}  vs rest {det[~bright].mean():.3f}")
print(f"  surf  on brightest 1%: {surf[bright].mean():.3f}  vs rest {surf[~bright].mean():.3f}")
lo = cv_span <= np.quantile(cv_span, 0.20)
hi = cv_span >= np.quantile(cv_span, 0.80)
print(f"  surf  on low-CV (homogeneous) {surf[lo].mean():.3f} vs high-CV {surf[hi].mean():.3f}")
print(f"  det   on low-CV {det[lo].mean():.3f} vs high-CV {det[hi].mean():.3f}")

rng = np.random.default_rng(0)
pha_null = rng.uniform(0, np.pi, size=pha.shape)
pm0 = phase_feedback_maps(pha=pha_null, win=7)
print("\nnull control (uniform random phase, same estimator):")
for n, g, g0 in [("snr", snr, pm0["snr"]), ("surface", surf, pm0["surface"]),
                 ("det", det, pm0["det"])]:
    print(f"  {n:<8} real std {g.std():.3f} vs null std {g0.std():.3f}   "
          f"| corr w/ logspan real {c(g,logspan):+.3f} null {c(g0,logspan):+.3f}")

np.save("results/ri_compare/phase_maps.npy", np.stack([snr, surf, det]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
ax[0].imshow(np.clip(amp[0]/np.quantile(amp[0],0.99),0,1), cmap="gray"); ax[0].set_title("HH amplitude")
for i,(n,g) in enumerate([("snr (HV-VH)",snr),("surface (HH-VV)",surf),("det (spatial)",det)]):
    im = ax[i+1].imshow(g, cmap="viridis", vmin=0, vmax=1); ax[i+1].set_title(n)
    plt.colorbar(im, ax=ax[i+1], fraction=0.046)
for a in ax: a.axis("off")
fig.tight_layout(); fig.savefig("results/ri_compare/phase_maps_extra.png", dpi=130)
print("\nSaved results/ri_compare/phase_maps_extra.png")
