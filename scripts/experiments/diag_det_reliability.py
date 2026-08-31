"""Reliability test: 'det' is the mean of 4 per-channel spatial phase
coherences. If it measures a scene property (deterministic targets), the
per-channel estimates must agree with each other; if it is estimator noise
they are independent. Same test for a point-target proxy from amplitude."""
import os, sys
import numpy as np
sys.path.insert(0, "/content/SSPM-Net"); os.chdir("/content/SSPM-Net")
from sspmnet import load_quadpol_tiffs, load_quadpol_phase
from sspmnet.phase_data import _local_coherence, _local_mean, _robust_norm

amp, _ = load_quadpol_tiffs("data/tiff")
pha = load_quadpol_phase("data/tiff")
u = np.exp(2j * pha.astype(np.float64))
POLS = ["HH", "HV", "VH", "VV"]
d = [_local_mean(_local_coherence(u[c], 7), 3) for c in range(4)]

def c(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
print("per-channel 'det' pairwise correlations (real):")
for i in range(4):
    for j in range(i+1, 4):
        print(f"  {POLS[i]}-{POLS[j]}: {c(d[i], d[j]):+.3f}", end="")
    print()

rng = np.random.default_rng(1)
pn = rng.uniform(0, np.pi, size=pha.shape)
un = np.exp(2j*pn)
dn = [_local_mean(_local_coherence(un[c], 7), 3) for c in range(4)]
print("\nnull control pairwise:")
print("  ", "  ".join(f"{POLS[i]}-{POLS[j]}: {c(dn[i], dn[j]):+.3f}"
                      for i in range(4) for j in range(i+1, 4)))

# point-target proxy: peak-to-local-mean ratio on the span
span = amp[0]**2 + 2*amp[1]**2 + amp[3]**2
ptl = span / (_local_mean(span.astype(np.float64), 9) + 1e-6)
det = _robust_norm(np.mean(d, axis=0))
print(f"\ncorr(det, peak-to-local-mean) {c(det, np.log(ptl+1e-6)):+.3f}")
top = ptl >= np.quantile(ptl, 0.999)     # strongest 0.1% point-like pixels
print(f"det on top-0.1% point-like: {det[top].mean():.3f} vs rest {det[~top].mean():.3f}")

# how much of det survives if we use a bigger window (more averaging)?
d11 = _robust_norm(np.mean([_local_mean(_local_coherence(u[c], 15), 3)
                            for c in range(4)], axis=0))
print(f"corr(det@win7, det@win15) {c(det, d11):+.3f}   "
      f"(null: {c(_robust_norm(np.mean(dn,axis=0)), _robust_norm(np.mean([_local_mean(_local_coherence(un[c],15),3) for c in range(4)],axis=0))):+.3f})")
