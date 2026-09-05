"""Independent re-evaluation: are the headline wins real, or protocol artefacts?

Every synthetic-GT number in this repo up to 2026-09-05 was measured against a
"clean proxy" that was itself an output of this pipeline
(`denoise(amp, TrainConfig(iters=700))`).  The proxy carries only ~46% of the
input's fine texture, so resembling it rewards smoothing — and a variant whose
texture happens to match the proxy's can score well for the wrong reason.  That
makes every "+0.x dB" conclusion suspect, including Track E1's.

This script re-scores the three headline claims against a ground truth that
owes nothing to the network (`sspmnet.phantom.make_phantom`):

    A1/A2  pixel+group          claimed GT PSNR +0.64/+1.51 dB (the big win)
    C1     xpol_pair_input      claimed GT PSNR(HV) +0.13 dB
    E1     sat_censored         claimed +0.85/+0.63 dB of the clipping cost

Rows:
    baseline        amplitude-only TrainConfig(iters=700)  (thesis baseline)
    stack-oldbase   full stack, historical model (band dropout + batchnorm)
    stack-pg        full stack, pixel+group                (tests A1/A2)
    stack-pairin    + xpol_pair_input                      (tests C1; = today)
    clip-pairin     today's base on the CLIPPED phantom
    clip-E1         + sat_censored/sat_tv_relax=0.5        (tests E1)

The speckle/thermal simulation matches the real patch's HV-VH SINGLE-angle
complex coherence (0.772, using the corrected [0,2pi) phase decoding), and the
uint8 clipping matches the real per-channel saturation fractions.

Usage:  python scripts/experiments/run_indep_eval.py
"""
import os
import sys
import glob

import numpy as np

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")
from sspmnet import denoise, TrainConfig, load_quadpol_tiffs
from sspmnet.complex_data import calibrate_ri, _read_tiff
from sspmnet.phantom import make_phantom
from sspmnet.phase_data import phase_feedback_maps, _local_mean
from sspmnet.metrics import find_top_k_rois, enl_roi_multi, epi_metric, ssim_metric

OUT = "results/indep_eval"
os.makedirs(OUT, exist_ok=True)

amp, ri = load_quadpol_tiffs("data/tiff")
q99_real = float(np.percentile(amp[1], 99))

# ── ground truth: built from scratch, no network, no real pixels ──
clean, masks = make_phantom(shape=amp.shape[1:], seed=0, q99=q99_real,
                            return_masks=True)
clean = clean.astype(np.float64)
np.save(f"{OUT}/phantom_clean.npy", clean.astype(np.float32))


def cached(fname, fn):
    f = os.path.join(OUT, fname)
    if os.path.exists(f):
        print(f"[cached] {fname}", flush=True)
        return np.load(f)
    out = fn()
    np.save(f, out)
    return out


# ── speckle + thermal simulation (reciprocity: HV and VH share the speckle) ──
def _coh(x, y, w=7):
    from scipy.ndimage import uniform_filter
    n = (uniform_filter(np.real(x * np.conj(y)), w)
         + 1j * uniform_filter(np.imag(x * np.conj(y)), w))
    d = np.sqrt(uniform_filter(np.abs(x) ** 2, w) * uniform_filter(np.abs(y) ** 2, w))
    return float((np.abs(n) / np.maximum(d, 1e-9)).mean())


# real target: SINGLE-angle HV-VH coherence with the corrected decoding
pha_u8 = np.stack([_read_tiff(f) for f in
                   sorted(glob.glob("data/tiff/*_pha.tiff"))])          # hh,hv,vh,vv
S_real = amp.astype(np.float64) * np.exp(1j * pha_u8 / 255.0 * 2 * np.pi)
off = np.angle((S_real[1] * np.conj(S_real[2])).sum())
target_coh = _coh(S_real[1], S_real[2] * np.exp(1j * off))
print(f"real HV-VH single-angle coherence (offset {np.degrees(off):+.1f} deg) "
      f"= {target_coh:.4f}", flush=True)


def simulate(sigma_n, seed=7):
    rng = np.random.default_rng(seed)
    def cn(sig):
        return (rng.normal(0, sig / np.sqrt(2), clean[0].shape)
                + 1j * rng.normal(0, sig / np.sqrt(2), clean[0].shape))
    g = np.stack([cn(1.0) for _ in range(4)])
    g[2] = g[1]                                   # reciprocity: shared speckle
    n = np.stack([cn(sigma_n) for _ in range(4)])
    return clean * g + n


rms_x = float(np.sqrt((clean[1] ** 2).mean()))
lo, hi = 0.02 * rms_x, 3.0 * rms_x
for _ in range(14):
    mid = 0.5 * (lo + hi)
    z = simulate(mid)
    lo, hi = (mid, hi) if _coh(z[1], z[2]) > target_coh else (lo, mid)
sigma_n = 0.5 * (lo + hi)
z = simulate(sigma_n)
print(f"sigma_n={sigma_n:.3f}  simulated coherence={_coh(z[1], z[2]):.4f}", flush=True)

amp_sim = np.abs(z).astype(np.float32)
ri_sim = calibrate_ri(amp_sim, np.abs(z.real).astype(np.float32),
                      np.abs(z.imag).astype(np.float32), mode="l1")
maps_sim = phase_feedback_maps(z=z)

# ── uint8 clipping matched to the real per-channel saturation fractions ──
def u8_clip(x, frac):
    thr = np.quantile(x, 1.0 - frac)
    return np.clip(np.round(x * (255.0 / max(thr, 1e-9))), 0, 255).astype(np.float32)


def real_frac(comp):
    return [float((_read_tiff(f) >= 255).mean())
            for f in sorted(glob.glob(f"data/tiff/*_{comp}.tiff"))]


fa, fr, fi = (real_frac(c) for c in ("amp", "real", "imgy"))
amp_c = np.stack([u8_clip(amp_sim[c], fa[c]) for c in range(4)])
re_c = np.stack([u8_clip(np.abs(z.real[c]), fr[c]) for c in range(4)]).astype(np.float32)
im_c = np.stack([u8_clip(np.abs(z.imag[c]), fi[c]) for c in range(4)]).astype(np.float32)
sat_c = (amp_c >= 255.0) | (re_c >= 255.0) | (im_c >= 255.0)
ri_c = calibrate_ri(amp_c, re_c, im_c, mode="l1")
print(f"clipped fraction/ch: {(sat_c.mean(axis=(1, 2)) * 100).round(2)}", flush=True)

# flat / urban regions of the PHANTOM — known BY CONSTRUCTION, not thresholded
# (the flat band is exactly constant, so a percentile threshold selects nothing)
water = masks["flat"]
urban = masks["urban"]
print(f"phantom flat mask: {int(water.sum())} px, urban {int(urban.sum())} px",
      flush=True)

STACK = dict(iters=700, ri_mode="merlin", merlin_loss="l1", tv_mult=10.0,
             guide_cv_protect=0.3, phase_smooth_boost=3.0, phase_fidelity=0.5,
             whiteness_lambda=0.05, whiteness_lags=(1, 2, 3),
             polish=0.5, edge_boost=1.0)
OLD = {"dropout_style": "band", "norm": "batch"}
PG = {"dropout_style": "pixel", "norm": "group"}
PAIR = {**PG, "xpol_pair_input": True}

# (name, tag, TrainConfig kwargs, model_cfg, clipped?, sat?)
RUNS = [
    ("baseline",      "ie_baseline",  dict(iters=700), None,  False, False),
    ("stack-oldbase", "ie_oldbase",   STACK,           OLD,   False, False),
    ("stack-pg",      "ie_pg",        STACK,           PG,    False, False),
    ("stack-pairin",  "ie_pairin",    STACK,           PAIR,  False, False),
    ("clip-pairin",   "ie_clip",      STACK,           PAIR,  True,  False),
    ("clip-E1",       "ie_clip_e1",   {**STACK, "sat_censored": True,
                                       "sat_tv_relax": 0.5},  PAIR, True, True),
]

runs = {}
for name, tag, kw, mcfg, clipped, use_sat in RUNS:
    print(f"\n=== {name} ===", flush=True)
    a_in = amp_c if clipped else amp_sim
    r_in = ri_c if clipped else ri_sim
    s_in = sat_c if use_sat else None
    if name == "baseline":                     # amplitude-only: no RI, no phase
        runs[name] = cached(tag + ".npy", lambda: denoise(
            a_in, TrainConfig(**kw))["denoised"]).astype(np.float64)
    else:
        runs[name] = cached(tag + ".npy", lambda: denoise(
            a_in, TrainConfig(**kw, model_cfg=mcfg), ri_pair=r_in,
            pha=maps_sim, sat=s_in)["denoised"]).astype(np.float64)


# ── metrics, all against the KNOWN phantom ──
def scale_match(x, ref):
    o = x.copy()
    for c in range(x.shape[0]):
        o[c] = x[c] * float((x[c] * ref[c]).sum() / max((x[c] ** 2).sum(), 1e-9))
    return o


def psnr(x, ref):
    return 10 * np.log10(ref.max() ** 2 / max(((x - ref) ** 2).mean(), 1e-12))


thr99 = np.quantile(clean, 0.99)
bright = clean >= thr99
rois, rs = find_top_k_rois(clean[1])

lines = []
W = 11
cols = ["PSNR(HH)", "PSNR(HV)", "SSIM(HH)", "SSIM(HV)", "EPI(HH)", "EPI(HV)",
        "ENL(HV)", "EPIurb(HV)", "RMSEbrgt", "brightR", "flatHP"]
title = ("INDEPENDENT evaluation vs a from-scratch phantom ground truth "
         "(sspmnet/phantom.py) — no network, no real pixels in the reference. "
         "All outputs LS scale-matched. flatHP = high-pass std of HV over the "
         f"phantom's flat band (clean {float((clean[1] - _local_mean(clean[1], 5))[water].std()):.4f})")
hdr = "  {:<15}".format("Method") + "".join(f"{c:>{W}}" for c in cols)
print("\n" + title)
print(hdr)
lines += [title, hdr]
for name, d in runs.items():
    ds = scale_match(d, clean)
    hp = float((ds[1] - _local_mean(ds[1], 5))[water].std())
    vals = [psnr(ds[0], clean[0]), psnr(ds[1], clean[1]),
            ssim_metric(clean[0], ds[0]), ssim_metric(clean[1], ds[1]),
            epi_metric(clean[0], ds[0]), epi_metric(clean[1], ds[1]),
            enl_roi_multi(d[1], rois, rs),
            epi_metric(clean[1] * urban, ds[1] * urban),
            float(np.sqrt(((ds - clean) ** 2)[bright].mean())),
            float((ds[bright] / np.maximum(clean[bright], 1e-9)).mean()), hp]
    line = "  {:<15}".format(name) + "".join(f"{v:>{W}.4f}" for v in vals)
    print(line, flush=True)
    lines.append(line)


def delta(a, b, label):
    da = scale_match(runs[a], clean)
    db = scale_match(runs[b], clean)
    s = (f"  {label:<34} dPSNR(HH) {psnr(db[0], clean[0]) - psnr(da[0], clean[0]):+.3f}  "
         f"dPSNR(HV) {psnr(db[1], clean[1]) - psnr(da[1], clean[1]):+.3f}  "
         f"dEPI(HV) {epi_metric(clean[1], db[1]) - epi_metric(clean[1], da[1]):+.4f}")
    print(s)
    lines.append(s)


print("\nRe-test of the headline claims (independent GT):")
lines.append("")
lines.append("Re-test of the headline claims (independent GT):")
delta("stack-oldbase", "stack-pg",   "A1/A2 pixel+group (claimed +0.64/+1.51)")
delta("stack-pg", "stack-pairin",    "C1 xpol_pair_input (claimed +0.13 HV)")
delta("clip-pairin", "clip-E1",      "E1 censored (claimed +0.85/+0.63)")
delta("baseline", "stack-pairin",    "baseline -> today's full stack")

with open(f"{OUT}/metrics_indep_eval.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}/metrics_indep_eval.txt")
