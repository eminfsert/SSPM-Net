"""Visual comparison for Track E1: the amplitude-only thesis baseline vs the
current full stack vs the censored (E1) variant, on the real patch.

Four columns (noisy input, amplitude-only baseline, current base, E1) by four
rows: the full HV channel, an urban zoom, the archived flat-water crop (the
grain check from the Track C lesson) and the densest uint8-CLIPPED bright block
(what E1 is specifically about).  Per-column metrics are printed in the titles.

IMPORTANT — every panel is LEAST-SQUARES SCALE-MATCHED onto the noisy input
before display.  The three outputs do NOT sit on the same radiometric scale
(HV mean: noisy 81.8, amplitude-only baseline 52.9, current stack 72.7, E1
75.0; LS scale vs noisy 1.47 / 1.16 / 1.11), because MERLIN-style training
lands on the median convention and the amplitude-only baseline lands lower
still.  Without matching, the baseline merely looks darker and the flat-water
and clipped-block panels would be comparing brightness, not denoising.  The
metrics in the titles are scale-invariant by construction (EPI, ratio-ENL, CV);
waterHP and the block mean are reported AFTER matching so they are comparable.

Writes docs/figures/compare_track_e_baseline.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/content/SSPM-Net")
os.chdir("/content/SSPM-Net")
from sspmnet import load_quadpol_tiffs
from sspmnet.phase_data import _local_mean
from sspmnet.metrics import find_top_k_rois, enl_roi_multi, epi_metric

OUT = "results/ri_compare"
amp, ri, sat = load_quadpol_tiffs("data/tiff", return_sat=True)
ampf = amp.astype(np.float64)

COLS = [
    ("Noisy input",              None),
    ("Baseline (amplitude-only)", "denoised_baseline"),
    ("Current stack",            "trke_real_base"),
    ("+ E1 censored (tv.5)",     "trke_real_cens_tv05"),
]

# ── masks / crops ──
span8 = np.sqrt((ri.astype(np.float64) ** 2).mean(axis=(0, 1)))
mu = _local_mean(span8, 21)
cv = np.sqrt(np.maximum(_local_mean(span8 ** 2, 21) - mu ** 2, 0)) / (mu + 1e-6)
x2s = _local_mean(0.5 * (ri[:, 1] + ri[:, 2]).mean(axis=0).astype(np.float64), 21)
water = _local_mean(((x2s < np.percentile(x2s, 20)) & (cv < 0.40)
                     ).astype(np.float64), 5) > 0.999
rois, rs = find_top_k_rois(amp[1])
sat_hv = sat[1]

zy, zx, zs = 180, 260, 160                     # urban zoom
wy, wx, ws = 180, 408, 32                      # archived flat-water crop
dens = _local_mean(sat_hv.astype(np.float64), 33)
by, bx = np.unravel_index(np.argmax(dens), dens.shape)
bs = 96
by = int(np.clip(by - bs // 2, 0, amp.shape[1] - bs))
bx = int(np.clip(bx - bs // 2, 0, amp.shape[2] - bs))


def ratio_enl(d, c):
    rI = (ampf[c] ** 2 + 1e-3) / (d[c] ** 2 + 1e-3)
    v = rI[(d[c] > 2) & (amp[c] > 0)]
    return (v.mean() / v.std()) ** 2


def scale_to_noisy(d):
    """Least-squares scale onto the noisy input, per channel (display only)."""
    o = d.copy()
    for c in range(d.shape[0]):
        o[c] = d[c] * float((d[c] * ampf[c]).sum() / max((d[c] ** 2).sum(), 1e-9))
    return o


imgs = []
for name, tag in COLS:
    if tag is None:
        imgs.append((name, "1-look input", ampf))
        continue
    raw = np.load(f"{OUT}/{tag}.npy").astype(np.float64)
    # scale-invariant metrics on the RAW output; display + level metrics matched
    epi = epi_metric(ampf[1], raw[1])
    renl = ratio_enl(raw, 1)
    d = scale_to_noisy(raw)
    # satRatio is scale-SENSITIVE, so it must be measured after matching too:
    # on the raw outputs the baseline scores 0.553 purely because it sits 1.47x
    # lower overall, not because it flattens the bright tail more.
    sr = float(np.median(d[1][sat_hv] / np.maximum(amp[1][sat_hv], 1.0)))
    hp = (d[1] - _local_mean(d[1], 5))[water].std()
    sub = (f"EPI(HV) {epi:.3f}   ENLr(HV) {renl:.3f}   satRatio {sr:.3f} (matched)\n"
           f"waterHP {hp:.3f} (scale-matched)   LS scale "
           f"{float((raw[1] * ampf[1]).sum() / max((raw[1] ** 2).sum(), 1e-9)):.3f}")
    imgs.append((name, sub, d))

vmax = np.quantile(amp[1], 0.99)
wmax = 3.0 * np.median(imgs[2][2][1][wy:wy + ws, wx:wx + ws])
bmax = float(np.quantile(amp[1][by:by + bs, bx:bx + bs], 0.99))

fig, axes = plt.subplots(4, 4, figsize=(15.5, 16.6))
for col, (name, sub, d) in enumerate(imgs):
    v = np.clip(d[1] / vmax, 0, 1)
    axes[0, col].imshow(v, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"{name}\n{sub}", fontsize=9.5)

    axes[1, col].imshow(v[zy:zy + zs, zx:zx + zs], cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
    axes[1, col].set_title("urban zoom — HV", fontsize=8.5)

    wc = d[1][wy:wy + ws, wx:wx + ws]
    axes[2, col].imshow(wc, cmap="gray", vmin=0, vmax=wmax, interpolation="nearest")
    axes[2, col].set_title(f"flat water — std {wc.std():.2f}  "
                           f"CV {wc.std() / max(wc.mean(), 1e-9):.3f}", fontsize=8.5)

    bc = d[1][by:by + bs, bx:bx + bs]
    sm = sat_hv[by:by + bs, bx:bx + bs]
    axes[3, col].imshow(bc, cmap="gray", vmin=0, vmax=bmax, interpolation="nearest")
    axes[3, col].set_title(f"uint8-CLIPPED block — mean {bc.mean():.1f}  "
                           f"(clipped px {100 * sm.mean():.0f}%)", fontsize=8.5)

for ax in axes.ravel():
    ax.axis("off")
fig.suptitle("SSPM-Net — HV channel, real patch: amplitude-only baseline vs the "
             "current stack vs Track E1 (censored loss on clipped targets)\n"
             "all panels least-squares scale-matched onto the noisy input — the "
             "three outputs sit on different radiometric scales",
             fontsize=11.5, y=0.998)
fig.tight_layout(rect=[0, 0, 1, 0.985])
os.makedirs("docs/figures", exist_ok=True)
fig.savefig("docs/figures/compare_track_e_baseline.png", dpi=135)
print("wrote docs/figures/compare_track_e_baseline.png")
for name, sub, _ in imgs:
    print(f"  {name}: {sub}".replace("\n", " | "))
