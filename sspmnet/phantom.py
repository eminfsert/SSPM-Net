"""Synthetic quad-pol reflectivity phantom with KNOWN ground truth.

Why this exists
---------------
Every synthetic-GT number reported up to 2026-09-05 used a "clean proxy" that
was itself an output of this pipeline:

    clean = denoise(amp, TrainConfig(iters=700))["denoised"]

so the protocol asked how well a variant recovers a reference our own baseline
produced.  That is circular in a way that matters: the proxy carries only ~46%
of the input's fine texture, so resembling it rewards smoothing, and a variant
whose texture happens to match the proxy's scores well for the wrong reason.

This module builds a reflectivity field from scratch — no network, no real
data — so PSNR/SSIM/EPI against it measure recovery of a truly known signal.
It is deliberately *not* a copy of the real scene's statistics; use it together
with a classical-estimator proxy (see `scripts/experiments/run_indep_eval.py`)
rather than instead of it.

Content: piecewise-constant regions with sharp edges (field/parcel analogue),
a dark flat band (water analogue, for ENL and grain), textured blocks (urban
analogue, for EPI), thin linear features (roads/bridges) and isolated point
targets (the bright tail that uint8 clipping destroys).  Channel structure
follows the real physics used elsewhere in the repo: HV and VH are the same
physical channel (reciprocity) and sit well below the co-pol channels.
"""
import numpy as np


def _smooth(x, k):
    """Separable box blur, k odd."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(x, k)


def make_phantom(shape=(512, 512), seed=0, q99=None, return_masks=False):
    """Build a (4, H, W) float32 reflectivity (amplitude) ground truth.

    Parameters
    ----------
    shape : (H, W)
    seed : int
    q99 : float or None
        If given, each channel is scaled so its 99th percentile matches this
        value — use the real patch's q99 so the pipeline's internal
        normalization (per-channel q99, clip(0, 5)) sees a familiar range.
    return_masks : bool
        Also return a dict of region masks known BY CONSTRUCTION: "flat" (the
        dark homogeneous band, for ENL / grain), "urban" (textured blocks, for
        EPI) and "point" (the isolated bright targets). Deriving these from the
        phantom values instead would be fragile — the flat band is exactly
        constant, so a percentile threshold with a strict `<` selects nothing.

    Returns
    -------
    (4, H, W) float32 — or (phantom, masks) when ``return_masks``.
    """
    H, W = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # ── 1. piecewise-constant regions: quantize a smooth random field ──
    lo = _smooth(rng.normal(0, 1, (H, W)), 41)
    lo = (lo - lo.min()) / max(np.ptp(lo), 1e-9)
    region = np.floor(lo * 7).astype(int)                  # 8 levels
    base = 0.25 + 0.75 * (region / 7.0) ** 1.6             # region reflectivity

    # ── 2. dark flat band (water): a wide diagonal stripe ──
    band = (0.55 * xx + yy) / (0.55 * W + H)
    water = (band > 0.30) & (band < 0.40)
    base[water] = 0.035

    # ── 3. textured urban blocks: fine high-contrast structure ──
    urban = np.zeros((H, W), bool)
    for _ in range(6):
        by, bx = rng.integers(0, H - 110), rng.integers(0, W - 110)
        bh, bw = rng.integers(70, 110), rng.integers(70, 110)
        sl = (slice(by, by + bh), slice(bx, bx + bw))
        if water[sl].mean() > 0.05:
            continue
        urban[sl] = True
        # building rows: alternating bright/dark at a few-pixel pitch
        p = int(rng.integers(4, 8))
        rows = ((yy[sl] - by) % p) < (p // 2)
        cols = ((xx[sl] - bx) % (p + 2)) < ((p + 2) // 2)
        base[sl] = np.where(rows & cols, 1.9, 0.30)

    # ── 4. thin linear features (roads: dark; bridges: bright) ──
    for _ in range(5):
        a = rng.uniform(0, np.pi)
        c = rng.uniform(-1, 1) * 0.45 * W
        d = np.abs(np.cos(a) * (xx - W / 2) + np.sin(a) * (yy - H / 2) - c)
        base[d < 1.2] = 0.06 if rng.random() < 0.6 else 1.4

    # ── 5. isolated point targets: the bright tail uint8 clipping kills ──
    n_pt = 220
    py = rng.integers(2, H - 2, n_pt)
    px = rng.integers(2, W - 2, n_pt)
    base[py, px] = rng.uniform(6.0, 30.0, n_pt)

    # ── 6. per-channel structure ──
    # co-pol carry the full field; cross-pol is the same physical channel
    # (reciprocity) at a much lower level, and volume/urban areas raise it.
    g_hh = 1.0
    g_vv = 0.85 + 0.30 * _smooth(rng.normal(0, 1, (H, W)), 61)   # mild decorrelation
    g_vv = np.clip(g_vv, 0.5, 1.5)
    xpol_gain = 0.16 + 0.34 * urban.astype(np.float64)           # xpol up on structures
    hh = base * g_hh
    vv = base * g_vv
    xp = base * xpol_gain
    out = np.stack([hh, xp, xp, vv]).astype(np.float64)          # HV == VH exactly
    out = np.maximum(out, 1e-3)

    if q99 is not None:
        for c in range(4):
            out[c] *= q99 / max(np.percentile(out[c], 99), 1e-9)
    out = out.astype(np.float32)
    if return_masks:
        pts = np.zeros((H, W), bool)
        pts[py, px] = True
        # erode the flat band so no edge pixel leaks into the grain statistic
        flat = _smooth(water.astype(np.float64), 5) > 0.999
        return out, {"flat": flat, "urban": urban, "point": pts}
    return out
