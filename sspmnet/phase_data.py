"""
Quad-pol phase feedback maps for SSPM-Net.

The bundled uint8 phase TIFFs are folded to [0, pi] (sign lost), so the raw
phase value of a single channel is unusable. What survives the folding is
the DOUBLED angle 2*phi (exact under a mod-pi fold, and coherence-preserving
under an abs fold or a half-scale encoding), from which three per-pixel
feedback maps are built with local circular statistics:

  snr (HV-VH reciprocity coherence)
      Monostatic reciprocity gives S_hv = S_vh: the two cross-pol channels
      share the SAME complex speckle and differ only by additive
      thermal/system noise. The local concentration of
      exp(2j*phi_hv) * conj(exp(2j*phi_vh)) therefore estimates the
      per-pixel signal-to-noise ratio: ~1 where the observed value is
      signal (speckle included), ~floor where it is instrument noise.
      This is the "is this pixel's value noise?" feedback: low snr pixels
      (dark cross-pol: roads, water, shadow) may be regularized hard and
      their data term trusted less.

  surface (HH-VV co-pol coherence)
      High co-pol coherence indicates surface (Bragg-like) scattering --
      distributed, homogeneous targets that tolerate strong smoothing;
      volume / complex scattering decorrelates HH and VV.

  det (single-channel spatial phase coherence)
      For fully-developed speckle the phase field is spatially white (up
      to the oversampling correlation, removed by the robust
      normalization); excess LOCAL spatial coherence marks deterministic
      / point scatterers whose edges must be protected.

  helix (co-pol / cross-pol phase coherence)
      For reflection-symmetric (natural, distributed) media the co-pol and
      cross-pol channels are uncorrelated, so the local concentration of
      exp(2j*(phi_HH - phi_HV)) sits at the estimator's random-phase floor
      (~0.13 for a 7x7 window). Man-made / helical scatterers break the
      symmetry and light the map up (0.34 on the bright 10% of the real
      patch) — a cross-pol-specific structure detector, stronger than
      ``det`` (which is near its noise floor on this data).

Each map is normalized per scene to [0, 1] with robust percentiles, which
also removes the estimator's noise floor and the oversampling baseline.
"""
import os
import glob

import numpy as np

from .complex_data import POLS, _read_tiff

try:
    from scipy.ndimage import uniform_filter
except ImportError:                                        # pragma: no cover
    uniform_filter = None


def load_quadpol_phase(tiff_dir: str, prefix: str = None) -> np.ndarray:
    """Load the quad-pol phase stack, decoded to the full range [0, 2*pi).

    (Until 2026-09-05 this returned ``u8*pi/255``; the files are full-range
    SLC phase, and ``phase_feedback_maps`` used to double the angle, so the
    maps are numerically unchanged — but ``z=`` callers now get the same
    single-angle statistics as ``pha=`` callers.)

    Returns (4, H, W) float32 in channel order [HH, HV, VH, VV].
    """
    if prefix is None:
        cands = sorted(glob.glob(os.path.join(tiff_dir, "*hh_pha.tiff")))
        if not cands:
            raise FileNotFoundError(f"no '*hh_pha.tiff' found in {tiff_dir}")
        base = os.path.basename(cands[0])
        prefix = base[: base.index("hh_pha.tiff")]
    pha = np.stack([
        _read_tiff(os.path.join(tiff_dir, f"{prefix}{pol}_pha.tiff"))
        for pol in POLS
    ])
    return (pha * (2.0 * np.pi / 255.0)).astype(np.float32)


def _local_mean(x: np.ndarray, win: int) -> np.ndarray:
    if uniform_filter is not None:
        return uniform_filter(x, win, mode="nearest")
    # separable box fallback
    k = np.ones(win, dtype=np.float64) / win
    y = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 0, x)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, y)


def _local_coherence(z: np.ndarray, win: int) -> np.ndarray:
    """|local mean| of a unit-phasor field: circular concentration in [0,1]."""
    return np.abs(_local_mean(z.real, win) + 1j * _local_mean(z.imag, win))


def _robust_norm(g: np.ndarray, q_lo: float = 0.2, q_hi: float = 0.95):
    lo, hi = np.quantile(g, q_lo), np.quantile(g, q_hi)
    return np.clip((g - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def phase_feedback_maps(pha: np.ndarray = None, win: int = 7,
                        z: np.ndarray = None, smooth: int = 3) -> dict:
    """Per-pixel feedback maps from folded phase (or true complex data).

    Parameters
    ----------
    pha : (4, H, W) full-range phase (from :func:`load_quadpol_phase`).
    win : odd int — local circular-statistics window.
    smooth : int — box size for lightly smoothing the coherence maps
        before normalization (the win-sized estimator is itself noisy);
        0/1 disables.
    z : (4, H, W) complex, optional — maps are computed from angle(z);
        identical to passing ``pha=np.angle(z)`` (single angle, no doubling).

    Returns
    -------
    dict of (H, W) float32 maps in [0, 1]:
        'snr'     — HV-VH reciprocity coherence (high = signal-dominated)
        'surface' — HH-VV co-pol coherence      (high = surface scattering)
        'det'     — spatial phase coherence     (high = deterministic target)
        'helix'   — co/cross-pol phase coherence (high = symmetry-breaking,
                    man-made structure; cross-pol structure protection)
    """
    if pha is None:
        if z is None:
            raise ValueError("give either pha or z")
        pha = np.angle(z)
    u = np.exp(1j * pha.astype(np.float64))

    snr = _local_coherence(u[1] * np.conj(u[2]), win)
    surface = _local_coherence(u[0] * np.conj(u[3]), win)
    det = np.mean([_local_coherence(u[c], win) for c in range(4)], axis=0)
    helix = 0.5 * (_local_coherence(u[0] * np.conj(u[1]), win)
                   + _local_coherence(u[0] * np.conj(u[2]), win))

    if smooth and smooth > 1:
        snr, surface, det, helix = (
            _local_mean(g, smooth) for g in (snr, surface, det, helix))
    return {"snr": _robust_norm(snr),
            "surface": _robust_norm(surface),
            "det": _robust_norm(det),
            # the random-phase floor is well above the 20th percentile on
            # a mostly-natural scene, so the robust norm removes it
            "helix": _robust_norm(helix, q_lo=0.5, q_hi=0.99)}
