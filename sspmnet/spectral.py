"""Spectral (complex-domain) speckle whitening — Track W.

Why this exists
---------------
The blind-spot / Noise2Noise self-supervision in the trainer assumes the
speckle of neighbouring pixels is independent.  On the bundled Gaofen-3 SLC
it is not: the complex spectrum fills only ~40-60% of the Nyquist band
(~2x oversampling with a Hamming-like weighting), so the lag-1 complex
coherence is ~0.7 and the lag-1 correlation of the normalised speckle
intensity is ~0.6.  About 27% (clipped uint8) to 55% (full range) of a
pixel's log-speckle is linearly predictable from its 8 neighbours, which a
blind-spot network can reproduce as "signal".  Only the phase makes this
fixable: with the complex SLC we can centre the spectrum (also restoring the
Re/Im independence MERLIN relies on), flatten the transfer function inside
the band and decimate to the band's own Nyquist rate.  Measured on the real
patch: lag-1 speckle correlation 0.60 -> 0.21, neighbour-predictability
0.27 -> 0.16, scene preserved (local-mean corr 0.984).

The same spectrum also yields *sub-looks*: two disjoint half-bands whose
speckle is independent (synthetic control corr 0.02) while the reflectivity
is shared — a genuinely independent Noise2Noise pair, unlike HV/VH whose
speckle is ~70% common.

Use the FULL-RANGE amplitude (``data/example_quadpol.npy`` + the ``pha``
TIFFs, see :func:`sspmnet.complex_data.load_quadpol_slc`): uint8 clipping
puts 28% of the power out of band, the unclipped SLC only 2.5%.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import uniform_filter, zoom as _nd_zoom
except ImportError:                                        # pragma: no cover
    uniform_filter = None
    _nd_zoom = None


def _smooth1d(p: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return p
    if uniform_filter is not None:
        return uniform_filter(p, k, mode="wrap")
    ker = np.ones(k) / k
    return np.convolve(np.concatenate([p[-k:], p, p[:k]]), ker, mode="same")[k:-k]


# ── spectrum centring ────────────────────────────────────────────────────

def spectrum_centroid(z: np.ndarray):
    """Power-spectrum centroid (cy, cx) in cycles/pixel of a 2-D complex image."""
    F = np.fft.fft2(z)
    P = np.abs(F) ** 2
    fy = np.fft.fftfreq(z.shape[0])
    fx = np.fft.fftfreq(z.shape[1])
    # use the circular mean so a centroid straddling +-0.5 does not average to 0
    py, px = P.sum(axis=1), P.sum(axis=0)
    cy = np.angle((py * np.exp(2j * np.pi * fy)).sum()) / (2 * np.pi)
    cx = np.angle((px * np.exp(2j * np.pi * fx)).sum()) / (2 * np.pi)
    return float(cy), float(cx)


def centre_spectrum(z: np.ndarray, centroid=None):
    """Demodulate a (4,H,W) or (H,W) complex stack so its spectrum is centred.

    A shifted spectrum (Doppler centroid / range offset) makes the Re and Im
    FIELDS correlated at lag 1 (~0.1 on this patch), which the MERLIN
    Re->Im Noise2Noise term treats as signal.  Returns (z_centred, centroid).
    """
    z = np.asarray(z)
    single = z.ndim == 2
    zs = z[None] if single else z
    if centroid is None:
        # one common centroid for all channels (same imaging system)
        cs = np.array([spectrum_centroid(c) for c in zs])
        centroid = (float(cs[:, 0].mean()), float(cs[:, 1].mean()))
    cy, cx = centroid
    yy, xx = np.mgrid[:zs.shape[-2], :zs.shape[-1]]
    ramp = np.exp(-2j * np.pi * (cy * yy + cx * xx)).astype(np.complex64)
    out = (zs * ramp[None]).astype(np.complex64)
    return (out[0] if single else out), centroid


# ── transfer-function estimate and whitening ─────────────────────────────

def estimate_transfer(z: np.ndarray, smooth: int = 9):
    """Separable 1-D power profiles (fftshifted, max 1) of a centred stack.

    Averaged over channels: the transfer function is a property of the
    sensor, not of the polarisation.
    """
    zs = z[None] if z.ndim == 2 else z
    P = np.zeros(zs.shape[-2:])
    for c in zs:
        P += np.abs(np.fft.fftshift(np.fft.fft2(c))) ** 2
    py = _smooth1d(P.mean(axis=1), smooth)
    px = _smooth1d(P.mean(axis=0), smooth)
    return py / py.max(), px / px.max()


def whiten(z: np.ndarray, flatten: bool = True, decim: int = 2,
           thr: float = 0.1, smooth: int = 9, centroid=None):
    """Spectrally whiten a (4,H,W) complex SLC stack.

    Steps: centre the spectrum, optionally flatten the in-band magnitude
    (separable 1/sqrt(P) weights where P > thr*max, zero outside), keep the
    central (H/decim, W/decim) frequency block (band-limited decimation —
    lossless when the band fits), inverse-transform, and rescale so the mean
    intensity of every channel is preserved.

    Returns (z_w, info); ``info`` carries what :func:`unwhiten_amp` and the
    sub-look split need.
    """
    z = np.asarray(z, dtype=np.complex64)
    single = z.ndim == 2
    zs = z[None] if single else z
    zc, centroid = centre_spectrum(zs, centroid)
    py, px = estimate_transfer(zc, smooth)
    if flatten:
        wy = np.where(py > thr, 1.0 / np.sqrt(np.maximum(py, 1e-12)), 0.0)
        wx = np.where(px > thr, 1.0 / np.sqrt(np.maximum(px, 1e-12)), 0.0)
    else:
        wy = np.ones_like(py)
        wx = np.ones_like(px)
    H, W = zs.shape[-2:]
    h, w = H // decim, W // decim
    out = np.empty((zs.shape[0], h, w), dtype=np.complex64)
    for k, c in enumerate(zc):
        F = np.fft.fftshift(np.fft.fft2(c)) * wy[:, None] * wx[None, :]
        Fc = F[H // 2 - h // 2: H // 2 - h // 2 + h,
               W // 2 - w // 2: W // 2 - w // 2 + w]
        y = np.fft.ifft2(np.fft.ifftshift(Fc))
        # preserve the channel's mean intensity (amplitude scale)
        s = np.sqrt((np.abs(c) ** 2).mean() / max((np.abs(y) ** 2).mean(), 1e-12))
        out[k] = y * s
    info = {"centroid": centroid, "py": py, "px": px, "thr": thr,
            "flatten": flatten, "decim": decim, "shape": (H, W)}
    return (out[0] if single else out), info


def unwhiten_amp(a: np.ndarray, info: dict) -> np.ndarray:
    """Bring an amplitude stack from the decimated grid back to the original
    (H, W) grid with cubic interpolation (the band-limited content is fully
    represented on the small grid; this is only resampling for display and
    same-grid metrics).
    """
    a = np.asarray(a, dtype=np.float64)
    H, W = info["shape"]
    single = a.ndim == 2
    As = a[None] if single else a
    if As.shape[-2:] == (H, W):
        return a
    fy, fx = H / As.shape[-2], W / As.shape[-1]
    out = np.stack([_nd_zoom(c, (fy, fx), order=3) for c in As])
    out = np.clip(out, 0.0, None)[:, :H, :W]
    return out[0] if single else out


# ── sub-looks ────────────────────────────────────────────────────────────

def sublooks(z: np.ndarray, axis: int = -1, centroid=None):
    """Split a centred complex stack into two half-band images along ``axis``.

    The two halves have disjoint spectral support, hence independent speckle
    for distributed targets, and share the reflectivity (at half the
    bandwidth along that axis).  Returns (A, B) on the input grid.
    """
    z = np.asarray(z, dtype=np.complex64)
    zc, _ = centre_spectrum(z, centroid)
    ax = axis % zc.ndim
    F = np.fft.fft(zc, axis=ax)
    f = np.fft.fftfreq(zc.shape[ax])
    shp = [1] * zc.ndim
    shp[ax] = -1
    f = f.reshape(shp)
    A = np.fft.ifft(F * (f < 0), axis=ax) * np.sqrt(2)
    B = np.fft.ifft(F * (f >= 0), axis=ax) * np.sqrt(2)
    return A.astype(np.complex64), B.astype(np.complex64)


# ── diagnostics ──────────────────────────────────────────────────────────

def speckle_whiteness(intensity: np.ndarray, win: int = 15) -> dict:
    """How white is the speckle of a 2-D intensity image?

    lag1_x / lag1_y : correlation of the locally normalised intensity with
        its right / lower neighbour (white speckle: 0).
    neigh_r2 : fraction of the pixel's log-normalised intensity linearly
        predictable from its 8 neighbours (what a blind-spot net can copy).
    """
    I = np.asarray(intensity, dtype=np.float64)
    m = uniform_filter(I, win, mode="nearest") if uniform_filter is not None else I.mean()
    z = I / np.maximum(m, 1e-9)
    lag1_x = float(np.corrcoef(z[:, :-1].ravel(), z[:, 1:].ravel())[0, 1])
    lag1_y = float(np.corrcoef(z[:-1].ravel(), z[1:].ravel())[0, 1])
    lz = np.log(z + 1e-6)
    b = win // 2
    lz = lz[b:-b, b:-b]
    Hh, Ww = lz.shape
    t = lz[1:-1, 1:-1].ravel()
    cols = [lz[1 + dy:Hh - 1 + dy, 1 + dx:Ww - 1 + dx].ravel()
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    X = np.stack(cols + [np.ones(t.size)], axis=1)
    beta, *_ = np.linalg.lstsq(X, t, rcond=None)
    r2 = 1.0 - ((t - X @ beta) ** 2).mean() / max(t.var(), 1e-12)
    return {"lag1_x": lag1_x, "lag1_y": lag1_y, "neigh_r2": float(r2)}


def reim_leak(z: np.ndarray) -> float:
    """corr(Re(x), Im(x + 1 row)) — the MERLIN independence leak (0 = none)."""
    re, im = z.real, z.imag
    return float(np.corrcoef(re[:-1].ravel(), im[1:].ravel())[0, 1])
