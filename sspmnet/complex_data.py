"""
Complex (SLC) auxiliary data support for SSPM-Net.

The base pipeline is amplitude-only. When the real / imaginary parts of the
single-look complex (SLC) data are also available, two extra physics facts
can be exploited (in the spirit of MERLIN, Dalsasso et al. 2021):

  1. RI Noise2Noise pair. For fully-developed speckle the complex signal
     z = Re + j*Im is circular Gaussian, so Re and Im are INDEPENDENT given
     the underlying reflectivity. Hence |Re| and |Im| (suitably scaled) are
     two independent noisy amplitude observations of the same clean signal —
     a natural Noise2Noise target pair inside every polarimetric channel.
     Supervising the masked pixels with both targets halves the effective
     target-noise power compared to the single amplitude target.

  2. Multi-look edge guide. Averaging the pseudo-intensities |Re|^2, |Im|^2
     over all 4 polarimetric channels yields an ~8-look "span" image whose
     speckle is strongly reduced. Its log-domain (ratio-detector) gradients
     form a far cleaner edge map than the 1-look amplitude, so the TV / NLM
     regularizers can smooth flat areas harder without eating real edges.

The interferometric phase of a single channel is uniformly distributed for
distributed targets (verified on the bundled patch) and carries no per-pixel
structural information, so it is not used directly; its information content
enters through Re / Im.

Calibration note: the bundled TIFFs are uint8 with an independent scale per
file, and the L1 losses are median-seeking, so |Re| / |Im| are rescaled per
channel onto the amplitude's L1 convention: the relative file scale is
estimated from the per-pixel sqrt(Re^2+Im^2)/amp ratio (exact up to
quantization) and multiplied by the theoretical Rayleigh / half-normal
median ratio (~1.746), aligning the PER-PIXEL conditional medians of the
amplitude target (Rayleigh) and the RI targets (half-normal) so mixing or
swapping them introduces no scale bias.
"""
import os
import glob

import numpy as np

POLS = ("hh", "hv", "vh", "vv")          # channel order [HH, HV, VH, VV]


def _read_tiff(path: str) -> np.ndarray:
    try:
        import tifffile
        arr = tifffile.imread(path)
    except ImportError:
        from PIL import Image
        arr = np.array(Image.open(path))
    return arr.astype(np.float32)


# Median of Rayleigh |z| over median of half-normal |Re| for circular
# Gaussian z (the sigma of the scene cancels): 1.1774 / 0.6745
_L1_RATIO = float(np.sqrt(2.0 * np.log(2.0)) / 0.6744897501960817)


def calibrate_ri(amp: np.ndarray, re_abs: np.ndarray, im_abs: np.ndarray,
                 mode: str = "l1") -> np.ndarray:
    """Rescale |Re| / |Im| per channel onto the amplitude scale.

    Parameters
    ----------
    amp, re_abs, im_abs : np.ndarray, shape (4, H, W)
        Amplitude and the absolute real / imaginary parts (any scale;
        each channel of each array may carry its own linear scale).
    mode : "l1" | "median" | "mean"
        "l1" (default) aligns the PER-PIXEL conditional medians — the right
        convention for the L1 losses: the relative file scale is estimated
        exactly from the per-pixel ratio sqrt(Re^2+Im^2)/amp (a constant up
        to quantization), then multiplied by the theoretical Rayleigh /
        half-normal median ratio (~1.746, scene-independent). "median" /
        "mean" align the global mixture statistic instead (approximate for
        heterogeneous scenes).

    Returns
    -------
    ri_pair : np.ndarray, shape (2, 4, H, W)
        ``ri_pair[0]`` and ``ri_pair[1]`` are the two independent
        pseudo-amplitude observations, on the amplitude scale.
    """
    ri = np.stack([re_abs, im_abs]).astype(np.float32)     # (2, 4, H, W)
    for c in range(amp.shape[0]):
        if mode == "l1":
            mag = np.sqrt(re_abs[c] ** 2 + im_abs[c] ** 2)
            ok = (amp[c] > 1e-6) & (mag > 1e-6)
            file_scale = float(np.median(mag[ok] / amp[c][ok])) if ok.any() else 1.0
            for k in range(2):
                ri[k, c] *= _L1_RATIO / max(file_scale, 1e-9)
        else:
            stat = np.median if mode == "median" else np.mean
            s_a = float(stat(amp[c]))
            for k in range(2):
                s_k = float(stat(ri[k, c]))
                ri[k, c] *= s_a / max(s_k, 1e-9)
    return ri


def load_quadpol_tiffs(tiff_dir: str, prefix: str = None,
                       calibrate: str = "l1"):
    """Load a quad-pol patch from per-component TIFF files.

    Expects files named ``{prefix}{pol}_{comp}.tiff`` with
    pol in {hh, hv, vh, vv} and comp in {amp, real, imgy} (``pha`` files,
    if present, are ignored — see the module docstring).

    Parameters
    ----------
    tiff_dir : str
        Directory containing the TIFFs.
    prefix : str or None
        Filename prefix up to the polarization tag. Auto-detected from the
        ``*hh_amp.tiff`` file when None.
    calibrate : "l1" | "median" | "mean" | None
        RI calibration mode (see :func:`calibrate_ri`); None skips it.

    Returns
    -------
    (amp, ri_pair) : np.ndarray (4, H, W), np.ndarray (2, 4, H, W)
        Amplitude stack [HH, HV, VH, VV] and the calibrated |Re| / |Im|
        pseudo-amplitude pair, ready for ``denoise(amp, ri_pair=ri_pair)``.
    """
    if prefix is None:
        cands = sorted(glob.glob(os.path.join(tiff_dir, "*hh_amp.tiff")))
        if not cands:
            raise FileNotFoundError(f"no '*hh_amp.tiff' found in {tiff_dir}")
        base = os.path.basename(cands[0])
        prefix = base[: base.index("hh_amp.tiff")]

    def load(comp):
        return np.stack([
            _read_tiff(os.path.join(tiff_dir, f"{prefix}{pol}_{comp}.tiff"))
            for pol in POLS
        ])

    amp = load("amp")
    re_abs = np.abs(load("real"))
    im_abs = np.abs(load("imgy"))

    if calibrate:
        ri_pair = calibrate_ri(amp, re_abs, im_abs, mode=calibrate)
    else:
        ri_pair = np.stack([re_abs, im_abs]).astype(np.float32)
    return amp, ri_pair
