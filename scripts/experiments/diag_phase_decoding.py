"""Phase decoding audit: the bundled uint8 `*_pha.tiff` files are FULL-RANGE
[0, 2*pi) SLC phase, not folded to [0, pi].

Sessions up to 2026-09-02 decoded the phase as `pha/255*pi` (a [0, pi] fold),
which correlates ~0.02 with the quadrant angle implied by |Re|,|Im| -- and led
to the conclusion that the phase is "inconsistent with the components" and can
only be used as a spatial map (doubled-angle 2*phi). That conclusion was an
artefact of the wrong scale factor.

Decoding phi = pha/255 * 2*pi reproduces the components at corr 0.996 and the
four phase quadrants are uniformly occupied (~25% each), i.e. the sign
information is present. The complex SLC is therefore recoverable as

    S = amp * exp(1j * phi)                      (convention: |Re| ~ |sin phi|,
                                                  |Im| ~ |cos phi|; the global
                                                  90-degree rotation is the same
                                                  for all four channels and
                                                  cancels in every coherence)

Run:  python scripts/experiments/diag_phase_decoding.py
"""
import os
import numpy as np
from PIL import Image

try:
    from scipy.ndimage import uniform_filter
except ImportError:                                        # pragma: no cover
    raise SystemExit("scipy is required for this diagnostic")

TIFF_DIR = os.environ.get("SSPM_TIFF_DIR", "data/tiff")
PREFIX = os.environ.get("SSPM_PREFIX", "1_patch_3584_0_512_")
POLS = ("hh", "hv", "vh", "vv")


def _read(pol, comp):
    return np.array(Image.open(
        os.path.join(TIFF_DIR, f"{PREFIX}{pol}_{comp}.tiff"))).astype(np.float64)


def slc(pol):
    """Complex SLC recovered from the amplitude and the full-range phase."""
    return _read(pol, "amp") * np.exp(1j * _read(pol, "pha") / 255.0 * 2 * np.pi)


def align(x, y):
    """Remove the constant inter-channel phase offset before combining.

    HV and VH carry a fixed +140.25 deg relative phase on this product (the
    per-file uint8 phase encoding and/or the system calibration). Combining
    them without removing it is destructive interference, not fusion.
    """
    off = np.angle((x * np.conj(y)).sum())
    return y * np.exp(1j * off), off


def coherence(x, y, win=7):
    num = (uniform_filter(np.real(x * np.conj(y)), win)
           + 1j * uniform_filter(np.imag(x * np.conj(y)), win))
    den = np.sqrt(uniform_filter(np.abs(x) ** 2, win)
                  * uniform_filter(np.abs(y) ** 2, win))
    return np.abs(num) / np.maximum(den, 1e-9)


def enl(intensity, win=9):
    """Median-CV ENL over the mid-brightness (p20..p80) pixels."""
    m = uniform_filter(intensity, win)
    v = uniform_filter(intensity * intensity, win) - m * m
    cv2 = v / np.maximum(m * m, 1e-9)
    lo, hi = np.percentile(m, 20), np.percentile(m, 80)
    ok = np.isfinite(cv2) & (cv2 > 0) & (m > lo) & (m < hi)
    return 1.0 / np.median(cv2[ok])


def scale_test():
    """[0,pi] vs [0,2pi): which decoding matches the components?"""
    print("== 1. Decoding scale: corr(log|Im/Re|, log ratio implied by phi) ==")
    for pol in POLS:
        a, re, im, ph = (_read(pol, c) for c in ("amp", "real", "imgy", "pha"))
        ok = (a < 250) & (re > 2) & (im > 2) & (re < 250) & (im < 250) & (a > 10)
        obs = np.log(im[ok] / re[ok])
        row = []
        for scale, name in ((np.pi, "[0,pi]"), (2 * np.pi, "[0,2pi)")):
            phi = ph / 255.0 * scale
            # convention: |Re| ~ |sin phi|, |Im| ~ |cos phi|
            r = np.abs(np.cos(phi) / np.maximum(np.abs(np.sin(phi)), 1e-9))[ok]
            good = np.isfinite(r) & (r > 0.05) & (r < 20) & np.isfinite(obs)
            row.append(f"{name}={np.corrcoef(obs[good], np.log(r[good]))[0, 1]:+.4f}")
        print(f"   {pol}: " + "   ".join(row))


def quadrant_test():
    print("\n== 2. Quadrant occupancy of phi=pha/255*2pi (full range => ~25%) ==")
    for pol in POLS:
        phi = _read(pol, "pha") / 255.0 * 2 * np.pi
        q = np.floor(phi / (np.pi / 2)).astype(int) % 4
        print(f"   {pol}: " + "  ".join(f"Q{k+1}={(q == k).mean():.3f}"
                                        for k in range(4)))


def reconstruction_test():
    print("\n== 3. Reconstruction |Re|,|Im| from amp+phi alone ==")
    for pol in POLS:
        a, re, im, ph = (_read(pol, c) for c in ("amp", "real", "imgy", "pha"))
        phi = ph / 255.0 * 2 * np.pi
        ok = (a < 250) & (re < 250) & (im < 250) & (a > 10)
        pr, pi_ = np.abs(np.sin(phi)) * a, np.abs(np.cos(phi)) * a
        print(f"   {pol}: corr(|Re|)={np.corrcoef(re[ok], pr[ok])[0, 1]:.4f} "
              f" corr(|Im|)={np.corrcoef(im[ok], pi_[ok])[0, 1]:.4f}")


def physics_test():
    print("\n== 4. Physical coherences of the recovered SLC (7x7) ==")
    Shh, Shv, Svh, Svv = (slc(p) for p in POLS)
    rng = np.random.default_rng(0)
    rand = lambda X: np.abs(X) * np.exp(1j * rng.uniform(0, 2 * np.pi, X.shape))
    print(f"   HV-VH single angle (reciprocity) : {coherence(Shv, Svh).mean():.4f}")
    print(f"   HV-VH doubled angle (old route)  : "
          f"{coherence(np.abs(Shv) * np.exp(2j * np.angle(Shv)), np.abs(Svh) * np.exp(2j * np.angle(Svh))).mean():.4f}")
    print(f"   HH-VV                            : {coherence(Shh, Svv).mean():.4f}")
    print(f"   HH-HV                            : {coherence(Shh, Shv).mean():.4f}")
    print(f"   HV-VH random-phase NULL          : {coherence(rand(Shv), rand(Svh)).mean():.4f}")
    print("   spatial (oversampling) HH lag-k  : " + "  ".join(
        f"lag{k}={coherence(Shh[:, :-k], Shh[:, k:]).mean():.3f}" for k in (1, 2, 3)))


def payoff_test():
    """Coherent vs incoherent cross-pol combination — a NEGATIVE result.

    Reciprocity means HV and VH are largely the SAME complex sample (global
    coherence 0.83 after offset removal), so their SPECKLE is shared.
    Coherent averaging can only average the INDEPENDENT thermal part
    (-12% intensity in thermal-dominated areas) and therefore does not beat
    plain intensity averaging on ENL. The naive no-offset version scores a
    spuriously high ENL purely through destructive interference (its mean
    collapses to 0.44x) -- an artefact, not a gain.
    """
    print("\n== 5. Coherent vs incoherent cross-pol combination (NEGATIVE) ==")
    Shv, Svh = slc("hv"), slc("vh")
    Svh_a, off = align(Shv, Svh)
    print(f"   constant HV-VH phase offset removed: {np.degrees(off):+.2f} deg")
    variants = {
        "HV alone": np.abs(Shv) ** 2,
        "incoherent amp avg (current)": ((np.abs(Shv) + np.abs(Svh)) / 2) ** 2,
        "intensity avg": (np.abs(Shv) ** 2 + np.abs(Svh) ** 2) / 2,
        "coherent NAIVE (offset left in)": np.abs((Shv + Svh) / 2) ** 2,
        "coherent ALIGNED": np.abs((Shv + Svh_a) / 2) ** 2,
    }
    span = uniform_filter(np.abs(slc("hh")) ** 2 + np.abs(slc("vv")) ** 2, 9)
    dark = span < np.percentile(span, 20)          # fair, HV-independent mask
    print(f"   {'variant':<34} {'ENL':>7} {'mean':>8} {'darkI':>9} {'corrSpan':>9}")
    for name, X in variants.items():
        c = np.corrcoef(uniform_filter(X, 3).ravel(), span.ravel())[0, 1]
        print(f"   {name:<34} {enl(X):>7.3f} {np.sqrt(X).mean():>8.2f} "
              f"{X[dark].mean():>9.1f} {c:>9.4f}")
    print("   -> aligned coherent fusion LOSES to intensity averaging: shared")
    print("      speckle cannot be averaged away. Only the thermal floor drops.")


def snr_map_test():
    """The correct single-angle coherence map vs the old doubled-angle one."""
    print("\n== 6. Per-pixel SNR map: single angle vs doubled angle ==")
    Shv, Svh = slc("hv"), slc("vh")
    Svh_a, _ = align(Shv, Svh)
    single = coherence(Shv, Svh_a)
    dbl = coherence(np.abs(Shv) * np.exp(2j * np.angle(Shv)),
                    np.abs(Svh) * np.exp(2j * np.angle(Svh)))
    span = uniform_filter(np.abs(slc("hh")) ** 2 + np.abs(slc("vv")) ** 2, 9)
    dark = span < np.percentile(span, 20)
    bright = np.abs(Shv) ** 2 > np.percentile(np.abs(Shv) ** 2, 90)
    for name, m in (("single (correct)", single), ("doubled (old)", dbl)):
        print(f"   {name:<18} mean={m.mean():.4f}  bright10%={m[bright].mean():.4f}"
              f"  dark20%={m[dark].mean():.4f}  contrast={m[bright].mean()/m[dark].mean():.2f}")
    print("   -> single-angle is less noisy (higher mean) but NOT a better")
    print("      discriminator (lower bright/dark contrast). No free win here.")


if __name__ == "__main__":
    scale_test()
    quadrant_test()
    reconstruction_test()
    physics_test()
    payoff_test()
    snr_map_test()
    print("\nVERDICT: pha = full-range [0,2pi) SLC phase (corr 0.996 "
          "reconstruction, uniform quadrants). The complex data IS recoverable "
          "and the 'phase is destroyed' conclusion was a decoding bug.")
    print("HOWEVER: the obvious payoff does NOT materialise. HV/VH are ~0.83 "
          "coherent (the same physical sample), so coherent fusion cannot "
          "average speckle away, and the corrected SNR map is no better a "
          "discriminator than the old doubled-angle one. The value of the fix "
          "is that it unlocks the C3/T3 covariance route, not a quick win.")
