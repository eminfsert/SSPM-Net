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
    print("\n== 5. Payoff: coherent vs incoherent cross-pol combination ==")
    Shv, Svh = slc("hv"), slc("vh")
    Ihv = np.abs(Shv) ** 2
    variants = {
        "HV alone": Ihv,
        "VH alone": np.abs(Svh) ** 2,
        "incoherent amp avg (current)": ((np.abs(Shv) + np.abs(Svh)) / 2) ** 2,
        "intensity avg": (Ihv + np.abs(Svh) ** 2) / 2,
        "COHERENT avg (uses phase)": np.abs((Shv + Svh) / 2) ** 2,
    }
    dark = Ihv < np.percentile(Ihv, 20)
    span = uniform_filter(np.abs(slc("hh")) ** 2 + np.abs(slc("vv")) ** 2, 9)
    print(f"   {'variant':<30} {'ENL':>7} {'darkMean':>10} {'corrSpan':>9}")
    for name, X in variants.items():
        c = np.corrcoef(uniform_filter(X, 3).ravel(), span.ravel())[0, 1]
        print(f"   {name:<30} {enl(X):>7.3f} {X[dark].mean():>10.1f} {c:>9.4f}")


if __name__ == "__main__":
    scale_test()
    quadrant_test()
    reconstruction_test()
    physics_test()
    payoff_test()
    print("\nVERDICT: pha = full-range [0,2pi) SLC phase. The complex data is "
          "recoverable; the 'phase is destroyed' conclusion was a decoding bug.")
