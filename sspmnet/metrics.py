"""
SAR despeckling quality metrics.

Two flavours:
  • torch versions (``enl``, ``epi``, ``enl_roi``) — used inside the training
    loop for the best-checkpoint gate.
  • numpy versions (``enl_full``, ``epi_metric``, ``ssim_metric``,
    ``enl_roi_multi``, ``reciprocity_metrics``, ``lee_filter``) — used for the
    final, fair head-to-head report.

Definitions:
  ENL  = mean^2 / variance of intensity (higher = less speckle)
  EPI  = Pearson correlation of Sobel-gradient magnitudes (edge preservation)
  ENL is measured in the intensity domain (amplitude^2).
"""
import numpy as np
import torch
import torch.nn.functional as F


# ====================================================================== #
#  torch metrics (training-time best-checkpoint gate)                     #
# ====================================================================== #

def enl(image: torch.Tensor) -> float:
    """Global ENL = mean^2 / var (scale-invariant)."""
    mu = image.mean().item()
    var = image.var().item()
    return float("inf") if var == 0 else (mu ** 2) / var


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    C = x.shape[1]
    gx = F.conv2d(x, sobel_x.expand(C, -1, -1, -1), groups=C)
    gy = F.conv2d(x, sobel_y.expand(C, -1, -1, -1), groups=C)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def epi(denoised: torch.Tensor, original: torch.Tensor) -> float:
    """Edge Preservation Index in [-1, 1]; closer to 1 is better."""
    gd = _sobel_edges(denoised).reshape(-1)
    go = _sobel_edges(original).reshape(-1)
    gd_c = gd - gd.mean()
    go_c = go - go.mean()
    num = (gd_c * go_c).sum()
    den = torch.sqrt((gd_c ** 2).sum() * (go_c ** 2).sum() + 1e-8)
    return (num / den).item()


def _find_homogeneous_rois(ref_image: torch.Tensor, patch_size: int = 64,
                           top_k: int = 10) -> list:
    """Locate the most homogeneous ROIs in the reference image (CV + gradient
    filtered). Used so ENL is measured on flat regions, not edges."""
    ref = ref_image.detach().float()
    B, C, H, W = ref.shape
    ref_mean = ref.mean(dim=1, keepdim=True)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=ref.dtype, device=ref.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=ref.dtype, device=ref.device).view(1, 1, 3, 3)
    gx = F.conv2d(ref_mean, sobel_x, padding=1)
    gy = F.conv2d(ref_mean, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    scores = []
    for y in range(0, H - patch_size + 1, patch_size // 2):
        for x in range(0, W - patch_size + 1, patch_size // 2):
            ref_patch = ref[:, :, y:y + patch_size, x:x + patch_size]
            grad_patch = grad_mag[:, :, y:y + patch_size, x:x + patch_size]
            mean_all = ref_patch.mean().item()
            if not (0.10 < mean_all < 0.90):
                continue
            cv = ref_patch.std().item() / (mean_all + 1e-8)
            if cv > 0.8:
                continue
            scores.append((grad_patch.mean().item(), y, x))

    if not scores:
        return []
    scores.sort(key=lambda t: t[0])
    return [(y, x) for _, y, x in scores[:top_k]]


def enl_roi(image: torch.Tensor, patch_size: int = 64, top_k: int = 10,
            ref_image: torch.Tensor = None, _rois: list = None) -> float:
    """ROI-based ENL — per-channel ENL averaged over homogeneous ROIs
    (trimmed mean). ROIs are found once on ``ref_image``."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if ref_image is None:
        ref_image = image
    elif ref_image.dim() == 3:
        ref_image = ref_image.unsqueeze(0)

    img = image.detach().float()
    B, C, H, W = img.shape
    if _rois is None:
        _rois = _find_homogeneous_rois(ref_image, patch_size, top_k)
    if not _rois:
        return enl(image)

    vals = []
    for y, x in _rois:
        for c in range(C):
            roi = img[:, c, y:y + patch_size, x:x + patch_size]
            mu = roi.mean().item()
            var = roi.var().item()
            if var > 1e-10 and mu > 0.05:
                vals.append((mu ** 2) / var)
    if not vals:
        return enl(image)
    vals.sort()
    trim = max(1, len(vals) // 5)
    trimmed = vals[trim:-trim] if len(vals) > 2 * trim else vals
    return sum(trimmed) / len(trimmed)


# ====================================================================== #
#  numpy metrics (final fair report)                                      #
# ====================================================================== #

def lee_filter(img: np.ndarray, window: int = 7, cu: float = 0.523) -> np.ndarray:
    """Classic Lee (1980) filter — included as a simple baseline."""
    from scipy.ndimage import uniform_filter
    m = uniform_filter(img, window)
    s = uniform_filter(img ** 2, window)
    v = np.maximum(s - m ** 2, 0)
    ov = float(img.var())
    w = v / np.maximum(v + cu ** 2 * ov, 1e-9)
    return m + w * (img - m)


def enl_full(a: np.ndarray) -> float:
    """Global intensity-domain ENL of an amplitude image."""
    I = a ** 2
    return float((I.mean() / I.std()) ** 2) if I.std() > 0 else float("nan")


def find_top_k_rois(a: np.ndarray, roi_size: int = 64, top_k: int = 10,
                    cv_max: float = 0.7):
    """Top-k lowest-variance homogeneous ROIs (CV filtered). Selected ONCE on
    the noisy image so every method is scored on the same locations."""
    h, w = a.shape
    rs = roi_size
    cands = []
    for i in range(0, h - rs, rs // 2):
        for j in range(0, w - rs, rs // 2):
            patch = a[i:i + rs, j:j + rs]
            mu = patch.mean()
            if mu < 1e-6:
                continue
            if patch.std() / mu > cv_max:
                continue
            cands.append((float(patch.var()), i, j))
    cands.sort(key=lambda t: t[0])
    if not cands:                          # fallback: drop CV filter
        for i in range(0, h - rs, rs // 2):
            for j in range(0, w - rs, rs // 2):
                cands.append((float(a[i:i + rs, j:j + rs].var()), i, j))
        cands.sort(key=lambda t: t[0])
    return [(i, j) for (_, i, j) in cands[:top_k]], rs


def enl_roi_multi(a: np.ndarray, ij_list, rs: int, trim_frac: float = 0.2) -> float:
    """Trimmed-mean intensity ENL over a fixed list of ROIs."""
    vals = []
    for (i, j) in ij_list:
        roi = a[i:i + rs, j:j + rs]
        I = roi ** 2
        mu, sigma = I.mean(), I.std()
        if sigma > 1e-10 and mu > 1e-6:
            vals.append((mu / sigma) ** 2)
    if not vals:
        return float("nan")
    vals.sort()
    if len(vals) > 4:
        k = max(1, int(len(vals) * trim_frac))
        vals = vals[k:-k]
    return float(np.mean(vals))


def epi_metric(noisy: np.ndarray, denoised: np.ndarray) -> float:
    """Edge Preservation Index (numpy / Sobel)."""
    from scipy.ndimage import sobel
    gn = np.hypot(sobel(noisy, 0), sobel(noisy, 1))
    gd = np.hypot(sobel(denoised, 0), sobel(denoised, 1))
    num = np.sum((gn - gn.mean()) * (gd - gd.mean()))
    den = np.sqrt(np.sum((gn - gn.mean()) ** 2) * np.sum((gd - gd.mean()) ** 2))
    return float(num / max(den, 1e-9))


def ssim_metric(noisy: np.ndarray, denoised: np.ndarray) -> float:
    """SSIM between denoised and the noisy input (structure preservation)."""
    from skimage.metrics import structural_similarity as sk_ssim
    dr = float(max(noisy.max(), denoised.max()) - min(noisy.min(), denoised.min()))
    if dr <= 0:
        return float("nan")
    return float(sk_ssim(noisy, denoised, data_range=dr))


def reciprocity_metrics(hv: np.ndarray, vh: np.ndarray) -> dict:
    """corr / MAD / RMSE between HV and VH — the polarimetric reciprocity
    evidence (closer HV<->VH = better physics consistency)."""
    a = hv.flatten().astype(np.float64)
    b = vh.flatten().astype(np.float64)
    return {"corr": float(np.corrcoef(a, b)[0, 1]),
            "mad": float(np.mean(np.abs(a - b))),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2)))}
