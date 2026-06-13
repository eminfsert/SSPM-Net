"""
Loss functions and training utilities for SSPM-Net's zero-shot objective.

Total loss (assembled in ``trainer.py``):

    L = L_mask (co-pol blind-spot + cross-pol N2N)
        + lambda_tv   * L_tv        (edge-aware total variation)
        + lambda_pol  * L_pol       (HV ~ VH reciprocity)
        + lambda_bound* L_bound     (keep output in [0, 1])
        + lambda_nl   * L_nl        (non-local self-similarity)
        + lambda_hist * L_hist      (speckle histogram -> Rayleigh)
        + lambda_fact * L_fact      (speckle factorization: y ~ x * S)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== #
#  Core self-supervised losses                                            #
# ====================================================================== #

class MaskedL1Loss(nn.Module):
    """L1 loss evaluated only on the dropped (blind-spot) pixels."""

    def forward(self, pred, target, mask):
        """pred/target/mask: (B, 1, H, W); mask is 1=kept, 0=dropped."""
        inv_mask = 1.0 - mask
        diff = (pred - target).abs() * inv_mask
        n_dropped = inv_mask.sum().clamp(min=1.0)
        return diff.sum() / n_dropped


def adaptive_tv_loss(x: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Edge-aware total variation.

    TV is down-weighted where the original image has strong gradients, so
    flat areas are smoothed while edges are preserved.
    """
    grad_h = (original[:, :, 1:, :] - original[:, :, :-1, :]).abs()
    grad_w = (original[:, :, :, 1:] - original[:, :, :, :-1]).abs()

    weight_h = torch.exp(-grad_h * 10.0)
    weight_w = torch.exp(-grad_w * 10.0)

    tv_h = ((x[:, :, 1:, :] - x[:, :, :-1, :]).abs() * weight_h).mean()
    tv_w = ((x[:, :, :, 1:] - x[:, :, :, :-1]).abs() * weight_w).mean()
    return tv_h + tv_w


def polarization_consistency_loss(denoised: torch.Tensor) -> torch.Tensor:
    """HV ~ VH reciprocity constraint (channels 1 and 2)."""
    return (denoised[:, 1:2] - denoised[:, 2:3]).abs().mean()


def bound_loss(denoised: torch.Tensor) -> torch.Tensor:
    """Penalize values outside [0, 1] (stabilizes the Swin branch)."""
    return (torch.relu(denoised - 1.0) + torch.relu(-denoised)).mean()


def polarimetric_nl_loss(x: torch.Tensor, ref: torch.Tensor,
                         window: int = 7, sigma: float = 0.1) -> torch.Tensor:
    """Non-local self-similarity loss.

    Weights neighbors by similarity in the reference image and pulls each
    pixel toward its non-local weighted average, encouraging consistency in
    homogeneous regions without blurring edges.
    """
    B, C, H, W = x.shape
    pad = window // 2
    K = window * window

    ref_unf = F.unfold(ref, kernel_size=window, padding=pad).view(B, C, K, H, W)
    ref_self = ref.unsqueeze(2)
    dist = ((ref_unf - ref_self) ** 2).sum(dim=1)
    w = torch.exp(-dist / (sigma * sigma + 1e-12))
    w[:, K // 2, :, :] = 0.0
    w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-8)

    x_unf = F.unfold(x, kernel_size=window, padding=pad).view(B, C, K, H, W)
    x_avg = (x_unf * w_norm.unsqueeze(1)).sum(dim=2)
    return ((x - x_avg) ** 2).mean()


# ====================================================================== #
#  Speckle factorization + histogram matching                            #
# ====================================================================== #

def simulate_speckle_amplitude(clean_amp: np.ndarray, looks: int = 1,
                               rng: np.random.Generator = None) -> np.ndarray:
    """Goodman multiplicative speckle on amplitude: y = x * sqrt(U),
    U ~ Gamma(L, 1/L). For L=1 the amplitude speckle is Rayleigh."""
    if rng is None:
        rng = np.random.default_rng()
    speckle_intensity = rng.gamma(shape=looks, scale=1.0 / looks, size=clean_amp.shape)
    return (clean_amp * np.sqrt(speckle_intensity)).astype(clean_amp.dtype)


def compute_reference_histogram(looks: int, n_bins: int, range_max: float,
                                n_samples: int = 200000, device="cpu"):
    """Reference speckle histogram (ideal Rayleigh for L=1) on a flat image.

    Returns (hist, bin_centers, step) — ``hist`` is the target distribution
    that the learned speckle factor ``S`` is matched against.
    """
    rng = np.random.default_rng(0)
    side = int(np.ceil(np.sqrt(n_samples)))
    clean_unit = np.ones((side, side), dtype=np.float32)
    samples = simulate_speckle_amplitude(clean_unit, looks=looks, rng=rng).flatten()
    samples = np.clip(samples, 0, range_max)

    bin_edges = np.linspace(0.0, range_max, n_bins + 1)
    hist, _ = np.histogram(samples, bins=bin_edges)
    hist = hist / max(hist.sum(), 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    step = float(bin_edges[1] - bin_edges[0])
    return (torch.from_numpy(hist).float().to(device),
            torch.from_numpy(bin_centers).float().to(device), step)


def compute_soft_histogram(values_2d, bin_centers, step):
    """Differentiable soft histogram (triangular kernel) of ``values_2d``."""
    K = bin_centers.shape[0]
    bc = bin_centers.view(K, 1, 1, 1)
    v = values_2d.unsqueeze(0)
    delta = torch.clamp(1.0 - torch.abs(v - bc) / step, min=0.0)
    counts = delta.sum(dim=(1, 2, 3))
    n_total = values_2d.shape[-2] * values_2d.shape[-1] * values_2d.shape[-3]
    return counts / max(n_total, 1)


# ====================================================================== #
#  Pre-warmup target                                                      #
# ====================================================================== #

def warmup_target_4ch(noisy_4ch_np: np.ndarray) -> np.ndarray:
    """Bilateral-filtered version of the noisy image, used as a short
    pre-warmup target to quench the chaos of random initialization."""
    from skimage.restoration import denoise_bilateral
    out = np.zeros_like(noisy_4ch_np, dtype=np.float32)
    for c in range(noisy_4ch_np.shape[0]):
        out[c] = denoise_bilateral(noisy_4ch_np[c].astype(np.float32),
                                   sigma_color=0.1, sigma_spatial=2.0)
    return out
