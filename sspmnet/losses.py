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

When complex (real/imaginary) auxiliary data is supplied (see
``sspmnet.complex_data``), L_mask additionally uses the independent
|Re| / |Im| pseudo-amplitude targets, and L_tv / L_nl are steered by a
multi-look guide (``guide_edge_weights``) instead of the noisy amplitude.
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


def adaptive_tv_loss(x: torch.Tensor, original: torch.Tensor,
                     weights=None) -> torch.Tensor:
    """Edge-aware total variation.

    TV is down-weighted where the original image has strong gradients, so
    flat areas are smoothed while edges are preserved. When ``weights``
    (a ``(weight_h, weight_w)`` pair, e.g. from
    :func:`guide_edge_weights`) is given, those precomputed maps are used
    instead of the gradients of the noisy ``original`` — a multi-look
    guide gives a far less noisy edge map, so flats are smoothed harder
    while true edges are protected better.
    """
    if weights is None:
        grad_h = (original[:, :, 1:, :] - original[:, :, :-1, :]).abs()
        grad_w = (original[:, :, :, 1:] - original[:, :, :, :-1]).abs()
        weight_h = torch.exp(-grad_h * 10.0)
        weight_w = torch.exp(-grad_w * 10.0)
    else:
        weight_h, weight_w = weights

    tv_h = ((x[:, :, 1:, :] - x[:, :, :-1, :]).abs() * weight_h).mean()
    tv_w = ((x[:, :, :, 1:] - x[:, :, :, :-1]).abs() * weight_w).mean()
    return tv_h + tv_w


def _box_blur(x: torch.Tensor, k: int = 3, passes: int = 2) -> torch.Tensor:
    """Repeated box blur (~= small Gaussian), per channel."""
    C = x.shape[1]
    kern = torch.full((C, 1, k, k), 1.0 / (k * k), dtype=x.dtype, device=x.device)
    for _ in range(passes):
        x = F.conv2d(x, kern, padding=k // 2, groups=C)
    return x


def guide_edge_weights(guide_amp: torch.Tensor, alpha: float = 3.0,
                       cv_protect: float = None):
    """TV edge weights from a multi-look guide amplitude image.

    Speckle is multiplicative, so edges are detected on the LOG intensity
    (ratio detector) of the lightly smoothed guide; gradients are
    normalized by their own mean, making ``alpha`` dimensionless (larger
    alpha = edges protected more aggressively).

    When ``cv_protect`` is given, a Lee-style heterogeneity gate is applied
    on top: the local coefficient of variation of the guide separates
    homogeneous areas (CV near the pure-speckle floor — for the ~8-look
    amplitude span that floor is ~0.18) from texture / point targets.
    Regularization stays fully on in homogeneous areas and shuts off where
    CV exceeds the threshold, so flats can be smoothed hard without
    touching deterministic structure.

    Parameters
    ----------
    guide_amp : (B, 1, H, W) tensor — multi-look amplitude guide (e.g. the
        8-look span built from |Re| / |Im| of all 4 channels).
    alpha : float — edge sensitivity.
    cv_protect : float or None — CV threshold of the heterogeneity gate
        (e.g. 0.3 for the 8-look span); None disables the gate.

    Returns
    -------
    (weight_h, weight_w) : tensors of shape (B, 1, H-1, W) / (B, 1, H, W-1),
        in (0, 1]; broadcast against the (B, 4, ...) TV terms.
    """
    with torch.no_grad():
        g = torch.log(_box_blur(guide_amp, k=3, passes=2) ** 2 + 1e-6)
        grad_h = (g[:, :, 1:, :] - g[:, :, :-1, :]).abs()
        grad_w = (g[:, :, :, 1:] - g[:, :, :, :-1]).abs()
        norm = 0.5 * (grad_h.mean() + grad_w.mean()) + 1e-8
        weight_h = torch.exp(-alpha * grad_h / norm)
        weight_w = torch.exp(-alpha * grad_w / norm)

        if cv_protect is not None and cv_protect > 0:
            mu = _box_blur(guide_amp, k=9, passes=1)
            m2 = _box_blur(guide_amp ** 2, k=9, passes=1)
            cv = torch.sqrt((m2 - mu ** 2).clamp(min=0)) / (mu + 1e-6)
            gate = torch.sigmoid((cv_protect - cv) / (0.25 * cv_protect))
            weight_h = weight_h * 0.5 * (gate[:, :, 1:, :] + gate[:, :, :-1, :])
            weight_w = weight_w * 0.5 * (gate[:, :, :, 1:] + gate[:, :, :, :-1])
    return weight_h, weight_w


def modulate_edge_weights(weights, factor: torch.Tensor):
    """Multiply TV edge-weight maps by a per-pixel factor map.

    ``weights`` is the ``(weight_h, weight_w)`` pair from
    :func:`guide_edge_weights` (shapes (B,1,H-1,W) / (B,1,H,W-1));
    ``factor`` is a (B,1,H,W) map (e.g. the phase-feedback smoothing boost),
    averaged onto each edge's two endpoints, as in the CV gate.
    """
    weight_h, weight_w = weights
    f_h = 0.5 * (factor[:, :, 1:, :] + factor[:, :, :-1, :])
    f_w = 0.5 * (factor[:, :, :, 1:] + factor[:, :, :, :-1])
    return weight_h * f_h, weight_w * f_w


def polarization_consistency_loss(denoised: torch.Tensor) -> torch.Tensor:
    """HV ~ VH reciprocity constraint (channels 1 and 2)."""
    return (denoised[:, 1:2] - denoised[:, 2:3]).abs().mean()


def bound_loss(denoised: torch.Tensor) -> torch.Tensor:
    """Penalize values outside [0, 1] (stabilizes the Swin branch)."""
    return (torch.relu(denoised - 1.0) + torch.relu(-denoised)).mean()


def polarimetric_nl_loss(x: torch.Tensor, ref: torch.Tensor,
                         window: int = 7, sigma: float = 0.1,
                         pixel_weight: torch.Tensor = None,
                         sigma_map: torch.Tensor = None) -> torch.Tensor:
    """Non-local self-similarity loss.

    Weights neighbors by similarity in the reference image and pulls each
    pixel toward its non-local weighted average, encouraging consistency in
    homogeneous regions without blurring edges. ``pixel_weight``
    ((B, 1, H, W), mean ~1) spatially modulates the pull — e.g. the phase
    feedback map boosts it where the observation is noise-dominated.
    ``sigma_map`` ((B, 1, H, W)) replaces the scalar ``sigma`` per pixel —
    a larger sigma accepts more neighbors, averaging harder.
    """
    B, C, H, W = x.shape
    pad = window // 2
    K = window * window

    ref_unf = F.unfold(ref, kernel_size=window, padding=pad).view(B, C, K, H, W)
    ref_self = ref.unsqueeze(2)
    dist = ((ref_unf - ref_self) ** 2).sum(dim=1)
    sig2 = (sigma_map ** 2 if sigma_map is not None else sigma * sigma)
    w = torch.exp(-dist / (sig2 + 1e-12))
    w[:, K // 2, :, :] = 0.0
    w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-8)

    x_unf = F.unfold(x, kernel_size=window, padding=pad).view(B, C, K, H, W)
    x_avg = (x_unf * w_norm.unsqueeze(1)).sum(dim=2)
    sq = (x - x_avg) ** 2
    if pixel_weight is not None:
        sq = sq * pixel_weight
    return sq.mean()


def edge_fidelity_loss(d: torch.Tensor, guide_log: torch.Tensor,
                       edge_w=None) -> torch.Tensor:
    """Edge-sharpness fidelity: match the output's LOG-domain gradients to
    the multi-look guide's at edges.

    TV / non-local regularization and TTA averaging wash edge profiles
    out; this term pulls them back to the (far sharper) ~8-look guide.
    The log domain makes the gradients contrast ratios, so the guide's
    span scale vs. the per-channel amplitude scale cancels.

    ``guide_log`` : (B, C, H, W) log-amplitude of the smoothed PER-CHANNEL
        guide — the gradient target must be each channel's own (matching a
        dark cross-pol channel to the bright span's gradients collapses
        it); where a channel is flat its own target is ~0, so shared edge
        LOCATIONS with per-channel TARGETS are safe.
    ``edge_w``    : optional ((B,1,H-1,W), (B,1,H,W-1)) weights focusing
        the loss on true edge locations (normalized span gradients fused
        with the phase snr-coherence gradients).
    """
    d_log = 0.5 * torch.log(d ** 2 + 1e-4)
    dh = d_log[:, :, 1:, :] - d_log[:, :, :-1, :]
    dw = d_log[:, :, :, 1:] - d_log[:, :, :, :-1]
    gh = guide_log[:, :, 1:, :] - guide_log[:, :, :-1, :]
    gw = guide_log[:, :, :, 1:] - guide_log[:, :, :, :-1]
    if edge_w is None:
        return (dh - gh).abs().mean() + (dw - gw).abs().mean()
    w_h, w_w = edge_w
    return (((dh - gh).abs() * w_h).sum() / (w_h.sum() * d.shape[1] + 1e-8)
            + ((dw - gw).abs() * w_w).sum() / (w_w.sum() * d.shape[1] + 1e-8))


def nl_polish(x: torch.Tensor, guide: torch.Tensor = None, window: int = 9,
              sigma: float = 0.1, strength: float = 0.5,
              protect: torch.Tensor = None) -> torch.Tensor:
    """Final-stage non-local refinement of a denoised image.

    One guided-NLM pass whose similarity weights come from the (already
    clean) output itself, blended back with per-pixel strength:

        out = (1 - s_px) * x + s_px * NLM_avg(x)
        s_px = strength * (1 - protect)

    Unlike a blur, dissimilar neighbors get ~zero weight, so edges and
    point targets are preserved while residual speckle grain — which has
    many similar neighbors in the cleaned image — is averaged out.
    ``protect`` ((B, 1, H, W), in [0, 1]) shuts the refinement off on
    heterogeneous / deterministic pixels (CV gate, phase 'det' map).
    """
    with torch.no_grad():
        if guide is None:
            guide = _box_blur(x, k=3, passes=1)
        B, C, H, W = x.shape
        pad = window // 2
        K = window * window
        g_unf = F.unfold(guide, kernel_size=window, padding=pad).view(B, C, K, H, W)
        dist = ((g_unf - guide.unsqueeze(2)) ** 2).sum(dim=1)
        w = torch.exp(-dist / (sigma * sigma + 1e-12))
        w[:, K // 2, :, :] = 0.0
        w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        x_unf = F.unfold(x, kernel_size=window, padding=pad).view(B, C, K, H, W)
        avg = (x_unf * w_norm.unsqueeze(1)).sum(dim=2)
        s_px = strength * (1.0 - protect) if protect is not None else strength
        return (1.0 - s_px) * x + s_px * avg


def ratio_whiteness_loss(denoised: torch.Tensor, noisy: torch.Tensor,
                         lags=(1, 2, 3), eps: float = 1e-3) -> torch.Tensor:
    """Spatial-whiteness penalty on the intensity ratio image.

    For a perfect result the ratio noisy^2 / denoised^2 is pure speckle —
    spatially WHITE (for critically sampled data). Residual speckle left
    in the output structures the ratio instead; penalizing the ratio's
    small-lag autocorrelation (after local-mean normalization, which
    removes scene-level structure) actively pushes that residue out.

    CAUTION: oversampled real data has correlated speckle, so keep the
    lags above the oversampling correlation length there (or use only on
    simulated speckle, which is white by construction).
    """
    r = (noisy ** 2 + eps) / (denoised ** 2 + eps)
    mu = _box_blur(r, k=9, passes=1)
    e = r / (mu + 1e-6) - 1.0
    var = (e ** 2).mean() + 1e-8
    loss = denoised.new_zeros(())
    for lag in lags:
        for dim in (-2, -1):
            n = e.shape[dim] - lag
            a = e.narrow(dim, 0, n)
            b = e.narrow(dim, lag, n)
            loss = loss + ((a * b).mean() / var) ** 2
    return loss


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
