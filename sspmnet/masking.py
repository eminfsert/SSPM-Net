"""
Synchronized Spatio-Polarimetric Masking (SSPM).

The core methodological contribution of SSPM-Net.

  • Co-pol (HH, VV): independent blind-spot masks. Dropped pixels are
    replaced by a random neighbor (Noise2Void style), so the network must
    predict the hidden value from its surroundings.
  • Cross-pol (HV, VH): a SYNCHRONIZED mask drops the SAME pixels in both
    channels. Because monostatic reciprocity gives HV and VH the same clean
    signal but independent noise, predicting masked HV from noisy VH (and
    vice versa) is a Noise2Noise objective on a physical basis.

Channel order: HH=0, HV=1, VH=2, VV=3.
"""
import torch
import torch.nn as nn


class BernoulliMasker(nn.Module):
    """Pixel-wise Bernoulli blind-spot mask with neighbor replacement.

    ``p`` is the probability that a pixel is KEPT. Dropped pixels take the
    value of a randomly shifted neighbor (never zeroed — that would destroy
    image structure).
    """

    def __init__(self, p: float = 0.7):
        super().__init__()
        if not 0.0 < p < 1.0:
            raise ValueError(f"keep probability must be in (0, 1), got {p}")
        self.p = p

    def _neighbor_replace(self, x: torch.Tensor, mask: torch.Tensor):
        """Replace dropped pixels (mask==0) with a random shifted neighbor."""
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
        idx = torch.randint(0, len(shifts), (1,)).item()
        dy, dx = shifts[idx]
        shifted = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        inv_mask = 1.0 - mask
        return x * mask + shifted * inv_mask

    def forward(self, x: torch.Tensor):
        """Returns (masked_input, mask). In eval mode the input is untouched."""
        if not self.training:
            mask = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3],
                              device=x.device, dtype=x.dtype)
            return x, mask

        mask = torch.bernoulli(
            torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.p,
                       device=x.device, dtype=x.dtype)
        )
        return self._neighbor_replace(x, mask), mask


class QuadPolSpatialMasker(nn.Module):
    """SSPM: independent masks on HH/VV, synchronized mask on HV/VH."""

    def __init__(self, keep_prob: float = 0.7):
        super().__init__()
        self.spatial_masker = BernoulliMasker(p=keep_prob)

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, 4, H, W) [HH, HV, VH, VV].

        Returns a dict with the masked input and the masks needed by the
        loss functions: ``masked_input``, ``mask_hh``, ``mask_vv``,
        ``mask_xpol`` (shared HV/VH mask).
        """
        hh = x[:, 0:1]
        hv = x[:, 1:2]
        vh = x[:, 2:3]
        vv = x[:, 3:4]

        # Co-pol: independent blind-spot masks
        hh_masked, hh_mask = self.spatial_masker(hh)
        vv_masked, vv_mask = self.spatial_masker(vv)

        # Cross-pol: synchronized mask (same pixels in HV and VH)
        if not self.training:
            mask_xpol = torch.ones_like(hv)
            hv_masked, vh_masked = hv, vh
        else:
            mask_xpol = torch.bernoulli(torch.full_like(hv, self.spatial_masker.p))
            hv_masked = self.spatial_masker._neighbor_replace(hv, mask_xpol)
            vh_masked = self.spatial_masker._neighbor_replace(vh, mask_xpol)

        masked_input = torch.cat([hh_masked, hv_masked, vh_masked, vv_masked], dim=1)
        return {
            "masked_input": masked_input,
            "mask_hh": hh_mask,
            "mask_vv": vv_mask,
            "mask_xpol": mask_xpol,
        }
