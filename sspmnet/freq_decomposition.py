"""
Wavelet-based frequency decomposition.

A 1-level 2-D Haar DWT implemented as fixed (non-learnable) depthwise
convolutions. Splits an image into:
    • LL sub-band               — smooth content / radiometry
    • LH + HL + HH sub-bands    — edges, texture, speckle
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecomposition(nn.Module):
    """2-D Haar DWT and its inverse, as fixed depthwise convolutions.

    Forward:  (B, C, H, W) -> low (B, C, H/2, W/2), high (B, 3C, H/2, W/2)
    Inverse:  low, high     -> (B, C, H, W)
    """

    def __init__(self):
        super().__init__()
        self._register_haar_filters()

    def _register_haar_filters(self):
        """Create the four 2-D Haar separable kernels as buffers."""
        inv_sqrt2 = 0.5 ** 0.5
        lo = torch.tensor([inv_sqrt2, inv_sqrt2])
        hi = torch.tensor([-inv_sqrt2, inv_sqrt2])

        # 2-D separable products -> (1, 1, 2, 2)
        ll = (lo.unsqueeze(1) * lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        lh = (hi.unsqueeze(1) * lo.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        hl = (lo.unsqueeze(1) * hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        hh = (hi.unsqueeze(1) * hi.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

        self.register_buffer("_ll", ll)
        self.register_buffer("_lh", lh)
        self.register_buffer("_hl", hl)
        self.register_buffer("_hh", hh)

    def decompose(self, x: torch.Tensor):
        """One level of forward DWT.

        Returns ``low`` (LL) and ``high`` (stacked LH, HL, HH).
        """
        B, C, H, W = x.shape

        # Pad to even dimensions if needed
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        ll = self._ll.expand(C, -1, -1, -1)
        lh = self._lh.expand(C, -1, -1, -1)
        hl = self._hl.expand(C, -1, -1, -1)
        hh = self._hh.expand(C, -1, -1, -1)

        low = F.conv2d(x, ll, stride=2, groups=C)
        d_lh = F.conv2d(x, lh, stride=2, groups=C)
        d_hl = F.conv2d(x, hl, stride=2, groups=C)
        d_hh = F.conv2d(x, hh, stride=2, groups=C)

        high = torch.cat([d_lh, d_hl, d_hh], dim=1)     # (B, 3C, H/2, W/2)
        return low, high

    def reconstruct(self, low: torch.Tensor, high: torch.Tensor,
                    output_size: tuple = None):
        """Inverse DWT. ``output_size`` crops back to the pre-pad size."""
        B, C, Hh, Wh = low.shape
        d_lh, d_hl, d_hh = torch.chunk(high, 3, dim=1)

        ll = self._ll.expand(C, -1, -1, -1)
        lh = self._lh.expand(C, -1, -1, -1)
        hl = self._hl.expand(C, -1, -1, -1)
        hh = self._hh.expand(C, -1, -1, -1)

        x = F.conv_transpose2d(low, ll, stride=2, groups=C)
        x = x + F.conv_transpose2d(d_lh, lh, stride=2, groups=C)
        x = x + F.conv_transpose2d(d_hl, hl, stride=2, groups=C)
        x = x + F.conv_transpose2d(d_hh, hh, stride=2, groups=C)

        if output_size is not None:
            x = x[:, :, :output_size[0], :output_size[1]]
        return x

    def forward(self, x: torch.Tensor):
        return self.decompose(x)
