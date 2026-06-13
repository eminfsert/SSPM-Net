"""
Reconstruction layer — fuses the processed sub-bands via inverse DWT and
applies refinement convolutions to produce a per-channel feature map.
"""
import torch
import torch.nn as nn

from .freq_decomposition import FrequencyDecomposition


class ReconstructionLayer(nn.Module):
    """Inverse DWT + feature fusion + refinement convolutions."""

    def __init__(self, in_channels: int = 1, mid_channels: int = 64,
                 out_channels: int = None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.idwt = FrequencyDecomposition()   # uses .reconstruct()

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=True),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor,
                output_size: tuple = None) -> torch.Tensor:
        fused = self.idwt.reconstruct(low, high, output_size=output_size)
        return self.refine(fused)
