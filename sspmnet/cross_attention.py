"""
Cross-polarization attention.

After per-channel denoising, the four polarization feature maps
(HH, HV, VH, VV) exchange information: at every spatial position the four
channels act as four tokens and attend to one another. This exploits
cross-polarization correlations (e.g. HH/VV structure, HV/VH reciprocity).
"""
import torch
import torch.nn as nn


class CrossPolarizationAttention(nn.Module):
    """Multi-head self-attention across the polarization channels."""

    def __init__(self, feat_dim: int = 64, num_heads: int = 4,
                 num_pols: int = 4, dropout: float = 0.1):
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.num_pols = num_pols
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(feat_dim, 3 * feat_dim)
        self.proj = nn.Linear(feat_dim, feat_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features: list) -> list:
        """``features``: list of 4 tensors (B, D, H, W). Returns same shapes."""
        B, D, H, W = features[0].shape
        N = self.num_pols

        # (B, 4, D, H, W) -> (B*H*W, 4, D): 4 polarization tokens per pixel
        x = torch.stack(features, dim=1)
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, N, D)

        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B * H * W, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * H * W, N, D)
        x = self.proj_drop(self.proj(x))
        x = residual + x
        x = x + self.ffn(self.norm2(x))

        x = x.reshape(B, H, W, N, D).permute(0, 3, 4, 1, 2)   # (B, 4, D, H, W)
        return [x[:, i] for i in range(N)]
