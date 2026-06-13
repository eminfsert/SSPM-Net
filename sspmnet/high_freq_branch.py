"""
Swin Transformer branch.

Despite the historical name ``HighFreqBranch``, in SSPM-Net this branch
processes the LL (low-frequency) sub-band, where global context lives and
noise is low. It uses window-based multi-head self-attention with shifted
windows (Swin Transformer).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Linear -> GELU -> Drop -> Linear -> Drop."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Single Swin block with optional cyclic window shift."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def _create_mask(self, H: int, W: int, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = self._window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    @staticmethod
    def _window_partition(x: torch.Tensor, window_size: int):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size,
                   W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, window_size * window_size, C)

    @staticmethod
    def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
        B_nW = windows.shape[0]
        nH, nW = H // window_size, W // window_size
        B = B_nW // (nH * nW)
        x = windows.view(B, nH, nW, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad to multiples of window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        attn_mask = self._create_mask(Hp, Wp, x.device)
        x_windows = self._window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = self._window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class HighFreqBranch(nn.Module):
    """Swin Transformer stages over a single-channel sub-band."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        out_channels: int = None,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2]
        if num_heads is None:
            num_heads = [3, 6]
        if out_channels is None:
            out_channels = in_channels

        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 3, padding=1, bias=False)

        self.stages = nn.ModuleList()
        dim = embed_dim
        for depth, heads in zip(depths, num_heads):
            stage = nn.ModuleList()
            for j in range(depth):
                shift = 0 if (j % 2 == 0) else window_size // 2
                stage.append(SwinTransformerBlock(
                    dim=dim, num_heads=heads, window_size=window_size,
                    shift_size=shift, mlp_ratio=mlp_ratio,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                ))
            self.stages.append(stage)

        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, out_channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = self.patch_embed(x)                          # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)                 # (B, H*W, embed_dim)

        for stage in self.stages:
            for blk in stage:
                x = blk(x, H, W)

        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        return self.proj_out(x)
