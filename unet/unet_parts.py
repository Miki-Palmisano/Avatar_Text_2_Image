""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, time_emb_dim, mid_channels=None, groups=8):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # proietta l'embedding del timestep sullo stesso numero di canali
        # dell'input, così può essere sommato a x prima dei conv
        self.time_mlp = nn.Linear(time_emb_dim, in_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t_emb):
        t = self.time_mlp(t_emb)[:, :, None, None]  # (B, in_channels, 1, 1) -> broadcast su H,W
        x = x + t
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x, t_emb):
        x = self.pool(x)
        return self.conv(x, t_emb)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, time_emb_dim, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, time_emb_dim, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x1, x2, t_emb):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t_emb)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CrossAttention(nn.Module):
    """Spatial features (query) attend to text hidden states (key/value).
    This is the text-conditioning mechanism required by the assignment."""

    def __init__(self, channels, text_dim, n_heads=4, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.kv_proj = nn.Linear(text_dim, channels * 2)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, text_hidden, text_pad_mask=None):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q_proj(h).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k, v = self.kv_proj(text_hidden).chunk(2, dim=-1)  # (B, T, C) each
        attn_out, _ = self.attn(q, k, v, key_padding_mask=text_pad_mask)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.out_proj(attn_out)

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding (Ho et al. 2020)."""
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb