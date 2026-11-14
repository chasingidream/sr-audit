"""Lightweight backbone building blocks for the spatiotemporal SR model.

Designed for an 8 GB GPU: a small RCAB (Residual Channel Attention Block) stack
keeps the parameter count low (~hundreds of K) while staying competitive with the
HAUNet / TransENet family validated on RS-SR benchmarks. Temporal fusion uses a
lightweight multi-head self-attention over the time axis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RCAB(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        res = self.body(x)
        w = self.ca(res)
        return x + res * w


class ResidualGroup(nn.Module):
    def __init__(self, ch, n_blocks=4, reduction=4):
        super().__init__()
        self.body = nn.Sequential(*[RCAB(ch, reduction) for _ in range(n_blocks)])
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        return x + self.conv(self.body(x))


class FeatureExtractor(nn.Module):
    """Shared-weight encoder for both LR frames and the HR reference."""

    def __init__(self, in_ch=3, feat=64, n_groups=4, n_blocks=4):
        super().__init__()
        self.head = nn.Conv2d(in_ch, feat, 3, 1, 1)
        self.groups = nn.Sequential(
            *[ResidualGroup(feat, n_blocks) for _ in range(n_groups)])
        self.tail = nn.Conv2d(feat, feat, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        x = self.groups(x)
        return self.tail(x)


class TemporalFusion(nn.Module):
    """Self-attention fusion over the time axis of aligned per-frame features."""

    def __init__(self, feat, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(feat)
        self.attn = nn.MultiheadAttention(feat, num_heads=num_heads,
                                          batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(feat, feat * 2),
            nn.GELU(),
            nn.Linear(feat * 2, feat),
        )

    def forward(self, x):
        # x: [B, T, C, H, W] -> [B*H*W, T, C]
        B, T, C, H, W = x.shape
        flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        h = self.norm(flat)
        a, _ = self.attn(h, h, h)
        flat = flat + a
        flat = flat + self.ffn(flat)
        out = flat.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)  # [B, T, C, H, W]
        return out.mean(dim=1)  # collapse time -> [B, C, H, W]


class Reconstruction(nn.Module):
    def __init__(self, feat=64, scale=4, out_ch=3, n_blocks=4, upsampler="pixelshuffle",
                 zero_init=False):
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(*[RCAB(feat, 4) for _ in range(n_blocks)])
        # upsampler="pixelshuffle"(默认, 历史) | "convtranspose"：
        #   convtranspose = 单一 ConvTranspose2d(k=s, s=s)，无重叠核 -> 无棋盘伪影。
        #   动机（2026-09-04）：PixelShuffle(7)+bilinear 448->424 在所有档输出上产生
        #   系统性棋盘/摩尔纹（见 diag_vis_samples / MEMORY），模型被钉在 bicubic±3dB。
        if upsampler == "convtranspose":
            self.up = nn.Sequential(
                nn.ConvTranspose2d(feat, out_ch, kernel_size=scale, stride=scale),
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(feat, feat * scale * scale, 3, 1, 1),
                nn.PixelShuffle(scale),
                nn.Conv2d(feat, out_ch, 3, 1, 1),
            )
        if zero_init:
            # 残差头零初始化：网络输出恒 0 -> 整体输出 = bicubic 基底（严格起点）。
            # 残差只能随训练"渐进长出"，防止 540 patch 上幻觉大斑块（2026-09-04）。
            last = self.up[-1]
            nn.init.zeros_(last.weight)
            if getattr(last, "bias", None) is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x):
        x = self.body(x)
        return self.up(x)
