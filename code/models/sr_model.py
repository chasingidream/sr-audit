"""Full spatiotemporal SR model: Stage0 encoder -> Stage1 RSA -> Stage2 fusion
-> Stage3 reconstruction, with optional HR-reference guidance.

Pipeline (see docs/method_design.md):
  [X_1..X_T LR multi-temporal]  --shared encoder--> features
                                  --RSA--> align each frame to frame 0
                                  --TemporalFusion--> fused LR feature
  [Y_ref HR reference]          --shared encoder--> guidance feature
                                  --concat/add--> fused feature
                                  --Reconstruction (PixelShuffle x scale)--> Y_1..Y_T
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import FeatureExtractor, TemporalFusion, Reconstruction
from models.rsa import RSAModule
from models.xattn_align import CrossAttnAlign


class SpatioTemporalSR(nn.Module):
    def __init__(self, in_ch=3, feat=64, scale=4, T=5, rsa_radius=4,
                 n_groups=4, n_blocks=4, use_hr_ref=True, use_conf=True,
                 align_mode="cost_volume", align_ref="frame0", out_size=None,
                 upsampler="pixelshuffle", residual_base=False,
                 zero_init_residual=False):
        super().__init__()
        self.T = T
        self.scale = scale
        self.out_size = (out_size, out_size) if isinstance(out_size, int) \
            else out_size                   # 真实 GSD 非整数倍时末端插值目标尺寸
        self.use_hr_ref = use_hr_ref
        self.align_mode = align_mode      # "cost_volume" | "cross_attn" | "none"
        self.align_ref = align_ref        # "frame0" | "hr"
        # residual_base: 输出 = bicubic_up(lrs[:,0]) + 网络残差（2026-09-04 方案A）。
        # 结构上保证起点=bicubic，L1 会惩罚残差里的周期伪影 -> 抑制棋盘/网格。
        self.residual_base = residual_base
        self.extractor = FeatureExtractor(in_ch, feat, n_groups, n_blocks)
        self.rsa = RSAModule(feat, radius=rsa_radius, use_conf=use_conf)
        if align_mode == "cross_attn":
            self.xattn = CrossAttnAlign(feat)
        self.fusion = TemporalFusion(feat)
        self.recon = Reconstruction(feat, scale, in_ch, n_blocks, upsampler=upsampler,
                                    zero_init=zero_init_residual)
        if use_hr_ref:
            self.guidance = nn.Conv2d(feat, feat, 1, 1, 0)

    def forward(self, lrs, hr_ref=None, return_aux=False, use_rsa=None, use_ref=None):
        # lrs: [B, T, C, h, w]  (T low-res multi-temporal frames, same spatial size)
        if use_rsa is None:
            use_rsa = True
        if use_ref is None:
            use_ref = self.use_hr_ref
        B, T, C, h, w = lrs.shape
        feats = [self.extractor(lrs[:, t]) for t in range(T)]  # list [B,feat,h,w]
        ref_feat = feats[0]

        flows, confs = [], []
        if use_rsa and T > 1 and self.align_mode != "none":
            if self.align_mode == "cross_attn":
                # 内容鲁棒对齐：target(LR frame t) 作 query，reference 作 key/value。
                # reference 默认 frame0；若 align_ref=="hr" 且提供了 hr_ref，则用 HR 特征
                # （跨传感器设定，HR 仅作对齐锚点，不泄漏进重建）。
                if self.align_ref == "hr" and hr_ref is not None:
                    hr_f = self.extractor(hr_ref)
                    hr_f = F.interpolate(hr_f, size=(h, w), mode="bilinear",
                                         align_corners=True)
                    # 跨传感器：所有 LR 帧都对齐到 HR 参考
                    aligned = [self.xattn(hr_f, feats[t]) for t in range(T)]
                else:
                    # 多时相：frame0 作锚点（不对自己对齐），frame1..T-1 对齐到 frame0
                    aligned = [ref_feat] + [self.xattn(ref_feat, feats[t])
                                            for t in range(1, T)]
                aligned = torch.stack(aligned, dim=1)  # [B, T, feat, h, w]
            else:  # cost_volume (原 RSA)
                aligned = [ref_feat]
                for t in range(1, T):
                    a, flow, conf = self.rsa(ref_feat, feats[t])
                    if conf is not None:
                        a = a * conf
                    aligned.append(a)
                    flows.append(flow)
                    if conf is not None:
                        confs.append(conf)
                aligned = torch.stack(aligned, dim=1)  # [B, T, feat, h, w]
        else:
            # 不用对齐：直接对各帧特征做融合（隔离"自对齐"模块的贡献）
            aligned = torch.stack(feats, dim=1)

        fused = self.fusion(aligned)           # [B, feat, h, w]

        if use_ref and self.use_hr_ref and hr_ref is not None:
            hr_f = self.extractor(hr_ref)      # [B, feat, H, W]
            hr_f = F.interpolate(hr_f, size=(h, w), mode="bilinear",
                                 align_corners=True)
            fused = fused + self.guidance(hr_f)

        sr = self.recon(fused)                 # [B, C, h*scale, w*scale]
        if self.residual_base:
            # 方案A：bicubic 基底 + 残差。基底取 frame0（对齐锚点，无泄漏）。
            base = F.interpolate(lrs[:, 0].float(), size=sr.shape[-2:],
                                 mode="bicubic", align_corners=True)
            sr = sr + base
        if self.out_size is not None and tuple(sr.shape[-2:]) != tuple(self.out_size):
            # 真实 GSD 比例(6.63x)非整数倍：resize 到真值尺寸。
            # antialias=True 抑制非整数比重采样的摩尔/拍频伪影（2026-09-04 诊断）。
            sr = F.interpolate(sr, size=tuple(self.out_size), mode="bilinear",
                               align_corners=True, antialias=True)
        if return_aux:
            return sr, {"flows": flows, "confs": confs}
        return sr

    def cyclic_loss(self, lrs):
        """Aggregate self-supervised alignment loss over all frame pairs."""
        feats = [self.extractor(lrs[:, t]) for t in range(lrs.shape[1])]
        ref_feat = feats[0]
        loss = 0.0
        for t in range(1, len(feats)):
            loss = loss + self.rsa.cyclic_loss(ref_feat, feats[t])
        return loss / max(1, len(feats) - 1)


def build_model(cfg):
    return SpatioTemporalSR(
        in_ch=cfg.get("in_ch", 3),
        feat=cfg.get("feat", 64),
        scale=cfg.get("scale", 4),
        T=cfg.get("T", 5),
        rsa_radius=cfg.get("rsa_radius", 4),
        n_groups=cfg.get("n_groups", 4),
        n_blocks=cfg.get("n_blocks", 4),
        use_hr_ref=cfg.get("use_hr_ref", True),
        use_conf=cfg.get("use_conf", True),
        align_mode=cfg.get("align_mode", "cost_volume"),
        align_ref=cfg.get("align_ref", "frame0"),
        out_size=cfg.get("out_size", None),
        upsampler=cfg.get("upsampler", "pixelshuffle"),
        residual_base=cfg.get("residual_base", False),
        zero_init_residual=cfg.get("zero_init_residual", True)
        if cfg.get("residual_base", False) else False,
    )
