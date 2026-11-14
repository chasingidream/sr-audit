"""CrossAttnAlign — 内容鲁棒的特征对齐模块（替换 RSA 的 cost-volume 流）。

设计动机（见项目记忆 2026-08-31）：
  RSA 的 RAFT cost-volume 对齐依赖帧间"稠密像素相关峰值"，而遥感多时相/跨传感器
  帧间内容差异（云/季节/辐射/传感器）远大于几何位移 -> cost volume 不成峰 -> 架构性失效。
  CrossAttnAlign 改用交叉注意力：target 作 query、reference 作 key/value，按学到的
  相似度把 target 特征对齐到 reference，不要求稠密相关峰值，对内容差异鲁棒。
  保留"配准无关自对齐"卖点（无需外部配准、从数据自学对齐）。

前向：
  aligned = gate * Attn(q=tgt, k=ref, v=ref) + (1-gate) * tgt   (残差门控，稳定训练)
"""
import torch
import torch.nn as nn


class CrossAttnAlign(nn.Module):
    def __init__(self, feat_dim, num_heads=4, use_gate=True, dropout=0.0):
        super().__init__()
        self.feat = feat_dim
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.q = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
        self.k = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
        self.v = nn.Conv2d(feat_dim, feat_dim, 1, bias=False)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.proj = nn.Conv2d(feat_dim, feat_dim, 1)
        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim, 1), nn.Sigmoid())

    def forward(self, ref_feat, tgt_feat):
        """对齐 tgt_feat 到 ref_feat。
        ref_feat: [B, C, H, W] 参考特征（HR 或 frame0）
        tgt_feat: [B, C, H, W] 待对齐目标特征（LR frame t）
        return:   [B, C, H, W] 对齐后的 tgt 特征
        """
        B, C, H, W = tgt_feat.shape
        q = self.q(tgt_feat).flatten(2).transpose(1, 2)   # [B, N, C]
        k = self.k(ref_feat).flatten(2).transpose(1, 2)   # [B, N, C]
        v = self.v(ref_feat).flatten(2).transpose(1, 2)   # [B, N, C]
        out, _ = self.attn(q, k, v)                       # [B, N, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        out = self.proj(out)
        if self.use_gate:
            g = self.gate(tgt_feat)
            aligned = tgt_feat * (1.0 - g) + out * g
        else:
            aligned = tgt_feat + out
        return aligned
