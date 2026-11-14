#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-1 真实三臂 SR 生成：在 EuroSAT RGB (64x64) 上训练 EDSR(臂② 未审计失败 SOTA)
与 本文方法 SISR(臂③ 审计通过法)，产出下游验证所需的 SR .npy。

- 臂② = EDSR (Lim et al. 2017)：通用单图 SISR，本论文审计在 6.63x 跨传感器下判其 floor-failure。
- 臂③ = 本文方法 SpatioTemporalSR（in_ch=3, T=1, scale=4，单图 SISR 模式）。

训练目标：EuroSAT HR(64x64) -> bicubic ×4 下采样 LR(16x16) -> SR 重建 64x64。
下游验证时 harness 用同一 data_root + --limit_per_class 加载 HR，构造 baseline(bicubic) 臂，
本脚本产出 sr_edsr / sr_ours 两臂 .npy（与 harness 加载顺序严格一致）。

运行：
  python code/p2_eurosat_sr.py --method both --scale 4 --train_per_class 150 \
      --test_per_class 30 --epochs 40 --device cuda \
      --out_dir experiments/eurosat_sr
"""
import argparse
import atexit
import glob
import os
import sys
import time

# ---------------------------------------------------------------- vcomp140 guard
_DLL = r"C:\Windows\System32\vcomp140.dll"
_HIDDEN = _DLL + ".hide"


def _hide_dll():
    if os.path.exists(_HIDDEN) and not os.path.exists(_DLL):
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            pass
    if os.path.exists(_DLL):
        try:
            os.rename(_DLL, _HIDDEN)
            return True
        except Exception:
            print("[warn] could not hide vcomp140.dll; torch CUDA may segfault.")
    return False


def _restore_dll():
    if os.path.exists(_HIDDEN):
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            print("[warn] could not restore vcomp140.dll; rename manually.")


_hide_dll()
atexit.register(_restore_dll)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(BASE, "code")
sys.path.insert(0, CODE)
from models.sr_model import SpatioTemporalSR  # 本文方法


# ---------------------------------------------------------------- EDSR (臂②)
class _ResBlock(nn.Module):
    def __init__(self, n=64):
        super().__init__()
        self.c1 = nn.Conv2d(n, n, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(n, n, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.c2(self.relu(self.c1(x))) + x


class EDSR(nn.Module):
    """单图 SISR RGB，输入 [B,3,16,16] -> 输出 [B,3,64,64] (scale=4)。"""

    def __init__(self, scale=4, n_feats=64, n_blocks=16, out_size=64):
        super().__init__()
        self.out_size = (out_size, out_size)
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1, bias=False)
        self.body = nn.Sequential(*[_ResBlock(n_feats) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False)
        self.up = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, 1, 1, bias=False),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        x0 = self.head(x)
        x = self.tail(self.body(x0)) + x0
        x = self.up(x)
        if tuple(x.shape[-2:]) != tuple(self.out_size):
            x = F.interpolate(x, size=tuple(self.out_size), mode="bilinear",
                              align_corners=True, antialias=True)
        return x


# ---------------------------------------------------------------- 数据/LR 构造
def make_lr(hr, scale):
    """bicubic ×scale 下采样 [N,3,H,W] -> [N,3,H//s,W//s]。"""
    return F.interpolate(hr, size=(hr.shape[2] // scale, hr.shape[3] // scale),
                         mode="bicubic", align_corners=True)


def load_hr(per_class, scale=4, seed=0):
    from p2_downstream_verify import load_labeled_eo
    imgs, labels = load_labeled_eo(
        os.path.join(BASE, "data/eurosat_rgb/ImageFolder"),
        image_size=64, limit_per_class=per_class)
    return imgs, labels


def build_model(method, scale, out_size=64):
    if method == "edsr":
        return EDSR(scale=scale, n_feats=64, n_blocks=16, out_size=out_size)
    else:  # ours
        return SpatioTemporalSR(in_ch=3, feat=48, scale=scale, T=1,
                                rsa_radius=3, n_groups=3, n_blocks=3,
                                use_hr_ref=False, use_conf=True,
                                align_mode="cross_attn", align_ref="frame0",
                                out_size=None, upsampler="pixelshuffle",
                                residual_base=False)


def _to_sr(model, method, lr_t, device):
    model.eval()
    with torch.no_grad():
        if method == "edsr":
            sr = model(lr_t)
        else:
            sr = model(lr_t.unsqueeze(1))  # [B,1,3,h,w] T=1
        if tuple(sr.shape[-2:]) != (64, 64):
            sr = F.interpolate(sr, size=(64, 64), mode="bilinear",
                               align_corners=True, antialias=True)
    return sr.clamp(0, 1).cpu().numpy().astype(np.float32)


def train_one(method, hr, scale, epochs, device, lr=1e-4):
    lr_in = make_lr(torch.from_numpy(hr), scale)
    model = build_model(method, scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ds = TensorDataset(lr_in, torch.from_numpy(hr))
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        t0 = time.time()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if method == "edsr":
                pred = model(x)
            else:
                pred = model(x.unsqueeze(1))
            if tuple(pred.shape[-2:]) != tuple(y.shape[-2:]):
                pred = F.interpolate(pred, size=tuple(y.shape[-2:]),
                                     mode="bilinear", align_corners=True, antialias=True)
            loss = F.l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        print("[%s ep%d] loss=%.4f (%.1fs)" % (method, ep, tot / max(1, nb),
                                               time.time() - t0), flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="both", choices=["edsr", "ours", "both"])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--train_per_class", type=int, default=150)
    ap.add_argument("--test_per_class", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="experiments/eurosat_sr")
    ap.add_argument("--skip_train", action="store_true",
                   help="只生成 .npy（需已有 ckpt）")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    methods = ["edsr", "ours"] if args.method == "both" else [args.method]

    # 训练子集（与下游验证子集严格分离）
    hr_train, _ = load_hr(args.train_per_class)
    # 下游验证子集：harness 用 --limit_per_class TEST 加载，这里用完全相同参数加载以保序一致
    hr_test, _ = load_hr(args.test_per_class)
    print("[info] hr_train=%s hr_test=%s device=%s" %
          (hr_train.shape, hr_test.shape, device))

    for method in methods:
        ckpt_path = os.path.join(args.out_dir, "sr_%s_eurosat.pt" % method)
        if not args.skip_train:
            print("=== training %s ===" % method)
            model = train_one(method, hr_train, args.scale, args.epochs, device)
            torch.save({"model": model.state_dict(),
                        "cfg": {"method": method, "scale": args.scale}}, ckpt_path)
            print("[saved] %s" % ckpt_path)
        else:
            model = build_model(method, args.scale).to(device)
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
        # 生成臂 .npy（与 harness 加载顺序一致）
        lr_test = make_lr(torch.from_numpy(hr_test), args.scale)
        sr = _to_sr(model, method, lr_test.to(device), device)
        npy_path = os.path.join(args.out_dir, "sr_%s_eurosat.npy" % method)
        np.save(npy_path, sr)
        print("[saved] %s  shape=%s  range=[%.3f,%.3f]" %
              (npy_path, sr.shape, float(sr.min()), float(sr.max())))


if __name__ == "__main__":
    main()
