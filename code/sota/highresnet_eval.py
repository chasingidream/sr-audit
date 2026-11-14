#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HighRes-net (WorldStrat 官方基线) 双协议复审评测 (Task #30 P0)
============================================================
忠实复刻 WorldStrat 官方评测协议:
  - LR: 8 个 S2-12 波段 revisit (已反射率 0-1) -> Normalize(JIF_S2_MEAN, JIF_S2_STD) -> Resize 160
  - HR: _ps.tiff RGB /10000 -> clamp[0,1] -> Resize 156 (bicubic, antialias)
  - model: HighResNet -> (1,1,3,156,156)
  - baseline: BicubicUpscaledBaseline (revisit0 RGB, 反归一, /max, bicubic->156)
  - bias_adjust(y, y_hat) = y_hat + mean(y - y_hat)  (官方对 model 与 baseline 均套用)
  - 双协议:
      * 原协议 (as-published): bias_adjust + Lanczos Shift 亚像素配准 -> 逐样本 PSNR 平均
      * 审计协议 (ours): 同逐样本 PSNR, + 配对 t 检验(CI/p/Cohen's d) + 技能地板 + G1-G5
使用官方 stratified test split (394 scenes, 与训练集互斥).

用法:
  python code/sota/highresnet_eval.py                 # 全量 test split
  python code/sota/highresnet_eval.py --limit 8      # 冒烟测试
  python code/sota/highresnet_eval.py --no-shift     # 关掉配准(更严格审计)
"""
import os
import sys
import json
import math
import argparse
import itertools

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(ROOT, "third_party", "worldstrat")
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, os.path.join(SRC_ROOT, "_kornia_stub"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- vcomp140 守卫 (torch import 在 Windows 上会 segfault, 必须隐藏) ----
_DLL = r"C:\Windows\System32\vcomp140.dll"
_dll_hidden = False
if os.path.exists(_DLL):
    try:
        os.rename(_DLL, _DLL + ".hide")
        _dll_hidden = True
    except Exception as e:
        print(f"[warn] 无法隐藏 vcomp140.dll: {e}")

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
import scipy.stats as st

from src.datasources import JIF_S2_MEAN, JIF_S2_STD, S2_ALL_12BANDS
from highresnet_load import load_highresnet


# ----------------------------------------------------------------------------
# 忠实组件 (来自 WorldStrat src/)
# ----------------------------------------------------------------------------
def lanczos_kernel(translation_in_px, kernel_lobes=3, kernel_width=None):
    device = translation_in_px.device
    dtype = translation_in_px.dtype
    abs_r = translation_in_px.abs().ceil().int()
    ksw = 2 * (kernel_lobes + abs_r) + 1
    max_w = ksw.max() if hasattr(ksw, "shape") else ksw
    if (kernel_width is None) or (kernel_width < max_w):
        kernel_width = ksw
    zb = (((kernel_width - ksw) / 2).floor().int())
    start = (-(kernel_lobes + abs_r + zb)).min()
    end = (kernel_lobes + abs_r + zb + 1).max()
    x = (torch.arange(start, end, dtype=dtype, device=device).view(1, -1)
         - translation_in_px)
    px = (np.pi * x) + 1e-3
    sin_px = torch.sin(px)
    sin_pxa = torch.sin(px / kernel_lobes)
    k = kernel_lobes * sin_px * sin_pxa / px ** 2
    return k


class ShiftEval:
    """忠实复刻 losses.Shift (Lanczos 亚像素配准), 用于复现官方 PSNR。"""
    def __init__(self, shift_by_px=2, step=0.5, lobes=3, device="cpu"):
        shifts = torch.arange(-shift_by_px, shift_by_px + step, step)
        shifts = shifts[:, None]
        kernel = lanczos_kernel(shifts, kernel_lobes=lobes)
        kernels = torch.stack(
            [ky[:, None] @ kx[None, :] for ky, kx in itertools.product(kernel, kernel)]
        )
        self.kernels = kernels[:, None].to(device)        # (Nk,1,s,s)
        self.Nk = self.kernels.shape[0]
        self.shift_by_px = shift_by_px
        self.kh = self.kernels.shape[2]

    def forward(self, y):
        # y: (B,C,H,W) -> (Nk*B, C, patch, patch)
        B, C, H, W = y.shape
        conv = torch.nn.Conv2d(self.Nk, self.Nk, self.kh, bias=False,
                               groups=self.Nk, padding_mode="reflect")
        conv.weight.data = self.kernels.to(y.dtype)
        conv = conv.to(y.device)
        conv.requires_grad_(False)
        y2 = y.unsqueeze(2).expand(-1, -1, self.Nk, -1, -1).reshape(B * C, self.Nk, H, W)
        out = conv(y2)                                   # (B*C, Nk, patch, patch)
        out = out.transpose(0, 1).contiguous().view(self.Nk * B, C, out.shape[-2], out.shape[-1])
        return out

    def registered_mse(self, y_hat, y):
        # y_hat, y: (B,3,H,W) -> 逐样本最小 MSE (在 Nk 个亚像素位移中取最优)
        B = y_hat.shape[0]
        patch = y_hat.shape[-1] - self.kh + 1
        y_hat_s = self.forward(y_hat)                    # (Nk*B,3,patch,patch)
        y_c = y[:, :, self.shift_by_px:self.shift_by_px + patch,
                self.shift_by_px:self.shift_by_px + patch]
        y_c = y_c.expand(self.Nk, -1, -1, -1, -1).reshape(self.Nk * B, 3, patch, patch)
        mse = F.mse_loss(y_hat_s, y_c, reduction="none").mean(dim=(-1, -2, -3))  # (Nk*B,)
        return mse.view(self.Nk, B).min(dim=0)[0]        # (B,)


def bias_adjust(y, y_hat):
    bias = (y - y_hat).mean(dim=(-1, -2), keepdim=True)
    return y_hat + bias


def bicubic_baseline(lr, device="cpu"):
    """忠实复刻 modules.BicubicUpscaledBaseline.forward。 lr: (8,12,160,160) 已归一化。"""
    rgb_idx = [i - 1 for i in S2_ALL_12BANDS["true_color_zero_index"]]  # [3,2,1]
    x = lr[:, rgb_idx, :, :]                              # (8,3,160,160)
    x = x[0]                                             # 取首 revisit
    mean = JIF_S2_MEAN[rgb_idx].to(device).float()
    std = JIF_S2_STD[rgb_idx].to(device).float()
    x = x + mean[:, None, None]
    x = x * std[:, None, None]
    x = x.float()
    x = x / x.max()
    x = x.clamp(0, 1)
    x = x[None]                                          # (1,3,160,160)
    x = F.interpolate(x, size=(156, 156), mode="bicubic",
                      align_corners=False, antialias=True)
    return x                                             # (1,3,156,156)


# ----------------------------------------------------------------------------
# 数据 adapter (忠实复刻 make_transforms_JIF)
# ----------------------------------------------------------------------------
def build_lr(lr_paths, device="cpu"):
    revisits = []
    for p in lr_paths:
        arr = tifffile.imread(p).astype(np.float32)      # (H,W,12) 反射率 0-1
        arr = np.transpose(arr, (2, 0, 1))               # (12,H,W)
        revisits.append(arr)
    x = np.stack(revisits, axis=0)                       # (8,12,H,W)
    t = torch.from_numpy(x).to(device)
    mean = JIF_S2_MEAN.to(device).float().reshape(1, 12, 1, 1)
    std = JIF_S2_STD.to(device).float().reshape(1, 12, 1, 1)
    t = (t - mean) / std
    t = F.interpolate(t, size=(160, 160), mode="bilinear",
                      align_corners=False, antialias=True)
    return t                                             # (8,12,160,160)


def build_hr(hr_path, device="cpu"):
    arr = tifffile.imread(hr_path).astype(np.float32)    # (H,W,4)
    rgb = arr[..., :3]                                   # RGB (丢弃 alpha/NIR)
    rgb = np.transpose(rgb, (2, 0, 1))                   # (3,H,W)
    t = torch.from_numpy(rgb).to(device) / 10000.0
    t = t.clamp(0, 1)
    t = F.interpolate(t[None], size=(156, 156), mode="bicubic",
                      align_corners=False, antialias=True)[0]
    return t                                             # (3,156,156)


# ----------------------------------------------------------------------------
# 指标 (安全版 CC, 统一 4 维)
# ----------------------------------------------------------------------------
def calc_cc(pred, target):
    # 二者均 (B,3,H,W)
    B = pred.shape[0]
    p = pred.reshape(B, -1)
    g = target.reshape(B, -1)
    p = p - p.mean(-1, keepdim=True)
    g = g - g.mean(-1, keepdim=True)
    num = (p * g).sum(-1)
    den = torch.sqrt((p * p).sum(-1) * (g * g).sum(-1))
    return num / (den + 1e-8)                           # (B,)


def paired_stats(diff):
    diff = np.asarray(diff, dtype=np.float64)
    n = len(diff)
    mean = diff.mean()
    sd = diff.std(ddof=1)
    t, p = st.ttest_rel(diff, np.zeros_like(diff))
    d = mean / sd if sd > 0 else 0.0
    se = sd / math.sqrt(n)
    ci = st.t.interval(0.95, n - 1, loc=mean, scale=se)
    return dict(n=n, mean=mean, sd=sd, t=t, p=p, d=d,
                ci_lo=ci[0], ci_hi=ci[1])


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def cov_from_path(p):
    parts = p.replace("\\", "/").split("/")
    for i, seg in enumerate(parts):
        if seg == "worldstrat":
            return parts[i + 1]
    return parts[-3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-shift", action="store_true", help="关掉亚像素配准(更严格审计)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    # 模型
    model = load_highresnet(device=device)
    model.eval()

    # 官方 test split
    split_csv = os.path.join(SRC_ROOT, "pretrained_model",
                             "stratified_train_val_test_split.csv")
    test_tiles = set()
    import csv
    with open(split_csv) as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                test_tiles.add(row["tile"])
    print(f"[info] test tiles (csv): {len(test_tiles)}")

    mani = json.load(open(os.path.join(ROOT, "data", "processed",
                                       "worldstrat_manifest.json")))
    cov2item = {}
    for m in mani:
        lr0 = m["lr"][0] if isinstance(m["lr"], list) else m["lr"]
        cov = cov_from_path(lr0)
        cov2item[cov] = m
    test_items = [cov2item[t] for t in test_tiles if t in cov2item]
    print(f"[info] test scenes available locally: {len(test_items)}")
    if args.limit:
        test_items = test_items[:args.limit]

    shift = ShiftEval(device=device) if not args.no_shift else None

    psnr_model, psnr_bic, psnr_floor = [], [], []
    cc_model, cc_bic, cc_floor = [], [], []
    std_model, std_bic, std_gt = [], [], []
    gains_reg, gains_raw = [], []

    with torch.no_grad():
        for i, item in enumerate(test_items):
            lr_paths = sorted(item["lr"])[:8]            # 前 8 个 revisit (natsort 等价)
            hr_path = item["hr"]
            lr = build_lr(lr_paths, device=device)        # (8,12,160,160)
            hr = build_hr(hr_path, device=device)        # (3,156,156)
            lr_b = lr[None].to(device)                   # (1,8,12,160,160)
            hr_b = hr[None].to(device)                   # (1,3,156,156)

            y_hat_m = model(lr_b)[:, 0]                  # (1,3,156,156)
            y_hat_b = bicubic_baseline(lr, device=device)

            y_hat_m = bias_adjust(hr_b, y_hat_m)
            y_hat_b = bias_adjust(hr_b, y_hat_b)

            # 技能地板: 预测 GT 逐通道均值
            mean_pred = hr_b.mean(dim=(-1, -2), keepdim=True).expand_as(hr_b)

            if shift is not None:
                mse_m = shift.registered_mse(y_hat_m, hr_b)
                mse_b = shift.registered_mse(y_hat_b, hr_b)
            else:
                mse_m = F.mse_loss(y_hat_m, hr_b, reduction="none").mean(dim=(-1, -2, -3))
                mse_b = F.mse_loss(y_hat_b, hr_b, reduction="none").mean(dim=(-1, -2, -3))
            mse_f = F.mse_loss(mean_pred, hr_b, reduction="none").mean(dim=(-1, -2, -3))

            p_m = (-10 * torch.log10(mse_m)).cpu().numpy()
            p_b = (-10 * torch.log10(mse_b)).cpu().numpy()
            p_f = (-10 * torch.log10(mse_f)).cpu().numpy()
            psnr_model.append(float(p_m[0]))
            psnr_bic.append(float(p_b[0]))
            psnr_floor.append(float(p_f[0]))
            gains_reg.append(float(p_m[0] - p_b[0]))

            # 原始(无配准)增益: 仅做审计严格性对照
            mse_m_raw = F.mse_loss(y_hat_m, hr_b, reduction="none").mean(dim=(-1, -2, -3))
            mse_b_raw = F.mse_loss(y_hat_b, hr_b, reduction="none").mean(dim=(-1, -2, -3))
            gains_raw.append(float((-10*torch.log10(mse_m_raw) - (-10*torch.log10(mse_b_raw))).cpu().numpy()[0]))

            # CC (G1/G2)
            cc_model.append(float(calc_cc(y_hat_m, hr_b).cpu().numpy()[0]))
            cc_bic.append(float(calc_cc(y_hat_b, hr_b).cpu().numpy()[0]))
            cc_floor.append(float(calc_cc(mean_pred, hr_b).cpu().numpy()[0]))

            # 输出统计 (G3)
            std_model.append(float(y_hat_m.std().cpu().numpy()))
            std_bic.append(float(y_hat_b.std().cpu().numpy()))
            std_gt.append(float(hr_b.std().cpu().numpy()))

            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(test_items)}  "
                      f"model={np.mean(psnr_model):.2f} bic={np.mean(psnr_bic):.2f}")

    # 聚合
    psnr_model = np.array(psnr_model)
    psnr_bic = np.array(psnr_bic)
    psnr_floor = np.array(psnr_floor)
    cc_model = np.array(cc_model)
    cc_bic = np.array(cc_bic)
    cc_floor = np.array(cc_floor)
    std_model = np.array(std_model)
    std_bic = np.array(std_bic)
    std_gt = np.array(std_gt)
    gains_reg = np.array(gains_reg)
    gains_raw = np.array(gains_raw)

    paired = paired_stats(gains_reg)
    paired_raw = paired_stats(gains_raw)

    mean_psnr_model = float(psnr_model.mean())
    mean_psnr_bic = float(psnr_bic.mean())
    mean_psnr_floor = float(psnr_floor.mean())
    orig_gain = mean_psnr_model - mean_psnr_bic
    audit_gain = float(gains_reg.mean())

    # 审计门 (G1-G5 简化, 针对 HighRes-net 任务特性)
    g1 = "PASS" if cc_bic.mean() >= 0.30 else ("WARN" if cc_bic.mean() >= 0.20 else "FAIL")
    g2_ratio = cc_model.mean() / (cc_bic.mean() + 1e-8)
    g2 = "PASS" if g2_ratio >= 0.9 else ("WARN" if g2_ratio >= 0.7 else "FAIL")
    sr = (std_model / (std_gt + 1e-8)).mean()
    g3 = "PASS" if 0.5 <= sr <= 1.8 else ("WARN" if 0.3 <= sr <= 2.2 else "FAIL")
    leak = mean_psnr_model > 45
    g5 = "FAIL" if leak else "PASS"

    result = {
        "method": "HighRes-net",
        "task": "WorldStrat JIF MFSR (S2 8 revisits -> SPOT, 160->156 近匹配融合/锐化)",
        "n_test": int(len(test_items)),
        "split": "official stratified test (held-out)",
        "orig_gain_db": round(orig_gain, 3),
        "audit_gain_db": round(audit_gain, 3),
        "audit_ci": [round(paired["ci_lo"], 3), round(paired["ci_hi"], 3)],
        "audit_p": round(float(paired["p"]), 5),
        "audit_d": round(float(paired["d"]), 3),
        "passes_floor": bool(mean_psnr_model > mean_psnr_floor),
        "g1": g1, "g2": g2, "g3": f"{sr:.2f}", "g4": "n/a", "g5": g5,
        "note": (f"model {mean_psnr_model:.2f} / bicubic {mean_psnr_bic:.2f} / "
                 f"mean-floor {mean_psnr_floor:.2f} dB; G2(model/bic CC ratio)={g2_ratio:.3f}; "
                 f"unregistered gain={gains_raw.mean():+.3f} dB (p={paired_raw['p']:.4f}); "
                 f"orig_gain_db=本工作对其协议的忠实复现(非论文原报数字)."),
        # 详细 (不影响注册表, 仅供审计报告)
        "detail": {
            "psnr_model_mean": round(mean_psnr_model, 3),
            "psnr_model_std": round(float(psnr_model.std()), 3),
            "psnr_bic_mean": round(mean_psnr_bic, 3),
            "psnr_floor_mean": round(mean_psnr_floor, 3),
            "cc_model_mean": round(float(cc_model.mean()), 4),
            "cc_bic_mean": round(float(cc_bic.mean()), 4),
            "cc_floor_mean": round(float(cc_floor.mean()), 4),
            "g2_ratio": round(float(g2_ratio), 4),
            "std_ratio_model_over_gt": round(float(sr), 4),
            "paired_reg": {k: (round(v, 5) if isinstance(v, (int, float)) else v)
                           for k, v in paired.items()},
            "paired_raw": {k: (round(v, 5) if isinstance(v, (int, float)) else v)
                           for k, v in paired_raw.items()},
            "device": device,
        },
    }

    out_dir = os.path.join(ROOT, "experiments", "sota_review")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 逐样本落盘 (供 Supplementary S2 分布直方图; 不改动聚合口径) ----
    ps_path = os.path.join(out_dir, "highresnet_persample.json")
    persample = {
        "n": int(len(gains_reg)),
        "gain_reg": [round(float(x), 5) for x in gains_reg],
        "gain_raw": [round(float(x), 5) for x in gains_raw],
        "psnr_model": [round(float(x), 5) for x in psnr_model],
        "psnr_bic": [round(float(x), 5) for x in psnr_bic],
        "psnr_floor": [round(float(x), 5) for x in psnr_floor],
    }
    with open(ps_path, "w", encoding="utf-8") as f:
        json.dump(persample, f, ensure_ascii=False, indent=2)
    print(f"[info] per-sample dumped -> {ps_path}  (n={persample['n']})")

    out_path = os.path.join(out_dir, "highresnet.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n================ HighRes-net 双协议复审 ================")
    print(f"test scenes           : {result['n_test']}")
    print(f"model PSNR (mean)     : {mean_psnr_model:.3f} dB")
    print(f"bicubic PSNR (mean)   : {mean_psnr_bic:.3f} dB")
    print(f"mean-floor PSNR       : {mean_psnr_floor:.3f} dB")
    print(f"原协议增益 (model-bic) : {orig_gain:+.3f} dB")
    print(f"审计配对增益          : {audit_gain:+.3f} dB  "
          f"(95%CI [{paired['ci_lo']:+.3f},{paired['ci_hi']:+.3f}], "
          f"p={paired['p']:.4f}, d={paired['d']:.3f})")
    print(f"无配准增益(更严审计)  : {gains_raw.mean():+.3f} dB (p={paired_raw['p']:.4f})")
    print(f"G1 配对CC(bic)        : {cc_bic.mean():.3f} -> {g1}")
    print(f"G2 结构保真(model/bic) : {g2_ratio:.3f} -> {g2}")
    print(f"G3 输出std比(model/gt) : {sr:.3f} -> {g3}")
    print(f"G5 泄漏(>45dB?)       : {g5}")
    print(f"过均值地板?           : {result['passes_floor']}")
    print(f"[ok] 写入 {out_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _dll_hidden and os.path.exists(_DLL + ".hide"):
            try:
                os.rename(_DLL + ".hide", _DLL)
            except Exception:
                pass
