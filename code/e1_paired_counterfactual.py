#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主张 B —— 同地点 6.63×【配对反事实】去污实验（回应 CEE MR-2）。

MR-2（审稿人视角复评）："WorldStrat 上 Δ_floor < 0 只是常识——它无非告诉你
SPOT 与 Sentinel-2 是两个不同传感器"。该质疑的实质是：**负 Δ_floor 究竟是
"跨传感器不可恢复"的实质结论，还是"配对影像本身没对齐/没配辐"的方法学伪影？**

本脚本用【同一批 6.63× WorldStrat val 场景】做逐级"去污"反事实：从最脏的
真实跨传感器对，逐级替换掉一个具体污染源，看 Δ_floor 是否单调回升：

  stage 0  real          : 真实 SPOT HR vs 真实 S2 LR（现状基线，预期 ≈ −1.1 dB）
  stage 1  +radiometric  : 用逐通道仿射把 HR 匹配到 LR 的反射率分布（去辐照差）
  stage 2  +reregistered : 再对 HR 做亚像素平移搜索，最小化与 bicubic(LR) 的残差
                           （去空间配准误差；平移在 LR 尺度搜索后映射到 HR）
  stage 3  pure-scale    : 直接用 HR 自身 bicubic 下采样当 LR（去传感器差，纯尺度上界，
                           预期 ≈ +4.7 dB —— 与 e1_scale_isolation_matched 同口径）

判据：若 Δ_floor 随去污单调回升且 stage 3 明确转正，则说明负 Δ_floor **不是**
"常识性不可恢复"，而是"配对数据在辐照/配准上未校正"的可量化后果——**直接终结
MR-2**：跨传感器可恢复预算的崩塌有明确的、可归因的物理来源，不是同义反复。

零训练：只测 data-layer Δ_floor（PSNR(bicubic(LR),GT) − PSNR(mean(GT),GT)），
逐样本配对。口径与 e1_scale_isolation_matched.py / eval_paired_floor.py 一致：
  * per-scene 归一化（LR 侧 (mu,sd) 同变换 HR）；
  * bicubic align_corners=False；
  * data_range = 该批归一化 HR 的全局 min/max。

用法：
  python code/e1_paired_counterfactual.py                 # 默认 val-60，居中 424 窗口
  python code/e1_paired_counterfactual.py --n 60
输出：experiments/e1_paired_counterfactual.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import _dll_guard  # noqa: E402
_dll_guard.hide()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from data.dataset import get_dataset  # noqa: E402
from eval_metrics import calc_psnr  # noqa: E402
from eval_paired_floor import load_cfg  # noqa: E402

CFG_PATH = os.path.join(BASE, "code", "configs", "external_worldstrat.py")
OUT_JSON = os.path.join(BASE, "experiments", "e1_paired_counterfactual.json")
CROP = 424
STAGES = ["real", "radiometric", "reregistered", "pure_scale"]


def bicubic_up(x, size):
    return F.interpolate(x, size=size, mode="bicubic", align_corners=False)


def match_radiometry(hr, lr_up):
    """逐通道 affine (a,b) 使 HR 的均值/方差匹配 bicubic(LR)。

    只做逐通道线性重标（增益+偏置），不改变 HR 的空间纹理——因此若 Δ_floor 因此
    回升，说明崩塌的贡献里有一部分纯属辐射量纲不一致，与"跨传感器不可恢复"无关。
    """
    out = torch.empty_like(hr)
    for c in range(hr.shape[0]):
        h = hr[c]
        t = lr_up[c]
        hm, hs = h.mean(), h.std() + 1e-6
        tm, ts = t.mean(), t.std() + 1e-6
        out[c] = (h - hm) / hs * ts + tm
    return out


def _shift_img(x, dy, dx):
    """整像素平移（HR 尺度），用 replicate 填充边界——**不能用 torch.roll**
    （roll 会环绕回卷，让大位移借绕回内容伪造低残差，制造假配准）。"""
    if dy == 0 and dx == 0:
        return x
    h, w = x.shape[-2], x.shape[-1]
    # 先 pad 再 crop，等价于平移并复制边界
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))
    xp = F.pad(x, pad, mode="replicate")
    ys = 0 if dy >= 0 else -dy
    xs = 0 if dx >= 0 else -dx
    return xp[..., ys:ys + h, xs:xs + w]


def reregister(hr, lr_up, search_lr=4, margin=32):
    """在 LR 尺度做 ±search_lr 整像素平移搜索，映射到 HR，取残差最小者。

    真实 WorldStrat 对的 HR/LR 来自不同日期/传感器，存在残余配准误差。为消除
    边界回卷伪影：① 平移用 replicate 填充；② 残差只在【去掉 margin 的内区】
    上计算（边界像素在平移后不可靠）。
    """
    ratio = hr.shape[-1] / lr_up.shape[-1]
    H, W = hr.shape[-2], hr.shape[-1]
    m = margin
    ref = lr_up[..., m:H - m, m:W - m]
    best_s, best_res = (0, 0), None
    for dy in range(-search_lr, search_lr + 1):
        for dx in range(-search_lr, search_lr + 1):
            syd, sxd = int(round(dy * ratio)), int(round(dx * ratio))
            sh = _shift_img(hr, syd, sxd)
            cand = sh[..., m:H - m, m:W - m]
            res = float(((cand - ref) ** 2).mean())
            if best_res is None or res < best_res:
                best_res, best_s = res, (dy, dx)
    dy, dx = best_s
    return _shift_img(hr, int(round(dy * ratio)), int(round(dx * ratio))), best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--search-lr", type=int, default=10)
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(CFG_PATH)
    full = get_dataset(cfg, project_root=BASE, deterministic=True)
    val_frac = float(cfg.get("val_frac", 0.1))
    n_train = max(1, min(len(full) - 1, int((1.0 - val_frac) * len(full))))
    subset = list(range(n_train, len(full)))[-a.n:]
    loader = DataLoader(Subset(full, subset), batch_size=1, shuffle=False,
                        num_workers=0)
    print(f"[info] n={len(subset)} crop={CROP} search_lr=±{a.search_lr} device={device}")

    rows = {s: [] for s in STAGES}
    regs = []
    # 所有 stage 一律在【同一内区】评估：去掉 MARGIN 边框，消除平移后边界不一致
    MARGIN = int(np.ceil(a.search_lr * 6.63)) + 8
    with torch.no_grad():
        for k, batch in enumerate(loader):
            idx = n_train + len(subset) - a.n + k
            # dataset norm="per_scene"：gt 与 lr 已用该场景同一 (mu,sd) 标准化，
            # 二者本就同处一个归一化空间——直接在该空间内做四 stage 对照即可
            # （与 eval_paired_floor / e1_scale_isolation_matched 的 method 口径一致）。
            gt = batch["target"].float().to(device)          # 归一化 GT [1,3,424,424]
            lr = batch["lrs"][0, 0:1].float().to(device)     # frame0 LR [1,3,64,64]
            lr_up = bicubic_up(lr, (CROP, CROP))             # bicubic(LR) 到 HR 网格

            def inner(t):
                m = MARGIN
                return t[..., m:t.shape[-2] - m, m:t.shape[-1] - m]

            # stage 0: real —— 真实 HR(GT) vs 上采样真实 LR（同一归一化空间）
            rows["real"].append((lr_up, gt))

            # stage 1: radiometric —— GT 逐通道 affine 到 bicubic(LR) 的分布后再评
            gt_rm = match_radiometry(gt[0], lr_up[0])[None]
            rows["radiometric"].append((lr_up, gt_rm))

            # stage 2: reregistered —— 在 stage1 基础上再对 GT 做整像素配准
            gt_rr, sh = reregister(gt_rm[0], lr_up[0], search_lr=a.search_lr)
            regs.append(sh)
            rows["reregistered"].append((lr_up, gt_rr[None]))

            # stage 3: pure_scale —— 用 GT 自身 bicubic 下采样当 LR（纯尺度上界）
            lr_pure = F.interpolate(gt, size=(lr.shape[-2], lr.shape[-1]),
                                    mode="bicubic", align_corners=False,
                                    antialias=True)
            lr_pure_up = bicubic_up(lr_pure, (CROP, CROP))
            rows["pure_scale"].append((lr_pure_up, gt))

            # 统一内区裁剪（四 stage 同区域）
            for s in STAGES:
                up, g = rows[s][-1]
                rows[s][-1] = (inner(up), inner(g))

            if (k + 1) % 10 == 0:
                print(f"  ..{k+1}/{len(subset)}  seg_shift_hist={regs[-5:]}  margin={MARGIN}")

    # ---- 聚合：data_range = 该批归一化 GT 的全局 min/max（与 eval_paired_floor 同式）----
    allgt = torch.cat([r[1].reshape(-1) for r in rows["pure_scale"]])
    dr = float(allgt.max() - allgt.min())
    dr = dr if dr > 1e-8 else 1.0
    print(f"[info] data_range(归一化GT)={dr:.4f}")

    # ---- 逐样本 Δ_floor + 配对 t 检验（vs real）----
    from scipy import stats
    per = {}
    for s in STAGES:
        vals = []
        for up, gt in rows[s]:
            p_bic = float(calc_psnr(up[0], gt[0], data_range=dr))
            mean_img = torch.full_like(gt, float(gt.mean()))
            p_mean = float(calc_psnr(mean_img[0], gt[0], data_range=dr))
            vals.append(p_bic - p_mean)
        per[s] = np.array(vals)

    out = {"n": len(subset), "crop": CROP, "search_lr": a.search_lr,
           "data_range": round(dr, 4), "stages": {}}
    print("\n== 6.63× WorldStrat 配对反事实去污（逐样本 Δ_floor）==")
    print(f"{'stage':<14} {'mean':>8} {'std':>8} {'neg%':>6}")
    base = per["real"]
    for s in STAGES:
        v = per[s]
        t, p = stats.ttest_rel(v, base) if s != "real" else (0.0, 1.0)
        out["stages"][s] = {
            "mean": round(float(v.mean()), 3),
            "std": round(float(v.std(ddof=1)), 3),
            "neg_frac": round(float((v < 0).mean()), 3),
            "delta_vs_real": round(float(v.mean() - base.mean()), 3),
            "paired_p_vs_real": float(p),
            "per_sample": [round(float(x), 4) for x in v],
        }
        print(f"{s:<14} {v.mean():>+8.3f} {v.std(ddof=1):>8.3f} "
              f"{(v<0).mean()*100:>5.0f}%")
    out["shift_hist"] = [[int(x), int(y)] for x, y in regs]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {a.out}")
    print("\n判据：Δ_floor 单调回升 → 负值有明确可归因的污染源，非『常识同义反复』。")


if __name__ == "__main__":
    main()
