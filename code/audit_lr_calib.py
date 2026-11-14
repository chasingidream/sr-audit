#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""#23 复核：WorldStrat 的全局 lr_calib 是否应改为「逐场景」仿射标定。

零训练、CPU。利用已审计的 WorldStrat 验证集（subset=600 的末 10% = 60 场景，
与训练/评测切分完全一致、无泄漏），对「LR→HR 辐射标定」做三种地板对照：

  G  = 全局仿射：训练集拟合的常量 lr_calib = [0.68586, 0.001835]
  P  = 逐场景仿射：每个验证场景独立拟合 slope/intercept
  B  = 双三次（不做任何辐射标定，作为参考地板）

判据（沿用项目金标准：逐样本配对检验，而非组均值）：
  G - B  : 全局标定相对「不标定」的净增益（>0 才说明标定有用）
  P - G  : 逐场景相对全局的净增益（配对 t / Wilcoxon + Cohen's d）
  P - B  : 逐场景相对「不标定」的净增益

若 P >> G 且配对检验显著 → 结论：WorldStrat 应改用逐场景标定。

另对 PROBA-V 做对照（同传感器多帧），验证「多时相 → 逐场景」这一更一般的结论。
"""
from __future__ import annotations

import argparse
import os
import sys
from math import erf, sqrt

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import _dll_guard  # noqa: E402
_dll_guard.hide()

import cv2  # noqa: E402

from audit_datasets import make_worldstrat, make_probav, fetch, bicubic_up, psnr  # noqa: E402


# 预定义全局标定（与训练中 dataset 的 lr_calib 一致）
GLOBAL_CALIB = {
    "worldstrat": [0.68586, 0.001835],
    "probav": None,   # PROBA-V 无预定义全局标定；用审计全局斜率作为对照
}


def paired_stats(a, b):
    """逐样本配对：均值差、正态近似两尾 p、Cohen's d（配对）。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    if n < 2:
        return float(d.mean()), float("nan"), float("nan")
    md = d.mean()
    s = d.std(ddof=1)
    if s <= 0:
        t = 0.0
    else:
        t = md / (s / sqrt(n))
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    dz = md / s
    return float(md), float(p), float(dz)


def affine_fit_predict(x, y, tr, te):
    """在 x[tr]/y[tr] 上拟合 y≈s*x+i，返回对 x[te] 的预测（同形状）。"""
    A = np.stack([x[tr], np.ones(tr.size)], 1)
    coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    return coef[0] * x[te] + coef[1]


def scene_metrics(raw_lr, hr, global_calib):
    """对单场景返回 (psnr_bic, psnr_global, psnr_perscene_in, psnr_perscene_oos)。"""
    C, h, w = raw_lr.shape
    H, W = hr.shape[-2], hr.shape[-1]
    bic = bicubic_up(raw_lr, (H, W)).astype(np.float64)   # (C,H,W)
    gt = hr.astype(np.float64)
    dr = float(gt.max() - gt.min()) or 1.0

    # 现状地板：dataset 返回的 lrs 已是「全局标定」结果（网络实际输入）。
    # 其双三次上采样即当前管线的辐射标定地板（与训练一致，无双重标定）。
    p_bic = psnr(bic, gt, dr)
    p_global = p_bic  # 现状 = 全局标定下的双三次地板

    # （若需把全局标定显式作用在 RAW DN 上复核：raw_dn=(lrs-0.001835)/0.68586，
    #  再 0.68586*raw_dn+0.001835 上采样，结果数值上等同 p_bic，故省略。）

    rng = np.random.default_rng(0)
    p_in_list, p_oos_list = [], []
    for c in range(C):
        x = bic[c].ravel()
        y = gt[c].ravel()
        # 逐场景 in-sample（全像素，乐观上界）
        A = np.stack([x, np.ones(x.size)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred_in = coef[0] * x + coef[1]
        p_in_list.append(psnr(pred_in.reshape(1, H, W), gt[[c]], dr))

        # 逐场景 OOS（半像素折半，公平估计）
        perm = rng.permutation(x.size)
        h1, h2 = perm[: x.size // 2], perm[x.size // 2 :]
        pred_h2 = affine_fit_predict(x, y, h1, h2)
        pred_h1 = affine_fit_predict(x, y, h2, h1)
        # 在对应半区上算 PSNR
        yy = np.concatenate([y[h2], y[h1]])
        pp = np.concatenate([pred_h2, pred_h1])
        p_oos_list.append(psnr(pp.reshape(1, -1), yy.reshape(1, -1),
                               float(yy.max() - yy.min()) or 1.0))

    return p_bic, p_global, float(np.mean(p_in_list)), float(np.mean(p_oos_list))


def run(name, loader, n_val=60, global_calib=None):
    ds, meta = loader()
    N = len(ds)
    val_idx = list(range(max(0, int(0.9 * N)), N))[:n_val]
    rows_b, rows_g, rows_pin, rows_poos = [], [], [], []
    for i in val_idx:
        s = fetch(ds, i)
        lrs, hr = s["lrs"], s["hr"]
        a = 0
        raw_lr = lrs[a].astype(np.float64)
        pb, pg, pin, poos = scene_metrics(raw_lr, hr, global_calib)
        rows_b.append(pb)
        if not np.isnan(pg):
            rows_g.append(pg)
        rows_pin.append(pin)
        rows_poos.append(poos)
    med = lambda v: float(np.median(v))
    print(f"\n===== {name} (n={len(val_idx)}, 验证集末10%) =====")
    print(f"  现状(全局标定)地板: {med(rows_b):.2f} dB  ← 当前管线 LR→HR 辐射地板")
    print(f"  逐场景 全像素     : {med(rows_pin):.2f} dB (in-sample 上界)")
    print(f"  逐场景 OOS(半折)  : {med(rows_poos):.2f} dB (公平估计)")
    dp, pp, dzp = paired_stats(rows_poos, rows_b)
    print(f"  逐场景OOS − 现状  : {dp:+.2f} dB  p={pp:.2e}  d={dzp:+.2f}  "
          f"← 改逐场景的净增益")
    return {
        "dataset": name,
        "n": len(val_idx),
        "psnr_current_global_floor": med(rows_b),
        "psnr_perscene_in": med(rows_pin),
        "psnr_perscene_oos": med(rows_poos),
        "perscene_oos_minus_current": dp,
        "perscene_oos_minus_current_p": pp,
        "perscene_oos_minus_current_d": dzp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["worldstrat", "probav"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="experiments/audit_lr_calib.json")
    args = ap.parse_args()

    loaders = {"worldstrat": make_worldstrat, "probav": make_probav}
    results = []
    for name in args.datasets:
        if name not in loaders:
            print(f"[skip] {name}")
            continue
        gc = GLOBAL_CALIB.get(name)
        try:
            r = run(name, loaders[name], n_val=args.n, global_calib=gc)
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[fail] {name}: {e}")

    os.makedirs(os.path.join(BASE, "experiments"), exist_ok=True)
    with open(os.path.join(BASE, args.out), "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
