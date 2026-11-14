#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨数据集"数据层"审计（P1）——不训练、不需 GPU，纯数据属性量化。

目的：把 M1–M5 机制在三个真实数据集上做解耦对照。

  数据集              传感器      时相      尺度     隔离的变量
  ------------------------------------------------------------------
  PROBA-V             同传感器    多帧(9-35)  3x     无跨传感器辐射差 -> 纯多时相融合
  SEN2VENµS           跨传感器    同日单帧    2x     无时相内容差   -> 纯跨传感器失配
  WorldStrat          跨传感器    多帧(5)     6.63x  三者叠加       -> 主战场

每个数据集量化：
  cc_raw      CC(bicubic(LR_anchor), HR)        配对完整性（G1 同源指标）
  psnr_mean   常数图（HR 均值）预测的 PSNR      → 无技能地板
  psnr_bic    双三次上采样的 PSNR              → 最朴素地板
  psnr_aff    双三次 + 数据集级仿射标定的 PSNR → 剥离纯辐射差后的地板
  slope/R²    HR ≈ slope*bic + intercept 的全局拟合
  resid_frac  ||HR-bic||² / ||HR-mean(HR)||²   → 待恢复的信息缺口占比
  temp_var    多帧时相间的相对变化幅度（单帧数据集 = nan）

用法：
  python code/audit_datasets.py --n 30 --out experiments/dataset_audit.json
  python code/audit_datasets.py --datasets probav sen2venus --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
from utils.paths import rebase_manifest  # noqa: E402

import _dll_guard  # noqa: E402
_dll_guard.hide()   # 必须在任何 OpenMP 相关库之前（torch/cv2）

import cv2  # noqa: E402


# ------------------------------------------------------------------ 指标

def _flat(*arrs, mask=None):
    """按掩膜摊平成一维 float64（跨通道拼接）。"""
    out = []
    for a in arrs:
        a = np.asarray(a, dtype=np.float64)
        if mask is not None:
            a = a[mask]
        out.append(a.ravel())
    return out


def pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def psnr(pred, gt, data_range):
    mse = float(np.mean((pred - gt) ** 2))
    return float(10 * np.log10(data_range ** 2 / max(mse, 1e-20)))


def bicubic_up(lr, size):
    """lr (C,h,w) 或 (h,w) -> (C,H,W)；cv2 INTER_CUBIC。"""
    if lr.ndim == 2:
        return cv2.resize(lr, size[::-1], interpolation=cv2.INTER_CUBIC)[None]
    return np.stack([cv2.resize(c, size[::-1], interpolation=cv2.INTER_CUBIC) for c in lr])


# ------------------------------------------------------------------ Loaders

def make_probav():
    from datasets_external import ProbaVDataset
    root = os.path.join(BASE, "data", "external", "probav", "probav_data")
    if not os.path.isdir(root):
        root = os.path.join(BASE, "data", "external", "probav")
    ds = ProbaVDataset(root, band="RED")
    return ds, dict(scale=3, sensor="同传感器(PROBA-V)", temporal="多帧", n=len(ds))


def make_sen2venus():
    from datasets_external import Sen2VenusDataset
    root = os.path.join(BASE, "data", "external", "sen2venus")
    ds = Sen2VenusDataset(root)
    return ds, dict(scale=ds.scale, sensor="跨传感器(S2→VENµS)", temporal="同日单帧", n=len(ds))


def make_worldstrat():
    import _dll_guard
    _dll_guard.hide()          # 必须早于 import torch
    sys.path.insert(0, os.path.join(BASE, "code"))
    from data.dataset import WorldStratDataset
    mf = os.path.join(BASE, "data", "processed", "worldstrat_manifest.json")
    with open(mf) as f:
        manifest = json.load(f)
    manifest = rebase_manifest(manifest)   # 发布版清单为相对路径，按数据根补全
    manifest = manifest[:600]
    ds = WorldStratDataset(manifest, scale=7, lr_size=64, T=5,
                           hr_bands=(0, 1, 2), lr_bands=(3, 2, 1),
                           deterministic=True, hr_per_lr=6.63,
                           hr_div=10000.0, lr_calib=[0.68586, 0.001835])
    return ds, dict(scale=6.63, sensor="跨传感器(S2→SPOT)", temporal="多帧", n=len(ds))


LOADERS = {
    "probav": make_probav,
    "sen2venus": make_sen2venus,
    "worldstrat": make_worldstrat,
}


def fetch(ds, i):
    """统一成 dict(lrs=..., hr=..., mask=..., name=...)。"""
    s = ds.load(i) if hasattr(ds, "load") else ds[i]
    if not isinstance(s, dict):                       # torch Dataset 返回元组
        lrs, hr = s[0], s[1]
        s = {"lrs": lrs, "hr": hr, "name": str(i)}
    # WorldStratDataset 用 target/hr_ref 命名（torch 张量）
    if "hr" not in s:
        s["hr"] = s.get("target", s.get("hr_ref"))
    to_np = lambda t: (t.detach().cpu().numpy() if hasattr(t, "detach")   # noqa: E731
                       else np.asarray(t)).astype(np.float32)
    s["lrs"], s["hr"] = to_np(s["lrs"]), to_np(s["hr"])
    if s["lrs"].ndim == 3:                      # (C,h,w) 单帧多通道 -> (1,C,h,w)
        s["lrs"] = s["lrs"][None]
    if s["hr"].ndim == 2:                       # (H,W) 单通道 -> (1,H,W)
        s["hr"] = s["hr"][None]
    if s.get("sm") is not None:
        s["sm"] = np.asarray(s["sm"])
        if s["sm"].ndim == 2:
            s["sm"] = s["sm"][None]
    return s


# ------------------------------------------------------------------ 审计主流程

def audit_one(name, ds, meta, n=30, seed=0):
    rng = np.random.default_rng(seed)
    n = min(n, len(ds))
    idxs = rng.choice(len(ds), size=n, replace=False) if n < len(ds) else np.arange(n)

    ccs, p_bic, p_mean, resid = [], [], [], []
    xs_all, ys_all = [], []
    tvars, sizes = [], []
    p_aff_ps = []          # per-scene 仿射地板（折半像素外样本）

    for i in idxs:
        s = fetch(ds, int(i))
        lrs, hr = np.asarray(s["lrs"]), np.asarray(s["hr"])
        mask = s.get("sm", s.get("mask", None))
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape[-2:] != hr.shape[-2:]:
                mask = None
        if mask is not None and mask.size and mask.mean() < 0.05:
            continue

        # anchor: PROBA-V 取 clearance 最高帧，其余取 frame0
        if lrs.ndim == 3 and "qms" in s:
            clr = np.asarray(s["qms"]).reshape(len(lrs), -1).mean(1)
            a = int(np.argmax(clr))
        else:
            a = 0
        lr0 = lrs[a]

        H, W = hr.shape[-2], hr.shape[-1]
        bic = bicubic_up(lr0, (H, W))
        if lrs.ndim >= 3 and lrs.shape[0] > 1:      # (T,C,h,w) 或 (T,h,w)
            tv = float(np.abs(lrs - lrs[a:a + 1]).mean() / (lrs.std() + 1e-8))
            tvars.append(tv)

        gtf = hr.astype(np.float64)
        bicf = bic.astype(np.float64)
        if mask is not None:
            m = np.broadcast_to(mask.astype(bool), gtf.shape)
            gv, bv = gtf[m], bicf[m]
        else:
            gv, bv = gtf.ravel(), bicf.ravel()
        if gv.size < 100:
            continue

        ccs.append(pearson(bv, gv))
        dr = float(gtf.max() - gtf.min()) or 1.0
        p_bic.append(psnr(bv, gv, dr))
        p_mean.append(psnr(np.full_like(gv, gv.mean()), gv, dr))
        resid.append(float(((gv - bv) ** 2).sum() / max(((gv - gv.mean()) ** 2).sum(), 1e-20)))
        xs_all.append(bv[:: max(1, bv.size // 20000)])
        ys_all.append(gv[:: max(1, gv.size // 20000)])
        sizes.append((lr0.shape[-1], H))

        # --- per-scene 仿射地板 ---
        # 全局（跨场景）仿射在多时相数据集上不成立：PROBA-V 各场景 LR/HR 比值在
        # 0.77~1.32 间波动，少数"时相剧变"场景（方差大且近乎不相关）会主导最小二乘，
        # 把全局 slope 压到 0.155（毫无意义）。故逐场景拟合，并用折半像素外样本评估。
        nn = bv.size
        if nn >= 200:
            perm = rng.permutation(nn)
            h1, h2 = perm[: nn // 2], perm[nn // 2:]

            def _fit_eval(tr, te):
                Aa = np.stack([bv[tr], np.ones(tr.size)], 1)
                c, *_ = np.linalg.lstsq(Aa, gv[tr], rcond=None)
                return psnr(c[0] * bv[te] + c[1], gv[te], dr)

            p_aff_ps.append((_fit_eval(h1, h2) + _fit_eval(h2, h1)) / 2)

    # --- 全局仿射（HR ≈ slope·bic + intercept）：报告 slope/R² ---
    xs = np.concatenate(xs_all)
    ys = np.concatenate(ys_all)
    A = np.stack([xs, np.ones_like(xs)], 1)
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    pred = slope * xs + intercept
    r2 = float(1 - ((ys - pred) ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-20))
    dr_all = float(ys.max() - ys.min()) or 1.0

    # 全局仿射地板：同样用"逐场景 PSNR 的中位数"评估，才能与逐场景仿射比较。
    # （若用全像素聚合 MSE，会被多数简单场景主导，与中位数口径不可比——
    #   实测同一批数据上两种口径能差 4 dB，直接比较会得出相反结论。）
    p_aff = []
    for xv, yv in zip(xs_all, ys_all):
        if xv.size < 100:
            continue
        perm = np.random.default_rng(seed + 1).permutation(xv.size)
        tr, te = perm[: xv.size // 2], perm[xv.size // 2:]
        Aa = np.stack([xv[tr], np.ones(tr.size)], 1)
        c, *_ = np.linalg.lstsq(Aa, yv[tr], rcond=None)     # 全局参数只用于对比
        pred_g = slope * xv[te] + intercept                 # 全局
        p_aff.append(psnr(pred_g, yv[te], float(yv.max() - yv.min()) or 1.0))
    psnr_aff_oos = float(np.median(p_aff)) if p_aff else float("nan")
    psnr_aff_in = float(psnr(pred, ys, dr_all))

    return {
        "dataset": name,
        **meta,
        "n_audited": len(ccs),
        "lr_hw": f"{sizes[0][0]}→{sizes[0][1]}" if sizes else "-",
        "cc_raw": float(np.median(ccs)),
        "cc_mean": float(np.mean(ccs)),
        "psnr_mean_const": float(np.median(p_mean)),
        "psnr_bicubic": float(np.median(p_bic)),
        "psnr_affine": psnr_aff_oos,
        "psnr_affine_insample": psnr_aff_in,
        "psnr_affine_perscene": float(np.median(p_aff_ps)) if p_aff_ps else float("nan"),
        "affine_slope": slope,
        "affine_intercept": intercept,
        "affine_R2": r2,
        "resid_frac": float(np.median(resid)),
        "temporal_var": float(np.median(tvars)) if tvars else float("nan"),
        "bic_gain_over_const": float(np.median(p_bic) - np.median(p_mean)),
        "aff_gain_over_bic": float(psnr_aff_oos - np.median(p_bic)),
    }


def to_markdown(rows):
    head = ["数据集", "传感器", "时相", "尺度", "LR→HR", "CC↑",
            "PSNR常数", "PSNR双三次", "PSNR仿射(逐场景)", "PSNR仿射(全局)",
            "slope", "R²", "残差占比", "时相变异"]
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        d = dict(r)
        tv = r["temporal_var"]
        d["temporal_var"] = "-" if tv != tv else f"{tv:.3f}"
        out.append("| {dataset} | {sensor} | {temporal} | {scale} | {lr_hw} | "
                   "{cc_raw:.3f} | {psnr_mean_const:.2f} | {psnr_bicubic:.2f} | "
                   "{psnr_affine_perscene:.2f} | {psnr_affine:.2f} | "
                   "{affine_slope:.3f} | {affine_R2:.3f} | "
                   "{resid_frac:.3f} | {temporal_var} |".format(**d))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="experiments/dataset_audit.json")
    ap.add_argument("--md", default="docs/跨数据集审计.md")
    args = ap.parse_args()

    rows = []
    for name in args.datasets:
        if name not in LOADERS:
            print(f"[skip] 未知数据集 {name}")
            continue
        try:
            ds, meta = LOADERS[name]()
        except Exception as e:
            print(f"[skip] {name} 加载失败: {type(e).__name__}: {e}")
            continue
        print(f"[audit] {name}  n={min(args.n, len(ds))}/{len(ds)} ...", flush=True)
        try:
            r = audit_one(name, ds, meta, n=args.n)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[fail] {name}: {e}")
            continue
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)

    if not rows:
        print("无结果")
        return
    for p in (args.out, args.md):
        fp = os.path.join(BASE, p)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(os.path.join(BASE, args.out), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    md = ["# 跨数据集数据层审计（P1）", "",
          "指标定义见 docs/SR-Audit协议规范.md；本表为**零训练**的数据属性基线。", "",
          to_markdown(rows), ""]
    with open(os.path.join(BASE, args.md), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + to_markdown(rows))
    print(f"\n[saved] {args.out} / {args.md}")


if __name__ == "__main__":
    main()
