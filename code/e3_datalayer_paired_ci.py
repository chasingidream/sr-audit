#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E3 —— 数据层 `Δ_floor` 的逐样本配对置信区间（零训练）。

背景（`docs/尺度信息缺口假说.md` §10-5）：
  `dataset_audit_*.json` 只报告 `median(PSNR_bic) − median(PSNR_mean)` 的点值，
  未做配对显著性。E3 补上：对**同一批样本**逐样本算
      Δ_i = PSNR(bicubic(LR_i), GT_i) − PSNR(mean(GT_i), GT_i)
  再做配对 t 检验，给出 mean / 95%CI / p / Cohen's d / 中位数。

关键性质：Δ_floor = 10·log10( MSE_mean / MSE_bic )，与 data_range 无关
（两个 PSNR 用同一 dr，dr 在相减中约掉）→ 可直接与三轴方法层 Δ_floor 对照。

样本集：与三轴方法层**同一切分**（数据集末 val_frac 比例，deterministic），
使"数据层 Δ_floor"与"方法层 Δ_floor"在同一批样本上可配对比较。

用法：
  python code/e3_datalayer_paired_ci.py                       # 三轴全跑
  python code/e3_datalayer_paired_ci.py --datasets probav sen2venus
  python code/e3_datalayer_paired_ci.py --n_cap 200           # 大集上限（加速）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import audit_datasets as A  # noqa: E402  (模块导入即调用 _dll_guard.hide())


def _paired(diffs):
    from scipy import stats as st
    d = np.asarray(diffs, dtype=float)
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / (n ** 0.5) if n > 1 else float("nan")
    if n > 1 and sd > 0:
        t, p = st.ttest_rel(d, np.zeros_like(d))
        lo, hi = st.t.interval(0.95, n - 1, loc=mean, scale=se)
        cohen = mean / sd
        t, p, lo, hi, cohen = float(t), float(p), float(lo), float(hi), float(cohen)
    else:
        t = p = lo = hi = cohen = float("nan")
    return dict(n=n, mean=mean, sd=sd, median=float(np.median(d)),
                t=t, p=p, ci95=[lo, hi], cohen_d=cohen,
                n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()))


def _pick_anchor(s, lrs, anchor="auto"):
    """anchor: auto(=audit_datasets 口径, PROBA-V 取最高清晰度帧) / clearance / frame0."""
    multi = lrs.ndim >= 4
    if anchor == "frame0" or not multi:
        return 0
    if anchor == "clearance":
        if "qms" in s:
            clr = np.asarray(s["qms"]).reshape(len(lrs), -1).mean(1)
            return int(np.argmax(clr))
        return 0
    # auto
    if "qms" in s:
        clr = np.asarray(s["qms"]).reshape(len(lrs), -1).mean(1)
        return int(np.argmax(clr))
    return 0


def run_dataset(name, val_frac=0.1, n_cap=None, anchor="auto"):
    ds, meta = A.LOADERS[name]()
    N = len(ds)
    n_train = max(1, int((1.0 - val_frac) * N))
    idxs = list(range(n_train, N))
    if n_cap and len(idxs) > n_cap:
        idxs = idxs[:n_cap]

    d_list, cc_list = [], []
    p_bic, p_mean = [], []
    for i in idxs:
        s = A.fetch(ds, int(i))
        lrs, hr = np.asarray(s["lrs"]), np.asarray(s["hr"])
        mask = s.get("sm", s.get("mask", None))
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape[-2:] != hr.shape[-2:]:
                mask = None
        if mask is not None and mask.size and mask.mean() < 0.05:
            continue
        a = _pick_anchor(s, lrs, anchor)
        lr0 = lrs[a]
        H, W = hr.shape[-2], hr.shape[-1]
        bic = A.bicubic_up(lr0, (H, W))
        gtf = hr.astype(np.float64)
        bicf = bic.astype(np.float64)
        if mask is not None:
            m = np.broadcast_to(mask.astype(bool), gtf.shape)
            gv, bv = gtf[m], bicf[m]
        else:
            gv, bv = gtf.ravel(), bicf.ravel()
        if gv.size < 100:
            continue
        dr = float(gtf.max() - gtf.min()) or 1.0
        pb = A.psnr(bv, gv, dr)
        pm = A.psnr(np.full_like(gv, gv.mean()), gv, dr)
        p_bic.append(pb)
        p_mean.append(pm)
        d_list.append(pb - pm)
        cc_list.append(A.pearson(bv, gv))

    paired = _paired(d_list)
    return {
        "dataset": name,
        **meta,
        "anchor": anchor,
        "val_frac": val_frac,
        "n_val": len(idxs),
        "n_used": paired["n"],
        "cc_raw_mean": float(np.mean(cc_list)) if cc_list else float("nan"),
        "psnr_bic_mean": float(np.mean(p_bic)) if p_bic else float("nan"),
        "psnr_mean_mean": float(np.mean(p_mean)) if p_mean else float("nan"),
        # 逐样本配对（E3 主口径）
        "delta_floor_paired": paired,
        # 与历史口径对账：中位数之差（difference of medians）
        "delta_floor_median_diff": (float(np.median(p_bic) - np.median(p_mean))
                                    if p_bic else float("nan")),
    }


def to_md(rows):
    head = ["数据集", "尺度", "传感器", "时相", "n", "CC↑",
            "Δ_floor 配对均值 [95%CI]", "p", "d", "Δ_floor 中位数之差", "负值样本"]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        pf = r["delta_floor_paired"]
        ci = pf["ci95"]
        out.append("| {dataset} | {scale}× | {sensor} | {temporal} | {n} | {cc:.3f} | "
                   "**{m:+.2f}** [{lo:+.2f}, {hi:+.2f}] | {p:.3g} | {d:+.2f} | "
                   "{md:+.2f} | {neg}/{n} |".format(
                       dataset=r["dataset"], scale=r["scale"], sensor=r["sensor"],
                       temporal=r["temporal"], n=r["n_used"], cc=r["cc_raw_mean"],
                       m=pf["mean"], lo=ci[0], hi=ci[1], p=pf["p"], d=pf["cohen_d"],
                       md=r["delta_floor_median_diff"], neg=pf["n_neg"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(A.LOADERS))
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--n_cap", type=int, default=None)
    ap.add_argument("--anchor", default="auto",
                    choices=["auto", "clearance", "frame0"],
                    help="auto=audit 口径(PROBA-V 最高清晰度帧) / clearance / frame0")
    ap.add_argument("--out_suffix", default="", help="附加到输出文件名，便于并列多次运行")
    ap.add_argument("--out", default="experiments/e3_datalayer_paired_ci.json")
    ap.add_argument("--md", default="docs/E3_数据层配对CI.md")
    args = ap.parse_args()

    rows = []
    for name in args.datasets:
        if name not in A.LOADERS:
            print("[skip] unknown dataset %s" % name)
            continue
        try:
            r = run_dataset(name, val_frac=args.val_frac, n_cap=args.n_cap, anchor=args.anchor)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("[fail] %s: %s" % (name, e))
            continue
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)

    if not rows:
        print("no results")
        return

    if args.out_suffix:
        args.out = args.out.replace(".json", args.out_suffix + ".json")
        args.md = args.md.replace(".md", args.out_suffix + ".md")
    for p in (args.out, args.md):
        os.makedirs(os.path.dirname(os.path.join(BASE, p)), exist_ok=True)
    with open(os.path.join(BASE, args.out), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    md = ["# E3 —— 数据层 Δ_floor 逐样本配对置信区间", "",
          "**零训练**；样本集 = 与方法层同一切分（末 val_frac=%.2f）。"
          "Δ_floor = PSNR(bicubic)−PSNR(mean)，与 data_range 无关。" % args.val_frac, "",
          to_md(rows), ""]
    with open(os.path.join(BASE, args.md), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + to_md(rows))
    print("\n[saved] %s / %s" % (args.out, args.md))


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            import _dll_guard
            _dll_guard.restore()
        except Exception:
            pass
