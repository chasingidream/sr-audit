#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3-D 汇总：PROBA-V n≈297（val_frac=0.5）3 seed 正向复现汇总。

输入：experiments/paired_probav_n297_s{42,123,7}.json（由 eval_paired_floor.py 产出）
输出：experiments/p3_probav_n300_summary.json + 控制台 Markdown 表

判读：
  - 每 seed 的 model−bicubic（逐样本配对，含 95%CI/p/d）
  - 过技能地板（model−mean > 0）
  - 与原 n=60 轴（paired_probav.json）并列对照

用法：python code/p3_probav_n300_summary.py
"""
from __future__ import annotations

import json
import os
import statistics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(BASE, "experiments")

SEEDS = [42, 123, 7]
REF = "paired_probav.json"          # 历史金标准（val 后10%，n=60）


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def extract(d):
    pps = d.get("paired_per_sample", {})
    gm = d.get("group_mean_per_sample", {})
    mb = pps.get("model_vs_bicubic", {}).get("psnr", {})
    mm = pps.get("model_vs_mean", {}).get("psnr", {})
    bm = pps.get("bicubic_vs_mean", {}).get("psnr", {})
    return dict(
        n=d.get("n_sample"),
        val_dr=d.get("val_dr"),
        psnr_mean=gm.get("mean", {}).get("psnr"),
        psnr_bicubic=gm.get("bicubic", {}).get("psnr"),
        psnr_model=gm.get("model", {}).get("psnr"),
        gain=mb.get("mean_diff"), ci=mb.get("ci95"), p=mb.get("p"),
        d=mb.get("cohen_d"),
        floor_gap=(mm.get("mean_diff")),
        passes_floor=bool(mm.get("mean_diff", 0) > 0),
        bicubic_minus_mean=bm.get("mean_diff"),
    )


def main():
    rows = {}
    # 新 3 seed
    for s in SEEDS:
        p = os.path.join(EXP, f"paired_probav_n297_s{s}.json")
        if not os.path.exists(p):
            print(f"[skip] 缺 {p}")
            continue
        rows[f"n300_s{s}"] = extract(load(p))
    # 参考轴
    rp = os.path.join(EXP, REF)
    if os.path.exists(rp):
        rows["ref_n60"] = extract(load(rp))

    # 3 seed 聚合（gain 的均值/极差）
    gains = [r["gain"] for k, r in rows.items() if k.startswith("n300_")]
    ns = [r["n"] for k, r in rows.items() if k.startswith("n300_")]
    agg = dict(
        n_seeds=len(gains),
        gain_mean=(statistics.fmean(gains) if gains else None),
        gain_min=(min(gains) if gains else None),
        gain_max=(max(gains) if gains else None),
        all_positive=bool(gains) and all(g > 0 for g in gains),
        all_pass_floor=bool(gains) and all(rows[k]["passes_floor"]
                                           for k in rows if k.startswith("n300_")),
        n_sample_each=(max(ns) if ns else None),
    )

    out = dict(task="P3-D PROBA-V n≈297 3-seed 正向复现",
               val_frac=0.5, seeds=SEEDS, results=rows, aggregate=agg)
    op = os.path.join(EXP, "p3_probav_n300_summary.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Markdown 表
    print("\n| 运行 | n | Δmodel−bic (dB) | 95%CI | p | d | 过地板? |")
    print("|---|---|---|---|---|---|---|")
    for k, r in rows.items():
        ci = r["ci"] or [float("nan"), float("nan")]
        pv = r["p"]
        ps = "n/a" if pv is None else f"{pv:.4g}"
        print(f"| {k} | {r['n']} | {r['gain']:+.3f} | "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}] | {ps} | {r['d']:+.2f} | "
              f"{'✓' if r['passes_floor'] else '✗'} |")
    print(f"\n[agg] 3 seed gain 均值={agg['gain_mean']:+.3f} "
          f"极差=[{agg['gain_min']:+.3f},{agg['gain_max']:+.3f}] "
          f"全正={agg['all_positive']} 全过地板={agg['all_pass_floor']} "
          f"n_each={agg['n_sample_each']}")
    print(f"[saved] {op}")


if __name__ == "__main__":
    main()
