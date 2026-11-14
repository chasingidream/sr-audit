#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E2 —— 多 seed 汇总：seed 间方差，填充"单 seed"limitation。

对每个轴（SEN2VENµS 2× / WorldStrat 6.63×）读取 3 个 seed 的 `eval_paired_floor.py`
输出（`paired_{s2v,ws}_s{42,123,7}.json`），汇总：
  * 每 seed：model−bicubic（方法增益）、model−mean（地板余量）、bicubic−mean（Δ_floor）；
  * 跨 seed：增益均值 / 极差 / 标准差 / 是否全部显著 / 是否全部越过地板；
  * 参照该轴历史单 seed 金标准（`paired_s2v.json` / `paired_worldstrat.json`）。

用法：python code/e2_multiseed_summary.py
"""
from __future__ import annotations

import json
import os
import statistics as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AXES = [
    {
        "name": "SEN2VENuS 2x (sisr, cross-sensor, same-day)",
        "files": ["experiments/paired_s2v_s42.json",
                  "experiments/paired_s2v_s123.json",
                  "experiments/paired_s2v_s7.json"],
        "ref": "experiments/paired_s2v.json",
    },
    {
        "name": "WorldStrat 6.63x (mtxa_aligned, cross-sensor, multi-frame)",
        "files": ["experiments/paired_ws_s42.json",
                  "experiments/paired_ws_s123.json",
                  "experiments/paired_ws_s7.json"],
        "ref": "experiments/paired_worldstrat.json",
    },
]


def _psnr_block(obj, key):
    try:
        return obj["paired_per_sample"][key]["psnr"]
    except Exception:
        return None


def load_row(path):
    if not os.path.isabs(path):
        path = os.path.join(BASE, path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    mb = _psnr_block(d, "model_vs_bicubic")
    mm = _psnr_block(d, "model_vs_mean")
    bm = _psnr_block(d, "bicubic_vs_mean")
    gm = d.get("group_mean_per_sample", {})
    return {
        "file": os.path.basename(path),
        "n": d.get("n_sample"),
        "val_dr": d.get("val_dr"),
        "group_mean": {k: (gm.get(k, {}) or {}).get("psnr") for k in ("mean", "bicubic", "model")},
        "model_vs_bicubic": mb,
        "model_vs_mean": mm,
        "bicubic_vs_mean": bm,
        "passes_floor": bool(mm and mm["mean_diff"] > 0),
        "gain_significant": bool(mb and mb["p"] is not None and mb["p"] < 0.05),
    }


def summarize(axis):
    rows = []
    for f in axis["files"]:
        r = load_row(f)
        if r:
            rows.append(r)
    agg = None
    if rows:
        gains = [r["model_vs_bicubic"]["mean_diff"] for r in rows if r["model_vs_bicubic"]]
        floors = [r["bicubic_vs_mean"]["mean_diff"] for r in rows if r["bicubic_vs_mean"]]
        agg = {
            "n_seeds": len(rows),
            "gain_mean": st.mean(gains) if gains else None,
            "gain_min": min(gains) if gains else None,
            "gain_max": max(gains) if gains else None,
            "gain_range": (max(gains) - min(gains)) if gains else None,
            "gain_sd": (st.stdev(gains) if len(gains) > 1 else 0.0) if gains else None,
            "all_gain_significant": all(r["gain_significant"] for r in rows) if rows else None,
            "all_pass_floor": all(r["passes_floor"] for r in rows) if rows else None,
            "delta_floor_mean": st.mean(floors) if floors else None,
        }
    ref = load_row(axis["ref"]) if axis.get("ref") else None
    return {"axis": axis["name"], "seeds": rows, "aggregate": agg, "reference_single_seed": ref}


def _f(x, fmt="%+.3f"):
    return "n/a" if x is None else (fmt % x)


def main():
    out = []
    for ax in AXES:
        out.append(summarize(ax))

    res = {"E2": "multi-seed variance (>=3 seeds)", "axes": out}
    op = os.path.join(BASE, "experiments/e2_multiseed_summary.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    lines = ["# E2 —— 多 seed 汇总（seed 间方差）", ""]
    for a in out:
        lines.append("## %s" % a["axis"])
        lines.append("")
        lines.append("| seed (file) | n | model−bicubic (p) | model−mean | bicubic−mean (Δ_floor) | 过地板? |")
        lines.append("|---|---|---|---|---|---|")
        for r in a["seeds"]:
            mb, mm, bm = r["model_vs_bicubic"], r["model_vs_mean"], r["bicubic_vs_mean"]
            lines.append("| %s | %s | %s (p=%s) | %s | %s | %s |" % (
                r["file"], r["n"],
                _f(mb["mean_diff"] if mb else None), ("%.3g" % mb["p"]) if mb and mb["p"] is not None else "n/a",
                _f(mm["mean_diff"] if mm else None), _f(bm["mean_diff"] if bm else None),
                r["passes_floor"]))
        ag = a["aggregate"]
        if ag:
            lines.append("")
            lines.append("- **聚合**：gain 均值 %s（极差 %s，sd %s，min %s / max %s）；全显著=%s；全过地板=%s；Δ_floor 均值 %s" % (
                _f(ag["gain_mean"]), _f(ag["gain_range"], "%.3f"), _f(ag["gain_sd"], "%.3f"),
                _f(ag["gain_min"]), _f(ag["gain_max"]), ag["all_gain_significant"], ag["all_pass_floor"],
                _f(ag["delta_floor_mean"])))
        ref = a["reference_single_seed"]
        if ref:
            mb, bm = ref["model_vs_bicubic"], ref["bicubic_vs_mean"]
            lines.append("- 参照单 seed 金标准 %s：n=%s，model−bicubic %s（p=%s），bicubic−mean %s" % (
                ref["file"], ref["n"], _f(mb["mean_diff"] if mb else None),
                ("%.3g" % mb["p"]) if mb and mb["p"] is not None else "n/a",
                _f(bm["mean_diff"] if bm else None)))
        lines.append("")
    md = os.path.join(BASE, "docs/E2_多seed汇总.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[saved] %s / %s" % (op, md))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
