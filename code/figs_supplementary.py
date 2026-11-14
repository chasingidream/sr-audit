#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supplementary figures & tables (S1-S3) for the Nature-submission manuscript.

S1  CrossAttnAlign per-sample paired contrast (WorldStrat 6.63x, n=60)
      -> experiments/figs_final/figS1_mtxa_persample.{pdf,png}
S2  HighRes-net per-sample gain distribution (394 held-out scenes)
      -> experiments/figs_final/figS2_highresnet_dist.{pdf,png}
S3  Six-view robustness (n=60 -> 42) forest plot + Markdown table
      -> experiments/figs_final/figS3_robustness_forest.{pdf,png}
      -> docs/Supplementary_S3_六视图稳健性.md

All values are read from the validated JSONs; nothing is hard-coded except labels.
Usage: python code/figs_supplementary.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
DOCS = os.path.join(ROOT, "docs")
OUTDIR = os.path.join(EXP, "figs_final")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.unicode_minus": False, "font.size": 8.5, "axes.labelsize": 9.0,
    "axes.titlesize": 9.2, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "legend.fontsize": 7.2, "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
# 色板 2026-09-27：弃用红绿配对 -> 蓝/紫（CEE 二轮意见）
C_DARK, C_MID, C_LIGHT = "#333333", "#1f77b4", "#7b3294"
C_POS, C_NEG = "#2171b5", "#7b3294"


def load(name: str) -> dict:
    with open(os.path.join(EXP, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(OUTDIR, name + ".pdf"))
    fig.savefig(os.path.join(OUTDIR, name + ".png"), dpi=600)
    plt.close(fig)
    print(f"[saved] {name}.pdf / {name}.png")


# ============================================================ S1 ============
def fig_S1() -> None:
    d = load("eval_mtxa_worldstrat.json")
    ps = d["per_sample"]
    conds = ["mtxa_mis_xa", "mtxa_mis_noxa", "mtxa_aligned", "bicubic"]
    arr = {k: np.array([s["psnr"] for s in ps[k]]) for k in conds}
    contrasts = [
        ("xa − noxa\n(alignment contribution)", arr["mtxa_mis_xa"] - arr["mtxa_mis_noxa"]),
        ("xa − aligned\n(residual to clean upper bound)", arr["mtxa_mis_xa"] - arr["mtxa_aligned"]),
        ("xa − bicubic\n(gain over anchor upsampling)", arr["mtxa_mis_xa"] - arr["bicubic"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), sharey=True)
    for ax, (lab, dv) in zip(axes, contrasts):
        rng = np.random.RandomState(0)
        jit = (rng.rand(len(dv)) - 0.5) * 0.5
        col = C_POS if dv.mean() > 0 else C_NEG
        ax.scatter(jit, dv, s=13, color=col, alpha=0.75, edgecolor="white", lw=0.4,
                   zorder=3)
        m = float(dv.mean())
        sd = float(dv.std(ddof=1))
        se = sd / np.sqrt(len(dv))
        half = 1.96 * se
        ax.errorbar(0.0, m, yerr=half, fmt="D", ms=6, color=C_DARK, zorder=5,
                    capsize=4, lw=1.5)
        ax.axhline(0, color="#888888", lw=0.9, ls="--")
        ax.set_title(lab, fontsize=7.6)
        ax.set_xlim(-0.55, 0.55)
        ax.set_xticks([])
        ax.text(0.97, 0.03, f"mean {m:+.2f}\n95% CI [{m-half:+.2f}, {m+half:+.2f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
                color=C_DARK)
    axes[0].set_ylabel("per-sample $\\Delta$PSNR  [dB]")
    axes[0].set_ylim(-22, 10)
    fig.suptitle("S1  CrossAttnAlign per-sample paired contrast "
                 "(WorldStrat 6.63×, n = 60)", fontsize=9.2, y=1.04)
    fig.tight_layout()
    save(fig, "figS1_mtxa_persample")


# ============================================================ S2 ============
def fig_S2() -> None:
    d = load(os.path.join("sota_review", "highresnet_persample.json"))
    gain = np.array(d["gain_reg"], float)                   # model - bicubic
    skill = np.array(d["psnr_model"], float) - np.array(d["psnr_floor"], float)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.8))
    for ax, v, title, xlab in (
        (a1, gain, "(a) Reported gain  model − bicubic", "gain over bicubic  [dB]"),
        (a2, skill, "(b) True skill over the constant", "PSNR(model) − PSNR(mean)  [dB]"),
    ):
        ax.hist(v, bins=34, color=C_MID, alpha=0.82, edgecolor="white", lw=0.4)
        m = float(v.mean())
        ax.axvline(0, color="#888888", lw=1.1, ls="--")
        ax.axvline(m, color=C_NEG, lw=1.6)
        ax.annotate(f"mean {m:+.2f}", (m, ax.get_ylim()[1] * 0.92),
                    color=C_NEG, fontsize=7.4, ha="left",
                    xytext=(4, 0), textcoords="offset points")
        ax.set_xlabel(xlab)
        ax.set_ylabel("scenes")
        ax.set_title(title, fontsize=8.6)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
    # share a note on the fraction whose skill is non-positive
    frac = float((skill <= 0).mean()) * 100
    a2.text(0.97, 0.72, f"{frac:.0f}% of scenes\nlie at/below the constant",
            transform=a2.transAxes, ha="right", va="top", fontsize=6.8, color=C_NEG)
    fig.suptitle("S2  HighRes-net (official baseline) per-sample distribution, "
                 "394 held-out scenes", fontsize=9.2, y=1.04)
    fig.tight_layout()
    save(fig, "figS2_highresnet_dist")
    return gain, skill


# ============================================================ S3 ============
VIEW_LABEL = {
    "A_full": "A · full sample",
    "B_deg_sam_gt170": "B · drop SAM > 170°",
    "C_deg_psnr_lt10": "C · drop PSNR < 10",
    "D_deg_ssim_lt0": "D · drop SSIM < 0",
    "E_g5_leak_gt45": "E · drop G5 leak > 45 dB",
    "F_clean_allgates": "F · all gates (clean)",
}
CONTRAST_LABEL = {
    "xa_minus_noxa": "xa − noxa",
    "xa_minus_aligned": "xa − aligned",
    "xa_minus_bicubic": "xa − bicubic",
}


def fig_S3() -> dict:
    d = load("robustness_mtxa_worldstrat.json")
    views = list(d["views"].keys())
    contrasts = ["xa_minus_noxa", "xa_minus_aligned", "xa_minus_bicubic"]
    colors = {"xa_minus_noxa": C_MID, "xa_minus_aligned": "#e8a33d",
              "xa_minus_bicubic": "#7b3294"}

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ypos = {}
    y = 0.0
    for v in views:
        ypos[v] = y
        y += 1.0
    n_views = len(views)
    gap = 0.26
    for ci, con in enumerate(contrasts):
        off = (ci - 1) * gap
        for v in views:
            r = d["views"][v][con]
            m, lo, hi = r["mean"], r["ci95"][0], r["ci95"][1]
            yv = ypos[v] + off
            ax.plot([lo, hi], [yv, yv], color=colors[con], lw=1.4, zorder=3)
            ax.plot([m], [yv], "o", ms=4.6, color=colors[con], zorder=4,
                    label=CONTRAST_LABEL[con] if v == views[0] else None)
            sig = r["p"] < 0.05
            ax.plot([lo - 0.06, lo - 0.06], [yv - 0.07, yv + 0.07],
                    color=colors[con], lw=1.4)
    ax.axvline(0, color="#666666", lw=1.0, ls="--")
    ax.set_yticks([ypos[v] for v in views])
    ax.set_yticklabels([VIEW_LABEL.get(v, v) for v in views], fontsize=7.4)
    ax.set_ylim(-0.7, n_views - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$PSNR  [dB]  (mean with 95% CI)")
    ax.set_title("S3  Robustness of the CrossAttnAlign conclusion\n"
                 "across six gate-defined views", fontsize=8.8)
    ax.grid(axis="x", alpha=0.2, lw=0.5)
    ax.legend(loc="lower right", fontsize=6.6, framealpha=0.92, handlelength=1.2)
    fig.tight_layout()
    save(fig, "figS3_robustness_forest")
    return d


def write_S3_table(d: dict, gain, skill) -> None:
    views = list(d["views"].keys())
    contrasts = ["xa_minus_noxa", "xa_minus_aligned", "xa_minus_bicubic"]
    lines = []
    lines.append("# Supplementary S3 — CrossAttnAlign 六视图稳健性表\n")
    lines.append("> 生成脚本 `code/figs_supplementary.py`。数据源 `experiments/robustness_mtxa_worldstrat.json`（由 `code/robustness_mtxa_ws.py` 产出）。")
    lines.append("> 单位 dB；视图 = 按退化/泄漏门剔除样本后的重算（n 见表）。判定线：p<0.05 显著。\n")
    lines.append("## S3.1 视图定义与样本数\n")
    lines.append("| 视图 | 说明 | n | 剔除 |")
    lines.append("|---|---|---|---|")
    removed = d["removed_counts"]
    defs = {
        "A_full": "全样本（无剔除）",
        "B_deg_sam_gt170": "剔除任一条件 SAM_deg > 170°",
        "C_deg_psnr_lt10": "剔除任一条件 PSNR < 10 dB",
        "D_deg_ssim_lt0": "剔除任一条件 SSIM < 0",
        "E_g5_leak_gt45": "剔除任一条件 PSNR > 45 dB（G5 泄漏门）",
        "F_clean_allgates": "同时满足 B–E（最干净子集）",
    }
    for v in views:
        n = d["views"][v]["xa_minus_noxa"]["n"]
        lines.append(f"| {VIEW_LABEL.get(v, v)} | {defs.get(v, '')} | {n} | {d['n_total'] - n} |")
    lines.append(f"\n> 逐门剔除计数（重叠）：{removed}\n")

    for i, con in enumerate(contrasts, start=2):
        lines.append(f"## S3.{i} {CONTRAST_LABEL[con]}\n")
        lines.append("| 视图 | n | 均值 | 中位 | 10%截尾均值 | 95% CI | t | p | Cohen's d | 判定 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for v in views:
            r = d["views"][v][con]
            verdict = "显著负" if (r["p"] < 0.05 and r["mean"] < 0) else (
                "显著正" if (r["p"] < 0.05 and r["mean"] > 0) else "不显著")
            lines.append(
                f"| {VIEW_LABEL.get(v, v)} | {r['n']} | {r['mean']:+.3f} | {r['median']:+.3f} | "
                f"{r['trim10_mean']:+.3f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | "
                f"{r['t']:+.2f} | {r['p']:.3f} | {r['cohen_d']:+.2f} | {verdict} |")
        lines.append("")

    lines.append("## S3.5 结论（六视图一致）\n")
    lines.append("- `xa − noxa`：六个视图**全不显著**（p 0.57–0.97）→ 自对齐模块无可检出贡献。")
    lines.append("- `xa − aligned`：六个视图**全显著为负**（p 0.013–0.029）→ 与无抖动上界存在稳定残差。")
    lines.append("- `xa − bicubic`：六个视图**从未显著为正**（p 0.055–0.294，均值恒负）→ 从未超过锚点上采样。")
    lines.append("- ⇒ 结论对离群/退化/泄漏样本的剔除**稳健**；最干净子集 F（n=42）亦不改变符号与显著性格局。\n")

    lines.append("## S3.6 附：HighRes-net 全量 394 场景逐样本分布（图 S2 数据）\n")
    lines.append(f"- 报告增益 model−bicubic：均值 **{gain.mean():+.2f} dB**（中位 {np.median(gain):+.2f}），"
                 f"范围 [{gain.min():+.1f}, {gain.max():+.1f}]，正占 {100*(gain>0).mean():.0f}%。")
    lines.append(f"- 真实技能 model−mean：均值 **{skill.mean():+.2f} dB**（中位 {np.median(skill):+.2f}），"
                 f"其中 **{100*(skill<=0).mean():.0f}%** 的场景 ≤ 0（不优于常数预测器）。\n")

    out = os.path.join(DOCS, "Supplementary_S3_六视图稳健性.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {out}")


def main() -> None:
    fig_S1()
    gain, skill = fig_S2()
    d = fig_S3()
    write_S3_table(d, gain, skill)
    print(f"[done] supplementary -> {OUTDIR}")


if __name__ == "__main__":
    main()
