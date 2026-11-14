#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publication-quality final figures (Fig. 1-4) for the Nature-submission manuscript.

Design goals
------------
* English labels only (journal-ready), single consistent protocol everywhere:
  every gain is the **per-sample** paired contrast (never batch pooling).
* Vector output (PDF, with editable TrueType text via fonttype=42) + 600-dpi PNG.
* All headline numbers are *read from the validated JSONs* in experiments/,
  so a re-run of the audits automatically propagates into the figures.

Outputs (experiments/figs_final/):
  fig1_regime.{pdf,png}         three regimes phase diagram + gain bars
  fig2_floor_collapse.{pdf,png} Delta_floor collapse vs factor (+ HighRes-net)
  fig3_e1_decomposition.{pdf,png} pure-scale vs real arms (decorrelation cost)
  fig4_crosssensor_3x.{pdf,png}  same-factor 3x arm comparison (S2/L8)

Usage: python code/figs_final.py
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
OUTDIR = os.path.join(EXP, "figs_final")
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------------ style ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.6,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# 色板 2026-09-27：按 CEE 二轮意见弃用红绿配对，改 colorblind-safe 蓝/橙/紫
C_POS = "#2171b5"   # positive -> 蓝
C_CEIL = "#e8912d"  # ceiling -> 橙
C_FLOOR = "#7b3294"  # floor-failure -> 紫
C_BLUE = "#1f77b4"
C_GREEN = "#e8912d"  # (fig2/3/4 内的第二臂) -> 橙
C_RED = "#7b3294"    # (fig2/3/4 内的第三臂) -> 紫
GREY = "#555555"

REG_COLOR = {"positive": C_POS, "ceiling": C_CEIL, "floor-failure": C_FLOOR}


def load(name: str) -> dict:
    with open(os.path.join(EXP, name), encoding="utf-8") as f:
        return json.load(f)


def contrast(d: dict, which: str, metric: str = "psnr") -> dict:
    return d["paired_per_sample"][which][metric]


def save(fig, name: str) -> None:
    pdf = os.path.join(OUTDIR, name + ".pdf")
    png = os.path.join(OUTDIR, name + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f"[saved] {name}.pdf / {name}.png")


# ------------------------------------------------------- per-sample axes data -
def axis_rows() -> list[dict]:
    """Three axes, every value a per-sample paired contrast (single protocol)."""
    rows = []

    # PROBA-V 3x same-sensor multi-frame: 3 seeds x n=297 (P3-D)
    seeds = [contrast(load(f"paired_probav_n297_s{s}.json"), "model_vs_bicubic")
             for s in ("42", "123", "7")]
    floors = [contrast(load(f"paired_probav_n297_s{s}.json"), "bicubic_vs_mean")
              for s in ("42", "123", "7")]
    mv = [c["mean_diff"] for c in seeds]
    rows.append(dict(
        key="probav", regime="positive",
        label="PROBA-V 3$\\times$\nsame-sensor, multi-frame",
        x=float(np.mean([c["mean_diff"] for c in floors])), y=float(np.mean(mv)),
        lo=float(min(mv)), hi=float(max(mv)),
        p=float(max(c["p"] for c in seeds)), n=int(seeds[0]["n"]), seeds=mv))

    for fn, key, regime, lab in (
        ("paired_s2v.json", "s2v", "ceiling",
         "SEN2VENµS 2$\\times$ FGMANAUS\ncross-sensor, same-day"),
        ("paired_worldstrat.json", "ws", "floor-failure",
         "WorldStrat 6.63$\\times$\ncross-sensor, multi-frame"),
    ):
        d = load(fn)
        c = contrast(d, "model_vs_bicubic")
        f = contrast(d, "bicubic_vs_mean")
        rows.append(dict(key=key, regime=regime, label=lab,
                         x=f["mean_diff"], y=c["mean_diff"],
                         lo=c["ci95"][0], hi=c["ci95"][1],
                         p=c["p"], n=c["n"]))

    # SEN2VENµS SO2 —— 同一数据集的第二站点（三 seed 从零重训）。
    # regime 由 ceiling 反转为 positive：regime 标签随站点而变，不随数据集名。
    _so2 = [f"paired_s2v_so2_s{s}.json" for s in ("42", "123", "7")]
    if all(os.path.exists(os.path.join(EXP, f)) for f in _so2):
        sseeds = [contrast(load(f), "model_vs_bicubic") for f in _so2]
        sfloors = [contrast(load(f), "bicubic_vs_mean") for f in _so2]
        smv = [c["mean_diff"] for c in sseeds]
        rows.append(dict(
            key="so2", regime="positive",
            label="SEN2VENµS 2$\\times$ SO2\nsecond station",
            x=float(np.mean([c["mean_diff"] for c in sfloors])), y=float(np.mean(smv)),
            lo=float(min(smv)), hi=float(max(smv)),
            p=float(max(c["p"] for c in sseeds)), n=int(sseeds[0]["n"]), seeds=smv))

    # PROBA-V NIR —— 独立地理 ROI 子集的正向轴复现（评审意见①）。
    # 产物由 eval_paired_floor.py 在 nir cfg 上生成；文件缺失时静默跳过（图仍可出）。
    _nir = [f"paired_probav_nir_s{s}.json" for s in ("42", "123", "7")]
    if all(os.path.exists(os.path.join(EXP, f)) for f in _nir):
        nseeds = [contrast(load(f), "model_vs_bicubic") for f in _nir]
        nfloors = [contrast(load(f), "bicubic_vs_mean") for f in _nir]
        nmv = [c["mean_diff"] for c in nseeds]
        rows.append(dict(
            key="probav_nir", regime="positive",
            label="PROBA-V NIR 3$\\times$\nindependent ROIs",
            x=float(np.mean([c["mean_diff"] for c in nfloors])), y=float(np.mean(nmv)),
            lo=float(min(nmv)), hi=float(max(nmv)),
            p=float(max(c["p"] for c in nseeds)), n=int(nseeds[0]["n"]), seeds=nmv))

    # SEN2VENµS 追加站点（★R2 补轴）：ceiling / positive 是否为**站点**的性质。
    # 与 FGMANAUS / SO2 逐项同口径：Δ_floor 与 Δ_model 均取自 eval_paired_floor.py
    # 产出的 paired_s2v_<key>_s<seed>.json（三 seed 汇总）。文件缺失则静默跳过。
    _extra_sites = (
        ("naryn",    "NARYN"),
        ("atto",     "ATTO"),
        ("sudoue_2", "SUDOUE-2"),
    )
    for _key, _lab in _extra_sites:
        _fs = [f"paired_s2v_{_key}_s{s}.json" for s in ("42", "123", "7")]
        if not all(os.path.exists(os.path.join(EXP, f)) for f in _fs):
            continue
        _ms = [contrast(load(f), "model_vs_bicubic") for f in _fs]
        _fls = [contrast(load(f), "bicubic_vs_mean") for f in _fs]
        _mv = [c["mean_diff"] for c in _ms]
        _dfl = float(np.mean([c["mean_diff"] for c in _fls]))
        _dmo = float(np.mean(_mv))
        _beta = _dmo / (_dfl + _dmo) if (_dfl + _dmo) != 0 else float("nan")
        _reg = ("positive" if all(c["p"] < 0.05 and c["mean_diff"] > 0
                                  for c in _ms) else "ceiling")
        rows.append(dict(key=_key, regime=_reg, label=_lab,
                         x=_dfl, y=_dmo, lo=float(min(_mv)), hi=float(max(_mv)),
                         p=float(max(c["p"] for c in _ms)), n=int(_ms[0]["n"]),
                         seeds=_mv, beta=_beta))
    return rows


# =========================================================== Figure 1 ========
def fig_regime(rows: list[dict]) -> None:
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.1, 3.5), gridspec_kw={"width_ratios": [1.28, 1.0]})

    # ---- left: regime phase diagram ----
    xs = [r["x"] for r in rows] + [0.0]
    xlim = (min(xs) - 2.4, max(xs) + 2.4)
    ymax = max(abs(r["lo"]) for r in rows) + 1.6
    ylim = (-max(4.6, ymax), max(4.6, ymax))

    lx = np.array(xlim)
    ax.plot(lx, -lx, color=GREY, lw=1.2, ls="--", zorder=1)
    ax.fill_between(lx, -lx, ylim[1], color=C_POS, alpha=0.055, zorder=0)
    ax.fill_between(lx, ylim[0], -lx, color=C_FLOOR, alpha=0.055, zorder=0)
    ax.axvline(0.0, color="#999999", lw=0.9, ls=":", zorder=1)
    ax.axhline(0.0, color="#999999", lw=0.9, ls=":", zorder=1)

    ax.text(0.95, 3.55, "clears the floor", ha="left", va="center",
            fontsize=7.6, color=C_POS)
    ax.text(xlim[0] + 0.3, ylim[0] + 0.35, "below the floor", ha="left",
            va="bottom", fontsize=7.6, color=C_FLOOR)
    ax.text(xlim[0] + 0.3, ylim[1] - 0.35, "floor-failure", ha="left", va="top",
            fontsize=7.6, color=C_FLOOR, alpha=0.9)
    ax.text(xlim[1] - 0.3, -3.55, "ceiling", ha="right", va="top",
            fontsize=7.6, color=C_CEIL)
    ax.text(10.4, 3.00, "SEN2VENµS $2\\times$: five stations,\none protocol and backbone",
            ha="center", va="center", fontsize=7.0, color="#3a3a3a")

    # 短名（面板 (a) 用）：8 个臂全部落在 y ∈ [0.1, 2.5] 的窄带内，多行标签必然互叠，
    # 故 (a) 只标一行短名，(b) 的刻度标签承载 regime / n / p。
    _SHORT = {
        "probav": "PROBA-V 3×", "probav_nir": "PROBA-V NIR 3×",
        "s2v": "FGMANAUS", "so2": "SO2", "naryn": "NARYN",
        "atto": "ATTO", "sudoue_2": "SUDOUE-2", "ws": "WorldStrat 6.63×",
    }

    for r in rows:
        col = REG_COLOR[r["regime"]]
        ax.errorbar(r["x"], r["y"], yerr=[[r["y"] - r["lo"]], [r["hi"] - r["y"]]],
                    fmt="o", ms=9, color=col, ecolor=col, elinewidth=1.8,
                    capsize=3.5, capthick=1.6, zorder=5,
                    markeredgecolor="white", markeredgewidth=1.1)
        for sy in r.get("seeds", []):
            ax.plot(r["x"] + 0.28, sy, marker="o", ms=5.0, mfc="white",
                    mec=col, mew=1.3, zorder=7)
        _ann = {
            "probav":     dict(xytext=(-11, 9),   ha="right"),
            "probav_nir": dict(xytext=(-11, -9),  ha="right"),
            "atto":       dict(xytext=(0, -13),   ha="center"),
            "so2":        dict(xytext=(0, 13),    ha="center"),
            "s2v":        dict(xytext=(9, -4),    ha="left"),
            "sudoue_2":   dict(xytext=(0, 13),    ha="center"),
            "naryn":      dict(xytext=(0, -13),   ha="center"),
            "ws":         dict(xytext=(9, 0),     ha="left"),
        }.get(r["key"], dict(xytext=(9, 8), ha="left"))
        ax.annotate(_SHORT.get(r["key"], r["key"]), (r["x"], r["y"]),
                    textcoords="offset points", fontsize=7.4, color="#222222",
                    va="center", zorder=6, **_ann)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Recoverable-info gap  $\\Delta_{floor}$ = PSNR(bicubic) − PSNR(mean)  [dB]")
    ax.set_ylabel("Method gain  $\\Delta_{model}$\n"
                  "= PSNR(model) − PSNR(bicubic)  [dB]")
    ax.set_title("(a) Regime phase diagram", fontsize=9.5, pad=6)
    ax.grid(alpha=0.20, lw=0.5)
    ax.text(0.40, -1.55, "dashed: skill floor\n$\\Delta_{model}=-\\Delta_{floor}$",
            ha="left", va="center", fontsize=6.9, color=GREY)

    # ---- right: gain bars ----
    order = ["positive", "ceiling", "floor-failure"]
    srt = sorted(rows, key=lambda r: order.index(r["regime"]))
    ys = list(range(len(srt)))[::-1]
    for yi, r in zip(ys, srt):
        col = REG_COLOR[r["regime"]]
        ax2.barh(yi, r["y"], height=0.46, color=col, alpha=0.9,
                 edgecolor="white", zorder=3)
        ax2.plot([r["lo"], r["hi"]], [yi, yi], color="#333333", lw=1.5, zorder=4)
        for e in (r["lo"], r["hi"]):
            ax2.plot([e, e], [yi - 0.11, yi + 0.11], color="#333333", lw=1.5, zorder=4)
        ax2.text(r["hi"] + 0.18, yi, f"{r['y']:+.2f}", va="center", ha="left",
                 fontsize=7.6, color="#222222")
    ax2.axvline(0.0, color=GREY, lw=1.0, ls="--", zorder=2)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(
        [f"{_SHORT.get(r['key'], r['key'])} ({r['regime']})\n"
         f"n={r['n']}, p={r['p']:.2g}" for r in srt], fontsize=7.0)
    ax2.set_xlabel("$\\Delta_{model}$ (model − bicubic)  [dB]")
    ax2.set_title("(b) Method-layer gain (95% CI)", fontsize=9.5, pad=6)
    ax2.set_xlim(min(r["lo"] for r in rows) - 1.4, max(r["hi"] for r in rows) + 2.4)
    ax2.set_ylim(-0.6, len(srt) - 0.4)
    ax2.grid(axis="x", alpha=0.20, lw=0.5)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fig1_regime")


# =========================================================== Figure 2 ========
def fig_floor_collapse() -> None:
    # data layer: zero-training audit (docs/跨数据集审计_汇总.md), protocol-consistent
    data_layer = [(2.0, 21.96, "SEN2VENµS\n2×"), (3.0, 2.09, "PROBA-V\n3×"),
                  (6.63, -0.09, "WorldStrat\n6.63×")]
    # method layer: per-sample paired (this work) -- single consistent protocol
    method_layer = [(2.0, contrast(load("paired_s2v.json"), "bicubic_vs_mean")["mean_diff"],
                     contrast(load("paired_s2v.json"), "bicubic_vs_mean")["ci95"]),
                    (3.0, contrast(load("paired_probav.json"), "bicubic_vs_mean")["mean_diff"],
                     contrast(load("paired_probav.json"), "bicubic_vs_mean")["ci95"]),
                    (6.63, contrast(load("paired_worldstrat.json"), "bicubic_vs_mean")["mean_diff"],
                     contrast(load("paired_worldstrat.json"), "bicubic_vs_mean")["ci95"])]
    hr_floor = 24.677 - 31.593          # HighRes-net official bicubic baseline (394 scenes)

    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    xs = np.array([p[0] for p in data_layer])
    ax.plot(xs, [p[1] for p in data_layer], "o-", color=C_BLUE, ms=6,
            label="data layer (zero training)")
    for x, y, lab in data_layer:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=6.8, color=C_BLUE)

    mx = np.array([p[0] for p in method_layer])
    my = np.array([p[1] for p in method_layer])
    me = np.array([(p[2][1] - p[2][0]) / 2 for p in method_layer])
    ax.errorbar(mx, my, yerr=me, fmt="s--", color=C_RED, ms=5.5, capsize=3,
                label="method layer (per-sample, 95% CI)")
    ax.scatter([mx[0], mx[1], mx[2]], [my[0], my[1], my[2]], marker="s", s=22,
               facecolors="none", edgecolors=C_RED, lw=1.6, zorder=4)

    ax.scatter([6.63], [hr_floor], marker="*", s=170, color=C_GREEN, zorder=5,
               label="HighRes-net official bicubic\n(394 held-out scenes)")
    ax.annotate("24.68 vs floor 31.59\n= −6.92 dB", (6.63, hr_floor),
                textcoords="offset points", xytext=(-7, -26), ha="right",
                fontsize=6.6, color=C_GREEN)

    ax.axhline(0, color="k", lw=1.1)
    ax.axhspan(-14, 0, color="#e6d9f2", alpha=0.5, zorder=0)
    ax.text(2.02, -13.3, "below the MSE skill floor", fontsize=6.8,
            color="#5a2d73", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks([2.0, 3.0, 6.63])
    ax.set_xticklabels(["2×", "3×", "6.63×"])
    ax.minorticks_off()
    ax.set_xlim(1.8, 7.7)
    ax.set_ylim(-14, 31)
    ax.set_xlabel("Scale factor (log)")
    ax.set_ylabel("$\\Delta_{floor}$ = PSNR(bicubic) − PSNR(mean)  [dB]")
    ax.set_title("Recoverable-information gap collapses with factor", fontsize=9.2, pad=5)
    ax.grid(alpha=0.22, ls=":", which="both")
    ax.legend(loc="upper right", fontsize=6.6, framealpha=0.95, handlelength=1.6)
    fig.tight_layout()
    save(fig, "fig2_floor_collapse")


# =========================================================== Figure 3 ========
def fig_e1_decomposition() -> None:
    """Fig.3 —— E1 去相关分解。

    ⚠ 2026-09-26 起改用【场景匹配】纯尺度控制（e1_matched_aa.json，PRIMARY），
    旧 e1_scale_isolation.json（场景不匹配 tail60+aa）仅留作历史对照，不再入图。
    敏感性区间 = 2×2 网格（场景来源 × 降采样算子）四组合的 min–max。
    """
    xs = [2.0, 3.0, 4.0, 5.0, 6.63]
    KEYS = ["2.0", "3.0", "4.0", "5.0", "6.63"]
    NS = len(xs)

    def matched(fn):
        d = load(fn)["scales"]
        return [d[k]["delta_floor_matched_mean"] for k in KEYS]

    combos = ["e1_matched_aa_5pt.json", "e1_matched_noaa_5pt.json",
              "e1_tail60_aa_5pt.json", "e1_tail60_noaa_5pt.json"]
    grid = [matched(c) for c in combos]                  # 4 组合 × 5 倍率
    pure = matched(combos[0])                            # PRIMARY: matched + antialias
    lo = [min(g[i] for g in grid) for i in range(NS)]
    hi = [max(g[i] for g in grid) for i in range(NS)]

    # 真实臂仅在 3 个锚点倍率可得（数据/方法层各轴原生倍率）；4×/5× 无真实臂测点。
    real_xs = [2.0, 3.0, 6.63]
    real_d = [21.96, 2.09, -0.09]                                  # data layer
    real_m = [contrast(load("paired_s2v.json"), "bicubic_vs_mean")["mean_diff"],
              contrast(load("paired_probav.json"), "bicubic_vs_mean")["mean_diff"],
              contrast(load("paired_worldstrat.json"), "bicubic_vs_mean")["mean_diff"]]
    pure_at_real = [pure[KEYS.index(k)] for k in ["2.0", "3.0", "6.63"]]
    decorr = [m - p for p, m in zip(pure_at_real, real_m)]   # 负 = 真实臂低于纯尺度（去相关代价）

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    # 四组合敏感性区间（浅带）
    ax.fill_between(xs, lo, hi, color=C_GREEN, alpha=0.16, lw=0,
                    label="scene-set × operator range")
    ax.plot(xs, pure, "o-", color=C_GREEN, ms=5.5,
            label="pure-scale (scene-matched)")
    ax.plot(real_xs, real_d, "s--", color=C_BLUE, ms=5.5, label="real axes, data layer")
    ax.plot(real_xs, real_m, "^:", color=C_RED, ms=5.5, label="real axes, method layer")
    for s, p, dc in zip(real_xs, pure_at_real, decorr):
        ax.annotate(f"decorr. {dc:+.1f}", (s, p), textcoords="offset points",
                    xytext=(0, 9), ha="center", va="bottom",
                    fontsize=6.8, color=GREY)
    ax.axhline(0, color="k", lw=1.1)
    ax.axhspan(-12, 0, color="#e6d9f2", alpha=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["2×", "3×", "4×", "5×", "6.63×"], fontsize=7.6)
    ax.minorticks_off()
    ax.set_xlim(1.85, 8.5)
    ax.set_ylim(-13, 25)
    ax.set_xlabel("Scale factor (log)")
    ax.set_ylabel("$\\Delta_{floor}$ = PSNR(bicubic) − PSNR(mean)  [dB]")
    ax.set_title("Decorrelation dominates; cost grows with factor (E1)",
                 fontsize=9.2, pad=5)
    ax.grid(alpha=0.22, ls=":")
    ax.legend(loc="upper right", fontsize=6.0, framealpha=0.95, handlelength=1.6)
    fig.tight_layout()
    save(fig, "fig3_e1_decomposition")


# =========================================================== Figure 4 ========
def fig_crosssensor_3x() -> None:
    # 数据源（2026-09-30 定稿口径）：同集 22 ROI、有效重叠区、4 波段配对灰度，
    # 与 Table 1 的 Δ_model 同一次 eval 产出（paired_s2l8.json）；
    # 参照臂（纯尺度 3× / PROBA-V 3× 数据层）沿用 e1 敏感性文件的 reference_arms。
    ref = load("e1_real_crosssensor_3x.json")["reference_arms"]
    p = load("paired_s2l8.json")
    rows = p["per_roi"]
    gray = np.array([r["psnr_bicubic"] - r["psnr_mean"] for r in rows], float)
    pure = ref["pure_scale_3x_db"]
    probav = ref["probav_3x_samesensor_multitemp_db"]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    cols = [C_BLUE, C_GREEN, C_RED]
    for x, v, c in ((0, pure, cols[0]), (1, probav, cols[1])):
        ax.plot(x, v, "o", ms=8, color=c, zorder=5)
        ax.annotate(f"{v:.2f} dB", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.4,
                    fontweight="bold", color=c)

    rng = np.random.RandomState(0)
    jit = (rng.rand(len(gray)) - 0.5) * 0.16
    ax.scatter(np.full(len(gray), 2) + jit, gray, s=34, color=cols[2], zorder=4,
               edgecolor="white", lw=0.6, label="per ROI (valid overlap)")
    m = float(gray.mean())
    sd = float(gray.std(ddof=1)) if len(gray) > 1 else 0.0
    ax.errorbar(2, m, yerr=sd, fmt="none", ecolor=cols[2], capsize=4, lw=1.6, zorder=3)
    ax.plot(2, m, "D", ms=7, color="#4a1f5c", zorder=6)
    ax.annotate(f"mean {m:.2f}±{sd:.2f}", (2, m), textcoords="offset points",
                xytext=(0, -34), ha="center", va="top", fontsize=7.4,
                fontweight="bold", color=cols[2])

    ax.axhline(0.0, color=GREY, ls="--", lw=1.0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["pure-scale 3×\n(SPOT self-bic.)",
                        "PROBA-V 3×\n(same-sensor)",
                        "S2 / L8 3×\n(cross-sensor)"], fontsize=6.9)
    ax.set_ylabel("$\\Delta_{floor}$  [dB]")
    ax.set_title("Same-factor (3×) arms: scale vs decorrelation", fontsize=9.2, pad=5)
    ax.set_xlim(-0.4, 2.85)
    ax.set_ylim(-5.5, 11)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=6.4, framealpha=0.9, handlelength=1.4)
    ax.text(0.5, -0.30, f"n = {len(rows)} ROIs (S2 10 m / L8 30 m, native 3×)",
            transform=ax.transAxes, ha="center", fontsize=6.4, color=GREY)
    fig.tight_layout()
    save(fig, "fig4_crosssensor_3x")


def main() -> None:
    rows = axis_rows()
    print("=== per-sample axis table (single protocol) ===")
    for r in rows:
        print(f"  {r['key']:<7} D_floor={r['x']:+.3f}  D_model={r['y']:+.3f} "
              f"[{r['lo']:+.3f},{r['hi']:+.3f}]  p={r['p']:.3g}  n={r['n']}")
    fig_regime(rows)
    fig_floor_collapse()
    fig_e1_decomposition()
    fig_crosssensor_3x()
    print(f"\n[done] all figures -> {OUTDIR}")


if __name__ == "__main__":
    main()
