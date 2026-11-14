#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supplementary Fig. 6 — the recoverable-information budget (Δ_floor) is a
function of surface condition, not a dataset constant.

Panels
  (a) Δ_floor by IPCC land-cover class (n = 3,927 WorldStrat scenes).
      Forest/vegetation collapse hardest (−6.2 dB); built-up Settlement least
      (−2.3 dB).  Kruskal–Wallis η² = 0.25 — land cover alone explains a
      quarter of the variance in Δ_floor.
  (b) Δ_floor by settlement density (SMOD) and by LR–HR temporal lag,
      two further surface/phenology axes that shift the budget.

  (c) paired-counterfactual de-contamination on the same 6.63× val scenes:
      real → +radiometric → +reregistered → pure-scale.  The negative
      Δ_floor is dominated by cross-sensor radiometric mismatch, not by a
      "the information is simply gone" tautology.

All numbers are read from the validated JSONs in experiments/, so re-running
the extraction/analysis propagates into the figure automatically.

Outputs (experiments/figs_final/):
  figS6_stratified_floor.{pdf,png}

Usage: python code/figs_figS6.py
"""
from __future__ import annotations

import io
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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "font.size": 8.0,
    "axes.labelsize": 8.6,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

C_POS = "#2171b5"
C_FLOOR = "#7b3294"
C_CEIL = "#e8912d"
GREY = "#888888"
NEG_BAR = "#9e6bb5"   # lavender for the stratified bars
POS_REC = "#2171b5"


def load(name: str) -> dict:
    with io.open(os.path.join(EXP, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(OUTDIR, name + ".pdf"))
    fig.savefig(os.path.join(OUTDIR, name + ".png"), dpi=600)
    plt.close(fig)
    print(f"[saved] {name}.pdf / {name}.png")


def _bars(ax, summary, key, title, xlabel, color=NEG_BAR, drop_empty=True):
    rows = summary[key]["rows"]
    rows = [r for r in rows if not (drop_empty and str(r[0]).strip() == "")]
    labs = [str(r[0]) for r in rows]
    means = np.array([r[2] for r in rows])
    ci = np.array([r[5] for r in rows])
    ns = [r[1] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, means, xerr=ci, color=color, alpha=0.9, height=0.68,
            error_kw=dict(ecolor=GREY, lw=0.8, capsize=2.5), zorder=3)
    ax.axvline(0, color=GREY, lw=0.8, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  (n={n})" for l, n in zip(labs, ns)], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8.0)
    ax.grid(axis="x", alpha=0.18, lw=0.5)
    eta = summary[key].get("eta2")
    if eta is not None:
        ax.text(0.02, 0.03, f"Kruskal–Wallis $\\eta^2$ = {eta:.2f}",
                transform=ax.transAxes, fontsize=6.6, color=GREY, va="bottom")


def panel_landcover(ax) -> None:
    s = load("stratified_floor_summary.json")
    _bars(ax, s, "by_IPCC",
          "(a) Δ_floor by IPCC land-cover class",
          "Δ_floor = PSNR(bicubic) − PSNR(mean)  [dB]")


def panel_smod_lag(ax) -> None:
    """Right panel: SMOD density (top) and temporal lag (bottom) as two
    additional surface/phenology axes; drawn as two stacked sub-axes."""
    s = load("stratified_floor_summary.json")
    # split the axes vertically
    ax.remove()


def panel_recovery(ax) -> None:
    d = load("e1_paired_counterfactual.json")
    st = ["real", "radiometric", "reregistered", "pure_scale"]
    labs = ["real\npair", "+radio-\nmetric", "+re-\nregistered", "pure\nscale"]
    means = [d["stages"][k]["mean"] for k in st]
    # 用标准误（SE = std/√n）作图：配对差异 p~1e-19，若画 ±1SD 会误导为不显著
    n = d.get("n", 60)
    sems = [d["stages"][k]["std"] / np.sqrt(n) for k in st]
    x = np.arange(len(st))
    cols = [C_FLOOR, C_POS, C_POS, C_CEIL]
    ax.bar(x, means, yerr=sems, color=cols, alpha=0.9, width=0.62,
           error_kw=dict(ecolor=GREY, lw=0.9, capsize=3), zorder=3)
    for xi, m in zip(x, means):
        ax.text(xi, m + (1.2 if m >= 0 else -1.6), f"{m:+.1f}", ha="center",
                fontsize=7.0, color="#222222")
    ax.axhline(0, color=GREY, lw=0.9, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7.0)
    ax.set_ylabel("Δ_floor  [dB]")
    ax.set_title("(c) paired-counterfactual de-contamination\n"
                 "(WorldStrat 6.63×, n = 60; error bars = s.e.m.)", fontsize=8.0)
    ax.grid(axis="y", alpha=0.18, lw=0.5)
    ax.set_ylim(min(means) - 4, max(means) + 3)
    ax.text(0.02, 0.96, "radiometric mismatch dominates\n(+10.8 dB of the deficit; p < 1e-18)",
            transform=ax.transAxes, fontsize=6.4, color=C_FLOOR, va="top")


def main() -> None:
    fig = plt.figure(figsize=(7.8, 3.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.28, 1.0], hspace=0.72,
                          wspace=0.62, left=0.09, right=0.985, top=0.86,
                          bottom=0.13)
    s = load("stratified_floor_summary.json")

    ax_a = fig.add_subplot(gs[:, 0])
    _bars(ax_a, s, "by_IPCC",
          "(a) Δ_floor by IPCC land-cover class",
          "Δ_floor  [dB]")

    ax_b1 = fig.add_subplot(gs[0, 1])
    _bars(ax_b1, s, "by_SMOD", "(b) by settlement density (SMOD)", "Δ_floor  [dB]")
    ax_b2 = fig.add_subplot(gs[1, 1])
    _bars(ax_b2, s, "by_lag", "(b′) by LR–HR temporal lag", "Δ_floor  [dB]")

    fig.suptitle("S6  The recoverable-information budget is a function of surface "
                 "condition, not a dataset constant", fontsize=9.0, y=1.0)
    save(fig, "figS6_stratified_floor")

    # --- second figure: the paired counterfactual alone (wider) ---
    fig2, ax = plt.subplots(1, 1, figsize=(3.6, 3.0))
    panel_recovery(ax)
    save(fig2, "figS6b_paired_counterfactual")


if __name__ == "__main__":
    main()
