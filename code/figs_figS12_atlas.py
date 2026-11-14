#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supplementary Fig. S12 — the recoverable-information budget atlas and its
uncertainty tiers.

Panels
  (a) World map of the 1°-cell mean Δ_floor (ESA WorldCover–free; only cells
      with observations), with the two uncertainty tiers marked in the legend:
      tier A "geographic interpolation" and tier B "regime extrapolation".
  (b) Precision spectrum: honest R² under every cross-validation scheme we ran,
      ordered from optimistic to strict, with the 0.4 usability threshold drawn.
      This is the panel that answers "how good is the atlas, really?".
  (c) Where the atlas's skill comes from: the incremental R² of each static
      variable block (image texture / terrain / climate) on top of the base
      image features, under leave-one-campaign-out CV, with 95% paired-bootstrap
      CIs, shown under both the raw and the winsorised (1–99%) Δ_floor target.
  (d) Site-granularity dependence: R² of leave-one-station-out CV against the
      number of defined stations, with the DEM increment overlaid, showing that
      DEM earns its keep exactly when the held-out unit is large.
  (e) Why terrain is not a free lunch: ΔMAE of the terrain block over image
      texture, stratified by Δ_floor quintile.

Outputs (experiments/figs_final/):
  figS12_atlas_tiers.{pdf,png}

Usage: python code/figs_figS12_atlas.py
"""
from __future__ import annotations

import csv
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
OUTDIR = os.path.join(EXP, "figs_final")
COAST = os.path.join(ROOT, "data", "raw", "ne_110m_coastline.geojson")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "font.size": 8.0,
    "axes.labelsize": 8.6,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

C_POS = "#2171b5"     # positive Δ_floor / tier A
C_NEG = "#7b3294"     # negative Δ_floor / tier B
C_DEM = "#C98A1E"     # terrain block
C_CLIM = "#2C6E9B"    # climate block
C_WC = "#3E8E7E"     # land-cover block
C_TEX = "#7B4FA8"     # image-texture block
C_MID = "#9AA5AE"
CMAP = LinearSegmentedColormap.from_list(
    "atlas", ["#4a1d63", "#7b3294", "#c9a0dc", "#f2f2f2", "#8fb8dd",
              "#2171b5", "#0b3d63"])


def load(name):
    with io.open(os.path.join(EXP, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, name + ".pdf"))
    fig.savefig(os.path.join(OUTDIR, name + ".png"), dpi=600)
    plt.close(fig)
    print(f"[saved] {name}.pdf / .png")


def coast_segments():
    if not os.path.exists(COAST):
        return []
    d = json.load(open(COAST, encoding="utf-8"))
    segs = []
    for feat in d.get("features", []):
        geom = feat.get("geometry", {})
        t, coords = geom.get("type"), geom.get("coordinates", [])
        if t == "LineString":
            segs.append(np.array(coords))
        elif t == "MultiLineString":
            segs.extend(np.array(c) for c in coords)
        elif t == "Polygon":
            segs.append(np.array(coords[0]))
        elif t == "MultiPolygon":
            for poly in coords:
                segs.append(np.array(poly[0]))
    return segs


def panel_a(ax, cax=None):
    """1°-cell mean Δ_floor map + observation density."""
    path = os.path.join(EXP, "pathc_atlas_10deg.csv")
    if not os.path.exists(path):
        ax.text(0.5, 0.5, "atlas csv missing", ha="center", va="center")
        ax.set_axis_off()
        return None
    la, lo, df, n = [], [], [], []
    with io.open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            la.append(float(r["cell_lat"]))
            lo.append(float(r["cell_lon"]))
            df.append(float(r["dfloor_mean"]))
            n.append(int(r["n_scenes"]))
    la, lo, df, n = map(np.array, (la, lo, df, n))
    d = 1.0
    vmax = float(np.percentile(df, 98))
    norm = TwoSlopeNorm(vmin=float(np.percentile(df, 2)), vcenter=0.0,
                        vmax=max(vmax, 0.3))
    segs = coast_segments()
    if segs:
        ax.add_collection(LineCollection(segs, colors="#5b5b5b", linewidths=0.28,
                                         zorder=1))
    # 观测少的格子画小点，多的画方块 —— 避免"稀疏地区被当成缺测"
    for size, sel in ((9, n == 1), (26, n == 2), (60, n >= 3)):
        m = sel if sel is not True else n >= 3
        if not np.any(m):
            continue
        ax.scatter(lo[m], la[m], c=df[m], cmap=CMAP, norm=norm, s=size,
                   marker="s", linewidths=0, zorder=3)
    ax.scatter([], [], s=9, marker="s", c="#999", label="cell with 1 scene")
    ax.scatter([], [], s=26, marker="s", c="#666", label="2 scenes")
    ax.scatter([], [], s=60, marker="s", c="#333", label="≥3 scenes")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.grid(True, lw=0.2, color="#dddddd", zorder=0)
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    ax.set_title("(a) Cell-mean recoverable budget Δ_floor\n"
                 "1° cells, marker size = scenes per cell", loc="left")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = plt.colorbar(sm, cax=cax) if cax is not None else \
        plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("cell-mean Δ_floor (dB)", fontsize=7.6)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, handletextpad=0.3,
              borderpad=0.4, labelspacing=0.35)
    return norm


def panel_b(ax):
    """Precision spectrum across every CV scheme, ordered optimistic→strict."""
    cellcv = load("pathc_cellcv.json")
    buf = load("pathc_buffered_cv.json")
    st = load("pathc_stations.json")
    dem = load("pathc_dem_analysis.json")
    models = load("pathc_models.json")

    items = []
    g = cellcv["grids"]["1.0"]["kfold_by_cell_r2"]
    items.append(("K-fold by cell (1°, atlas product)", g, C_POS))
    b30 = buf["blocks"]["30x60"]
    items.append(("Spatial block CV 30°×60°",
                  b30["0.0"]["cv_r2"], C_POS))
    items.append(("Buffered CV, 30° buffer", b30["30.0"]["cv_r2"], C_POS))
    items.append(("Leave-one-campaign-out",
                  st["S4_campaign_dem"]["cv_r2"], C_NEG))
    items.append(("Leave-one-country-out (91)",
                  st["S87_country_dem"]["cv_r2"], C_NEG))
    # 尺度无关复核（corr_bic_hr 口径）
    cc = models.get("corr_bic_hr", {})
    for k, v in cc.items():
        if k.endswith("gbdt60"):
            items.append(("Scale-free corr cross-check", v["cv_leave_campaign"], C_MID))
            break

    items.sort(key=lambda t: -t[1])
    y = np.arange(len(items))
    vals = np.array([v for _, v, _ in items])
    cols = [c for _, _, c in items]
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="none")
    ax.axvline(0.4, color="#444", lw=0.9, ls="--", zorder=3)
    ax.text(0.4, -0.62, "0.4 usability", fontsize=6.6, color="#444",
            va="bottom", ha="center")
    for yi, (lab, v, _) in enumerate(items):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.0)
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _, _ in items], fontsize=7.0)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.85, vals.max() * 1.12))
    ax.set_ylim(len(items) - 0.4, -0.95)
    ax.set_xlabel("R² of Δ_floor (leave-out schemes, GBDT)")
    ax.set_title("(b) Precision spectrum: what the atlas\n"
                 "can and cannot claim", loc="left")
    ax.grid(axis="x", lw=0.3, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)


BLOCKS = [("image texture\n(SPOT |∇HR|)", "texture", C_TEX),
          ("terrain\n(AWS DEM)", "dem", C_DEM),
          ("land cover\n(ESA WorldCover)", "worldcover", C_WC),
          ("climate\n(WorldClim)", "climate", C_CLIM)]


def panel_c(ax):
    """Incremental R² of each static block, with 95% bootstrap CIs, under two
    target definitions: raw Δ_floor and winsorised Δ_floor (1st/99th pct).

    ⚠ 两个口径必须同时给：pooled R² 由 2.4% 的极端 Δ_floor 景主导，
    地形块在原始口径 +0.034（CI 排除 0）而在缩尾口径 −0.001（CI 跨 0），
    单报任何一个都会误导。
    """
    raw = load("pathc_increment_ci.json")["increments"]
    win = load("pathc_increment_ci_w199.json")["increments"]
    x = np.arange(len(BLOCKS))
    w = 0.36
    series = (("raw Δ_floor", raw, "#555555"),
              ("winsorised 1–99%", win, "#BFBFBF"))
    for s, (lab, src, col) in enumerate(series):
        v = np.array([src[k]["point"] for _, k, _ in BLOCKS])
        lo = np.array([src[k]["ci_lo"] for _, k, _ in BLOCKS])
        hi = np.array([src[k]["ci_hi"] for _, k, _ in BLOCKS])
        xs = x + (s - 0.5) * w
        ax.bar(xs, v, width=w, color=col, edgecolor="none", label=lab,
               zorder=3)
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor="#333",
                    elinewidth=0.7, capsize=2.0, zorder=4)
    ax.axhline(0, color="#222", lw=0.8, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([b for b, _, _ in BLOCKS], fontsize=6.4, rotation=18,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("Δ R² over the base image features")
    ax.set_ylim(-0.050, 0.070)
    ax.set_title("(c) Only image texture survives a\n"
                 "tail-robust re-check", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=6.2)
    ax.grid(axis="y", lw=0.3, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)


def panel_e(ax):
    """Where terrain actually pays: ΔMAE of +DEM over +texture, by Δ_floor quintile.

    原始口径下 DEM 降 RMSE（−0.113，P=1.000）却几乎不动 MedAE（−0.004，P=0.634），
    ⇒ 收益必然集中在尾段。分层证实了这一点：只在 Δ_floor 最负的两层改善，
    在有余量的两层反而变差。
    """
    st = load("pathc_increment_strata.json")
    rows = st["strata"]
    x = np.arange(len(rows))
    v = np.array([r["delta_mae"] for r in rows])
    lo = np.array([r["mae_ci"][0] for r in rows])
    hi = np.array([r["mae_ci"][1] for r in rows])
    cols = [C_DEM if a < 0 else C_MID for a in v]
    ax.bar(x, v, width=0.55, color=cols, edgecolor="none", zorder=3)
    ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="#333",
                elinewidth=0.7, capsize=2.0, zorder=4)
    ax.axhline(0, color="#222", lw=0.8, zorder=5)
    for xi, (r, a) in enumerate(zip(rows, v)):
        ax.text(xi, a + (0.006 if a >= 0 else -0.020), f"{a:+.3f}",
                ha="center", fontsize=6.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['q']}\n(r={r['range'][0]},{r['range'][1]}]"
                        for r in rows], fontsize=5.8)
    ax.set_xlabel("Δ_floor quintile (dB)  —  Q1 = budget exhausted", fontsize=7.0)
    ax.set_ylabel("Δ MAE of + terrain (dB)")
    ax.set_ylim(-0.22, 0.13)
    ax.set_title("(e) Terrain helps only where the budget is\n"
                 "already exhausted, and hurts where it is not", loc="left")
    ax.grid(axis="y", lw=0.3, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)


def panel_d(ax):
    """Site granularity vs R², with the DEM increment overlaid."""
    st = load("pathc_stations.json")
    dem = load("pathc_dem_analysis.json")
    labels = ["S4\ncampaign\n(4)", "S87\ncountry\n(91)", "S_geo\n10° grid\n(242)",
              "S_mixed\n(327)"]
    keys = ["S4_campaign", "S87_country", "S_geo", "S_mixed"]
    n_st = [4, 91, 242, 327]
    r_img, r_dem, inc = [], [], []
    for k in keys:
        a = st[k]["cv_r2"]
        b = st[k + "_dem"]["cv_r2"]
        r_img.append(a)
        r_dem.append(b)
        inc.append(b - a)
    x = np.arange(len(keys))
    ax.plot(x, r_img, "-o", color=C_POS, lw=1.1, ms=4.2, label="image features only")
    ax.plot(x, r_dem, "-s", color=C_DEM, lw=1.1, ms=4.2,
            label="+ terrain (DEM)")
    for xi, (a, b) in enumerate(zip(r_img, r_dem)):
        ax.plot([xi, xi], [a, b], color=C_MID, lw=0.8, ls=":", zorder=0)
        ax.text(xi - (0.14 if xi == len(keys) - 1 else 0), min(a, b) - 0.030,
                f"{b - a:+.3f}", fontsize=6.6,
                color="#555", va="top",
                ha="right" if xi == len(keys) - 1 else "center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.9)
    ax.set_ylabel("Leave-one-station-out R²")
    ax.set_title("(d) Terrain pays exactly when the\nheld-out unit is large",
                 loc="left")
    ax.legend(loc="center right", frameon=False, fontsize=6.9)
    ax.grid(axis="y", lw=0.3, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0.28, 0.80)


def main():
    fig = plt.figure(figsize=(7.6, 9.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 0.92],
                          width_ratios=[1.0, 1.06, 0.035],
                          hspace=0.52, wspace=0.42,
                          left=0.075, right=0.975, top=0.950, bottom=0.062)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_cax = fig.add_subplot(gs[0, 2])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d2 = fig.add_subplot(gs[1, 2])
    ax_d2.axis("off")
    ax_e = fig.add_subplot(gs[2, :2])
    panel_a(ax_a, ax_cax)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)
    panel_e(ax_e)
    save(fig, "figS12_atlas_tiers")


if __name__ == "__main__":
    main()
