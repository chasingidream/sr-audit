#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supplementary Fig. 7 — the collapse is spatially structured and predictable
from the surface.

Panels
  (a) World map of the per-scene recoverable-information budget Δ_floor for all
      WorldStrat coverages (equirectangular; ~110m Natural Earth coastline).
      Negative (floor-failure) in purple, positive in blue, diverging at zero.
      Shows that the collapse is not uniform: it clusters over vegetated and
      seasonally changing regions.
  (b) Predicted vs observed Δ_floor from the zero-training physical-predictor
      regression (A′), with the OLS R² annotated — the budget is not merely
      spatially clustered but predictable from continuous surface properties.

Outputs (experiments/figs_final/):
  figS7_feasibility_map.{pdf,png}

Usage: python code/figs_figS7_map.py
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection

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

# 负=floor-failure(紫) → 0 → 正=positive(蓝)，与前图一致的色系
CMAP = LinearSegmentedColormap.from_list(
    "floor_map", ["#4a1d63", "#7b3294", "#c9a0dc", "#f2f2f2", "#8fb8dd",
                  "#2171b5", "#0b3d63"])


def load(name):
    with io.open(os.path.join(EXP, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, name + ".pdf"))
    fig.savefig(os.path.join(OUTDIR, name + ".png"), dpi=600)
    plt.close(fig)
    print(f"[saved] {name}.pdf / {name}.png")


def coast_segments():
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


def main():
    d = load("stratified_floor_ws.json")
    sc = [s for s in d["scenes"]
          if s.get("lat") is not None and s.get("delta_floor") is not None]
    lon = np.array([s["lon"] for s in sc])
    lat = np.array([s["lat"] for s in sc])
    df = np.array([s["delta_floor"] for s in sc])
    # 稳健色标：负值区间占满紫-浅紫梯度，正值（少数）走蓝端
    lo = float(np.percentile(df, 2))
    hi = float(np.percentile(df, 99.5))
    print(f"[info] n={len(df)} lon[{lon.min():.1f},{lon.max():.1f}] "
          f"lat[{lat.min():.1f},{lat.max():.1f}] df[{df.min():.2f},{df.max():.2f}] "
          f"p2/p99.5={lo:.2f}/{hi:.2f}")
    norm = TwoSlopeNorm(vmin=min(lo, -0.5), vcenter=0.0, vmax=max(hi, 0.3))

    fig = plt.figure(figsize=(7.4, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.35, 1.0, 0.09], wspace=0.16)

    # ---------------- (a) map ----------------
    ax = fig.add_subplot(gs[0, 0])
    for seg in coast_segments():
        ax.add_collection(LineCollection([seg], colors="#bbbbbb", linewidths=0.35,
                                         zorder=1))
    sct = ax.scatter(lon, lat, c=df, cmap=CMAP, norm=norm, s=6.5,
                     linewidths=0, zorder=3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-82, 83)
    ax.set_xlabel("longitude (°)")
    ax.set_ylabel("latitude (°)")
    ax.set_title("(a) Δ_floor is spatially structured", fontsize=8.2)
    ax.grid(alpha=0.15, lw=0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(sct, cax=cax)
    cb.set_label("Δ_floor  [dB]", fontsize=7.6)
    cb.ax.tick_params(labelsize=6.8)
    cb.ax.axhline(0, color="#333333", lw=0.8)

    # ---------------- (b) predictability ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    reg_path = os.path.join(EXP, "floor_predictors_regression.json")
    if os.path.exists(reg_path):
        reg = json.load(open(reg_path, encoding="utf-8"))
        feats = [f["feature"] for f in reg["features"]]
        X = np.array([[float(s[f]) for f in feats] for s in sc])
        yc = df - df.mean()
        Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
        beta = np.array([f["beta_std"] for f in reg["features"]])
        pred = Z @ beta + df.mean()
        ax2.scatter(df, pred, s=4.5, c="#7b3294", alpha=0.30, linewidths=0, zorder=3)
        lim = [max(lo, df.min()), min(hi, df.max())]
        ax2.plot(lim, lim, ls="--", lw=0.9, color="#555555", zorder=2)
        ax2.set_xlim(lim)
        ax2.set_ylim(lim)
        ax2.set_xlabel("observed Δ_floor  [dB]")
        ax2.set_ylabel("predicted Δ_floor  [dB]")
        ax2.set_title("(b) …and predictable from\nthe surface (zero training)",
                      fontsize=8.2)
        ax2.grid(alpha=0.18, lw=0.5)
        ax2.text(0.04, 0.94, f"OLS $R^2$ = {reg['r2']:.2f}\n(n = {reg['n']:,})",
                 transform=ax2.transAxes, fontsize=7.2, va="top", color="#4a1d63")
        for sp in ("top", "right"):
            ax2.spines[sp].set_visible(False)
    else:
        ax2.text(0.5, 0.5, "regression JSON missing", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=8, color="#888888")
        ax2.set_title("(b) predictability", fontsize=8.2)

    fig.suptitle("S7  The recoverable-information budget collapses where the land "
                 "surface says it should", fontsize=9.0, y=1.02)
    save(fig, "figS7_feasibility_map")


if __name__ == "__main__":
    main()
