# -*- coding: utf-8 -*-
"""C-4：栅格化产品 + 双档不确定性层 + 格网级精度。

交付什么
  1. **格网产品**（1°×1° 与 0.5°×0.5°）：每格网一行，含样本数、Δ_floor 的
     观测统计、模型预测、双档不确定性标记。诚实边界：**只覆盖有观测的格网**，
     不对无数据区外推（3,927 景全球覆盖极稀疏，凭空外推会造假）。
  2. **格网级精度**（图集产品的真实可用性）：把"留一格网"作为 CV 的 fold，
     检验聚合到格网后模型还准不准 —— 这是审稿人会问的"图集能不能用"。
  3. **双档不确定性**：由已测的泛化口径给出，而非主观设定
       档 A「地理插值」  = 目标格网附近有观测  → 用 buffered CV / 细网格 CV
       档 B「体制外推」  = 目标区域无同类观测  → 用留战役 / 留国家 CV

输出
  experiments/pathc_atlas_1deg.csv / pathc_atlas_05deg.csv
  experiments/pathc_atlas.json
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M  # noqa: E402
import pathc_dem_analysis as DA  # noqa: E402
import pathc_climate as C  # noqa: E402

# 最优特征集（C-5 判定：tex+dem = 0.508 最高；加气候量反而降到 0.448）
KEYS = C.BASE_TEX + ["dem_std", "slope_mean", "tri_mean", "relief",
                     "roughness", "dem_mean"]

# 双档精度的来源（C-3 / C-2 实测）
TIER_A = {"name": "geographic_interpolation",
          "r2_buffered30": 0.631, "r2_fine_grid": 0.656,
          "use": "target cell has observations nearby"}
TIER_B = {"name": "regime_extrapolation",
          "r2_leave_campaign": 0.508, "r2_leave_country": 0.505,
          "use": "no comparable acquisition regime in the target region"}


def build():
    rows = M.load()
    C.merge(rows)                       # 并入 WorldClim + DEM
    y = np.array([r["delta_floor"] for r in rows], float)
    ok = np.isfinite(y)
    rows = [r for r, o in zip(rows, ok) if o]
    y = y[ok]
    X = DA.matrix(rows, KEYS)
    m = np.isfinite(X).all(axis=1)
    return [r for r, o in zip(rows, m) if o], X[m], y[m]


def cell_id(lat, lon, res):
    return (int(np.floor(lat / res)), int(np.floor(lon / res)))


def atlas(rows, X, y, res, fitter):
    """按格网聚合；预测值来自全量训练好的模型（非 CV，仅作产品列）。"""
    keys = [cell_id(r["lat"], r["lon"], res) for r in rows]
    uniq = sorted(set(keys))
    karr = np.array([uniq.index(k) for k in keys])
    pred = fitter(X, y, X)

    out = []
    for gi, k in enumerate(uniq):
        sel = karr == gi
        ys, ps = y[sel], pred[sel]
        n = int(sel.sum())
        # 观测统计（逐景 PSNR 平均口径，禁止池化）
        rec = {
            "cell_lat": round(k[0] * res, 3),
            "cell_lon": round(k[1] * res, 3),
            "res_deg": res,
            "n_scenes": n,
            "dfloor_mean": round(float(ys.mean()), 3),
            "dfloor_median": round(float(np.median(ys)), 3),
            "dfloor_sd": round(float(ys.std()), 3) if n > 1 else "",
            "dfloor_min": round(float(ys.min()), 3),
            "dfloor_max": round(float(ys.max()), 3),
            "model_pred_mean": round(float(ps.mean()), 3),
            "model_bias": round(float((ps - ys).mean()), 3),
            "model_rmse": round(float(np.sqrt(((ps - ys) ** 2).mean())), 3),
            # 该格网内部的空间一致性（多景时的格内可预测性上界）
            "within_cell_r2": (round(float(M.r2_of(ys, ps)), 3) if n > 5 else ""),
        }
        out.append(rec)
    return out, uniq, karr


def cell_cv(rows, X, y, res, fitter):
    """留一格网 CV —— 图集产品的真实精度。"""
    keys = np.array([cell_id(r["lat"], r["lon"], res) for r in rows], dtype=np.int64)
    uniq = sorted(set(map(tuple, keys.tolist())))
    P = np.column_stack([np.array([r["lat"] for r in rows], float),
                         np.array([r["lon"] for r in rows], float)])
    pred = np.full(len(y), np.nan)
    for k in uniq:
        te = np.where((keys[:, 0] == k[0]) & (keys[:, 1] == k[1]))[0]
        tr = np.where((keys[:, 0] != k[0]) | (keys[:, 1] != k[1]))[0]
        if len(tr) < 100 or len(te) == 0:
            continue
        pred[te] = fitter(X[tr], y[tr], X[te])
    m = np.isfinite(pred)
    if m.sum() < 50:
        return float("nan"), int(m.sum()), 0
    return (float(M.r2_of(y[m], pred[m])), int(m.sum()), len(uniq))


def main():
    rows, X, y = build()
    print(f"[build] n={len(rows)} p={X.shape[1]}", flush=True)
    # 图集产品的格网级 CV：格网数可达数千，逐格网重训 GBDT 会超时，
    # 故 CV 用 ridge（闭式解，每折瞬时）；全量模型仍用 GBDT 产出预测值。
    fit_pred = M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)
    fit_cv = M.make_ridge(10.0)
    # 粗格网（5°，663 个）用小树 GBDT 也能在超时内跑完，给出更接近产品的估计
    fit_fast = M.make_gbdt(n_trees=20, lr=0.30, depth=4, min_leaf=30, seed=1)

    rep = {"n_scenes": int(len(rows)), "n_feat": int(X.shape[1]), "keys": KEYS,
           "tier_A": TIER_A, "tier_B": TIER_B, "grids": {}}

    # 分辨率列表。⚠ 格网数随 res 急剧下降：0.5°→2773、1°→2273、5°→663、
    # 10°→约 200。GBDT 逐格网重训的代价 ∝ 格网数，10°(200 折 × 1.5 s ≈ 300 s)
    # 在本沙箱 4 min 上限内跑不稳，故默认只出 5°/1°/0.5° 三档，CV 用 ridge
    # （闭式）作**保守下界**。要 GBDT 版格网 CV 需分段跑，见文档 §C-4。
    for res in (5.0, 1.0, 0.5):
        recs, uniq, karr = atlas(rows, X, y, res, fit_pred)
        # 格网数少时用 GBDT（更接近产品实际模型），多时用 ridge（闭式，避免超时）
        n_cells_est = len({cell_id(r["lat"], r["lon"], res) for r in rows})
        f = fit_pred if n_cells_est <= 400 else (fit_fast if n_cells_est <= 700
                                                  else fit_cv)
        r2, npred, ncell = cell_cv(rows, X, y, res, f)
        cs = np.array([r["n_scenes"] for r in recs])
        dfm = np.array([r["dfloor_mean"] for r in recs])
        out_csv = os.path.join(M.EXP, f"pathc_atlas_{str(res).replace('.', '')}deg.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print(f"\n[{res}°] 格网数={ncell}  非空格网={len(recs)}  "
              f"每格网景数 中位={np.median(cs):.0f} 最大={cs.max()}", flush=True)
        print(f"      Δ_floor 格网均值 range=[{dfm.min():.2f}, {dfm.max():.2f}] dB  "
              f"s.d.={dfm.std():.2f}")
        print(f"      **留一格网 CV R² = {r2:.3f}**  (n_pred={npred}, "
              f"CV 模型={'GBDT' if f is fit_pred else 'ridge'})", flush=True)
        print(f"      → {out_csv}", flush=True)
        rep["grids"][f"{res}"] = {
            "n_cells": int(ncell), "n_nonempty": len(recs),
            "scenes_per_cell_median": float(np.median(cs)),
            "scenes_per_cell_max": int(cs.max()),
            "dfloor_cellmean_range": [float(dfm.min()), float(dfm.max())],
            "dfloor_cellmean_sd": float(dfm.std()),
            "leave_cell_cv_r2": r2, "n_pred": int(npred),
            "cv_model": ("GBDT60" if f is fit_pred
                         else "GBDT20" if f is fit_fast else "ridge"),
            "csv": os.path.basename(out_csv),
        }
        json.dump(rep, open(os.path.join(M.EXP, "pathc_atlas.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n[双档不确定性层]")
    for t in (TIER_A, TIER_B):
        print(f"  {t['name']:<28s} R²≈{max(v for k, v in t.items() if k.startswith('r2')):.3f}"
              f"   用途: {t['use']}")
    print("\n[ok] wrote pathc_atlas.json + 2 CSV")


if __name__ == "__main__":
    main()
