# -*- coding: utf-8 -*-
"""C-5：第二类独立静态量（WorldClim 气候学量）——检验 DEM 结论的可复现性。

问题
  C-1 用 DEM 地形量提供了独立于 HR 影像的第二来源证据（留站点 +0.034）。
  但地形只是**静态几何**。真正独立于一切卫星产品的另一类是**气候学**：
  WorldClim 2.1（Fick & Hijmans 2017）由全球气象站观测空间插值得到，
  与任何传感器/处理链都不重叠，且是 1970–2000 的多年代气候学（时间上恒定）。

问题
  1. 气候量与 DEM 地形量是否正交？
  2. 在 DEM 之外，气候量能否提供增量（即"三类独立静态量"是否层层递进）？
  3. 结论是否可复现：若加入气候量的增益与 DEM 同号同量级，则 C-1 的
     "独立静态量提供正交增量"不是 DEM 的偶然。

⚠ 与 DEM 的冗余：WorldClim 的 `elevation` 与 DEM 的 `dem_mean` 同源，
  故气候集**不含 elevation**，只用降水/气温量。

输出
  experiments/pathc_climate_static.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M  # noqa: E402
import pathc_dem_analysis as DA  # noqa: E402

# ⚠ 必须复用 C-1 的 matrix：它带 DERIVE 映射（log_tex<-texture_hr、
# abs_lat<-lat、abs_dt_log<-delta_days、log_area<-area_km2、log_lccs<-lccs）。
# 自写 matrix 会漏掉这些派生量 → 整列 NaN → 公共子集为 0（静默失败）。
matrix = DA.matrix
load_dem = DA.load_dem

CLIM = os.path.join(M.EXP, "climate_independent.jsonl")
CLIM_KEYS = ["log_precip", "t_mean_ann", "t_seasonality", "precip_cv",
             "precip_seasonality"]
DEM_KEYS = ["dem_std", "slope_mean", "tri_mean", "relief", "roughness", "dem_mean"]
# 与 pathc_models.engineer level1 的 leak-free 基础特征（去掉 log_tex）
# ⚠ 列顺序必须与 pathc_dem_analysis.BASE_KEYS 完全一致，否则 GBDT 的特征采样
# 顺序不同 → 同一特征集给出不同 R²（实测 0.474 vs 0.491），增量分解将不可比。
BASE_NO_TEX = ["ndvi_mean", "ndvi_std", "bright_lr", "spec_het", "bright_hr",
               "abs_lat", "cloud", "abs_dt_log", "lon", "log_area", "log_lccs"]
BASE_TEX = ["ndvi_mean", "ndvi_std", "bright_lr", "spec_het", "bright_hr",
            "log_tex", "abs_lat", "cloud", "abs_dt_log", "lon",
            "log_area", "log_lccs"]


def load_clim():
    out = {}
    for line in open(CLIM, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        p = r.get("precip_ann")
        r["log_precip"] = float(np.log10(max(p, 0.1))) if p is not None else None
        out[r["cov"]] = r
    return out


def merge(rows):
    """并入 WorldClim 气候学量 + DEM 地形量（后者来自 C-1）。"""
    clim = load_clim()
    dem = load_dem()
    n = n_d = 0
    for r in rows:
        c = clim.get(r["cov"])
        if c:
            n += 1
            for k in CLIM_KEYS:
                v = c.get(k)
                if v is not None:
                    r[k] = float(v)
        d = dem.get(r["cov"])
        if d:
            n_d += 1
            for k in DEM_KEYS:
                v = d.get(k)
                if v is not None:
                    r[k] = float(v)
    return n, n_d


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="从第几个特征集开始（分段推进）")
    a0 = ap.parse_args()
    rows = M.load()
    n_clim, n_dem = merge(rows)
    print(f"[load] rows={len(rows)}  climate={n_clim}  dem={n_dem}", flush=True)
    y = np.array([r["delta_floor"] for r in rows], float)
    ok_y = np.isfinite(y)
    rows = [r for r, o in zip(rows, ok_y) if o]
    y = y[ok_y]

    OP = os.path.join(M.EXP, "pathc_climate_static.json")
    try:
        rep = json.load(open(OP, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        rep = {}
    rep.update({"n": int(len(rows)), "n_climate": n_clim, "n_dem": n_dem,
                "clim_keys": CLIM_KEYS})

    # ---------- 1. 正交性：气候 vs DEM vs texture ----------
    dem = matrix(rows, DEM_KEYS)
    tex = matrix(rows, ["texture_hr"])[:, 0]
    print("\n[1] 正交性（Pearson r）")
    print(f"{'气候量':<20s} {'vs DEM量':>12s} {'vs texture':>12s} {'vs Δ_floor':>12s}")
    ortho = {}
    for j, ck in enumerate(CLIM_KEYS):
        cv = matrix(rows, [ck])[:, 0]
        m = np.isfinite(cv) & np.isfinite(dem).all(axis=1) & np.isfinite(tex) & np.isfinite(y)
        if m.sum() < 50:
            continue
        rd = np.array([stats.pearsonr(cv[m], dem[m, q])[0] for q in range(dem.shape[1])])
        rt = stats.pearsonr(cv[m], tex[m])[0]
        ry = stats.pearsonr(cv[m], y[m])[0]
        py = stats.pearsonr(cv[m], y[m])[1]
        print(f"{ck:<20s} {np.abs(rd).max():+12.3f} {rt:+12.3f} "
              f"{ry:+12.3f} (p={py:.1e})")
        ortho[ck] = {"max_abs_r_vs_dem": float(np.abs(rd).max()),
                     "r_vs_texture": float(rt),
                     "r_vs_dfloor": float(ry), "p_vs_dfloor": float(py),
                     "n": int(m.sum())}
    rep["orthogonality"] = ortho

    # ---------- 2. 特征集阶梯 ----------
    dem_ok = np.isfinite(dem).all(axis=1)
    clim_ok = np.isfinite(matrix(rows, CLIM_KEYS)).all(axis=1)
    common = dem_ok & clim_ok
    print(f"\n[2] 特征集阶梯（**公共子集 n={common.sum()}**，留一战役 CV，GBDT 60 树）")

    sets = {
        "P_base(-tex,-dem,-clim)": BASE_NO_TEX,
        "T_base+tex": BASE_TEX,
        "D_base+dem": BASE_NO_TEX + DEM_KEYS,
        "C_base+clim": BASE_NO_TEX + CLIM_KEYS,
        "TD_base+tex+dem": BASE_TEX + DEM_KEYS,
        "TC_base+tex+clim": BASE_TEX + CLIM_KEYS,
        "TDC_all": BASE_TEX + DEM_KEYS + CLIM_KEYS,
    }
    fit = M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)
    print(f"{'集合':<30s} {'p':>3s} {'in':>6s} {'CV30x60':>8s} {'leaveCamp':>10s}")
    res = rep.get("featsets", {})
    for si, (name, keys) in enumerate(sets.items()):
        if si < a0.start:
            continue
        X = matrix(rows, keys)
        ok = np.isfinite(X).all(axis=1) & common
        Xo, yo = X[ok], y[ok]
        sub = [r for r, o in zip(rows, ok) if o]
        if len(yo) < 50:
            print(f"{name:<30s} {len(keys):3d}  子集不足({len(yo)})，跳过", flush=True)
            continue
        ins = float(M.r2_of(yo, fit(Xo, yo, Xo)))
        c30 = M.cv_spatial(sub, Xo, yo, 30, 60, fit)[0]
        lc, per = M.cv_group(sub, Xo, yo, "campaign", fit)
        print(f"{name:<30s} {len(keys):3d} {ins:6.3f} {c30:8.3f} {lc:10.3f}", flush=True)
        res[name] = {"p": len(keys), "n": int(len(yo)), "in_r2": ins,
                     "cv_30x60": c30, "cv_leave_campaign": lc,
                     "per_campaign": per, "keys": keys}
        rep["featsets"] = res
        json.dump(rep, open(OP, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    # ---------- 3. 判据 ----------
    g = lambda k: res[k]["cv_leave_campaign"]   # noqa: E731
    print("\n[3] 判据（留一战役 R² 的增量分解）")
    print(f"  base                       {g('P_base(-tex,-dem,-clim)'):.3f}")
    print(f"  + tex        增量 = {g('T_base+tex') - g('P_base(-tex,-dem,-clim)'):+.3f}")
    print(f"  + dem        增量 = {g('D_base+dem') - g('P_base(-tex,-dem,-clim)'):+.3f}")
    print(f"  + clim       增量 = {g('C_base+clim') - g('P_base(-tex,-dem,-clim)'):+.3f}")
    print(f"  + tex+dem    增量 = {g('TD_base+tex+dem') - g('T_base+tex'):+.3f}  "
          f"(vs base {g('TD_base+tex+dem') - g('P_base(-tex,-dem,-clim)'):+.3f})")
    print(f"  + tex+clim   增量 = {g('TC_base+tex+clim') - g('T_base+tex'):+.3f}  "
          f"(vs base {g('TC_base+tex+clim') - g('P_base(-tex,-dem,-clim)'):+.3f})")
    print(f"  + tex+dem+clim  总量 = {g('TDC_all') - g('P_base(-tex,-dem,-clim)'):+.3f}"
          f"   → R² = {g('TDC_all'):.3f}")
    rep["verdict"] = {
        "base": g("P_base(-tex,-dem,-clim)"),
        "inc_tex": g("T_base+tex") - g("P_base(-tex,-dem,-clim)"),
        "inc_dem": g("D_base+dem") - g("P_base(-tex,-dem,-clim)"),
        "inc_clim": g("C_base+clim") - g("P_base(-tex,-dem,-clim)"),
        "inc_tex_dem": g("TD_base+tex+dem") - g("T_base+tex"),
        "inc_tex_clim": g("TC_base+tex+clim") - g("T_base+tex"),
        "total_all": g("TDC_all"),
    }
    json.dump(rep, open(OP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[ok] wrote pathc_climate_static.json")


if __name__ == "__main__":
    main()
