# -*- coding: utf-8 -*-
"""C-1：用独立于 HR 影像的地形量替换 / 补充 texture_hr。

待回答的审稿人问题
  "texture_hr 与 Δ_floor 都出自同一张 HR 影像（Δ_floor 的基线是 std(HR)，
   texture_hr 是 |∇HR|），这是不是定义层耦合？换一个**独立来源**的地表量，
   机制还在不在？"

做法
  1. 相关性：DEM 地形量 vs texture_hr（若高度相关 ⇒ 纹理≈地形）；
     DEM 地形量 vs Δ_floor（独立于影像的机制证据）。
  2. 特征集对照（同一 GBDT，同一 CV）：
       A = leak-free 原版（含 texture_hr）
       B = 去掉 texture_hr，改用 DEM 地形量
       C = A ∪ B
     判据：B 能到 A 的多少？C 相对 A 的 ΔR² 是否显著为正？
  3. 若 B ≈ A 且 C > A，则"地形结构设定可恢复预算"的结论
     **有两个独立来源的印证**，定义层耦合的质疑被消解。

输出
  experiments/pathc_dem_analysis.json
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M  # noqa: E402

DEMF = os.path.join(M.EXP, "dem_terrain.jsonl")
DEM_KEYS = ["dem_std", "slope_mean", "tri_mean", "relief", "roughness", "dem_mean"]


def load_dem():
    out = {}
    if not os.path.exists(DEMF):
        return out
    for line in open(DEMF, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        out[r["cov"]] = r
    return out


# 原 leak-free 特征（与 pathc_models.engineer level1 对应）
BASE_KEYS = ["ndvi_mean", "ndvi_std", "bright_lr", "spec_het", "bright_hr",
             "log_tex", "abs_lat", "cloud", "abs_dt_log", "lon",
             "log_area", "log_lccs"]


# 派生量 -> 源字段（原始 JSON 里只有源字段，派生量在此处计算）
DERIVE = {
    "log_tex":    ("texture_hr", lambda v: np.log10(max(v, 1e-12))),
    "abs_dt_log": ("delta_days", lambda v: np.log10(abs(v) + 1.0)),
    "abs_lat":    ("lat",        lambda v: abs(v)),
    "log_area":   ("area_km2",   lambda v: np.log10(max(v, 1e-3))),
    "log_lccs":   ("lccs",       lambda v: np.log10(max(v, 1.0))),
}


def matrix(rows, keys):
    X = np.full((len(rows), len(keys)), np.nan)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            src, fn = DERIVE.get(k, (k, None))
            v = r.get(src)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            X[i, j] = fn(v) if fn else v
    return X


def r2_of(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    rows = M.load()
    dem = load_dem()
    print(f"[load] rows={len(rows)}  dem={len(dem)}")
    if not dem:
        print("[abort] 无 DEM 结果")
        return

    # 合并 DEM 量
    n_hit = 0
    for r in rows:
        d = dem.get(r["cov"])
        if d:
            n_hit += 1
            for k in DEM_KEYS:
                r[k] = d.get(k)
            r["dem_mpp"] = d.get("mpp")
    print(f"[merge] DEM 命中 {n_hit}/{len(rows)}")

    y = np.array([r.get("delta_floor", np.nan) for r in rows], float)
    # 保持顺序，过滤掉 y 无效的
    keep_rows, keep_y = [], []
    for r, v in zip(rows, y):
        if np.isfinite(v):
            keep_rows.append(r)
            keep_y.append(v)
    rows, y = keep_rows, np.array(keep_y, float)

    report = {"n_rows": len(rows), "n_dem": n_hit}

    # ---------------- 1. 相关性 ----------------
    print("\n[1] 相关性（DEM 量 vs texture_hr / Δ_floor）")
    print(f"{'变量':<14s} {'vs texture_hr':>16s} {'vs Δ_floor':>16s}  n")
    corr = {}
    tex = matrix(rows, ["log_tex"])[:, 0]
    for k in DEM_KEYS:
        v = np.array([r.get(k, np.nan) for r in rows], float)
        m = np.isfinite(v) & np.isfinite(tex) & np.isfinite(y)
        if m.sum() < 50:
            continue
        a = stats.pearsonr(v[m], tex[m])
        b = stats.pearsonr(v[m], y[m])
        corr[k] = {"vs_tex_r": float(a[0]), "vs_tex_p": float(a[1]),
                   "vs_df_r": float(b[0]), "vs_df_p": float(b[1]), "n": int(m.sum())}
        print(f"{k:<14s} r={a[0]:+.3f}(p={a[1]:.1e}) r={b[0]:+.3f}(p={b[1]:.3e}) {m.sum()}")
    report["correlations"] = corr

    # texture_hr 自身 vs Δ_floor（对照）
    m0 = np.isfinite(tex) & np.isfinite(y)
    report["tex_vs_df"] = {"r": float(stats.pearsonr(tex[m0], y[m0])[0]),
                           "p": float(stats.pearsonr(tex[m0], y[m0])[1]),
                           "n": int(m0.sum())}
    print(f"{'log_tex(自带)':<14s} r={'-':>16s} "
          f"r={report['tex_vs_df']['r']:+.3f}(p={report['tex_vs_df']['p']:.1e}) {m0.sum()}")

    # ---------------- 2. 特征集对照 ----------------
    # ⚠ 所有集合必须在**同一子集**上比较（有 DEM 的那些景），否则 n 不同不可比。
    BASE_NO_TEX = [k for k in BASE_KEYS if k != "log_tex"]
    sets = {
        "A_leakfree(+tex)": BASE_KEYS,
        "B_dem_only(-tex)": BASE_NO_TEX + DEM_KEYS,
        "C_both": BASE_KEYS + DEM_KEYS,
        "D_base_noderm_notext": BASE_NO_TEX,
    }
    fit = M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)

    # 公共子集：DEM 全覆盖 且 基础特征齐全
    dem_ok = np.array([all(np.isfinite(float(r[k])) for k in DEM_KEYS
                           if r.get(k) is not None) and
                       all(r.get(k) is not None for k in DEM_KEYS) for r in rows], bool)
    print(f"\n[2] 特征集对照（GBDT 60树，leak-free，**公共子集 n={dem_ok.sum()}**）")
    print(f"{'集合':<24s} {'p':>3s} {'n':>5s} {'in':>6s} "
          f"{'CV30x60':>8s} {'leaveCamp':>10s}")
    res = {}
    for name, keys in sets.items():
        X = matrix(rows, keys)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y) & dem_ok
        Xo, yo = X[ok], y[ok]
        sub = [r for r, o in zip(rows, ok) if o]
        ins = r2_of(yo, fit(Xo, yo, Xo))
        c30 = M.cv_spatial(sub, Xo, yo, 30, 60, fit)[0]
        lc, per = M.cv_group(sub, Xo, yo, "campaign", fit)
        print(f"{name:<24s} {len(keys):3d} {len(yo):5d} {ins:6.3f} "
              f"{c30:8.3f} {lc:10.3f}", flush=True)
        res[name] = {"p": len(keys), "n": int(len(yo)), "in_r2": ins,
                     "cv_30x60": c30,
                     "cv_leave_campaign": lc, "per_campaign": per, "keys": keys}
        report["featsets"] = res
        json.dump(report, open(os.path.join(M.EXP, "pathc_dem_analysis.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------------- 3. 判据 ----------------
    if "A_leakfree(+tex)" in res and "B_dem_only(-tex)" in res:
        a, b, c, d = (res["A_leakfree(+tex)"], res["B_dem_only(-tex)"],
                      res["C_both"], res["D_base_noderm_notext"])
        print("\n[3] 判据（公共子集）")
        print(f"  A (含 texture_hr)   留站点 = {a['cv_leave_campaign']:.3f}")
        print(f"  B (DEM 替代 tex)    留站点 = {b['cv_leave_campaign']:.3f}  "
              f"({b['cv_leave_campaign'] / max(a['cv_leave_campaign'], 1e-9):.0%} of A)")
        print(f"  C (两者并用)        留站点 = {c['cv_leave_campaign']:.3f}  "
              f"(Δ vs A = {c['cv_leave_campaign'] - a['cv_leave_campaign']:+.3f})")
        print(f"  D (都去掉)          留站点 = {d['cv_leave_campaign']:.3f}")
        print(f"  ⇒ texture_hr 的增量 (A−D) = "
              f"{a['cv_leave_campaign'] - d['cv_leave_campaign']:+.3f}")
        print(f"  ⇒ DEM 量的增量   (B−D) = "
              f"{b['cv_leave_campaign'] - d['cv_leave_campaign']:+.3f}")
        print(f"  ⇒ 两者互补      (C−A) = "
              f"{c['cv_leave_campaign'] - a['cv_leave_campaign']:+.3f}   "
              f"[CV30x60: {c['cv_30x60'] - a['cv_30x60']:+.3f}]")
        report["verdict"] = {
            "subset_n": int(a["n"]),
            "A_leavecamp": a["cv_leave_campaign"],
            "B_leavecamp": b["cv_leave_campaign"],
            "C_leavecamp": c["cv_leave_campaign"],
            "D_leavecamp": d["cv_leave_campaign"],
            "B_over_A": b["cv_leave_campaign"] / max(a["cv_leave_campaign"], 1e-9),
            "tex_increment_A_minus_D": a["cv_leave_campaign"] - d["cv_leave_campaign"],
            "dem_increment_B_minus_D": b["cv_leave_campaign"] - d["cv_leave_campaign"],
            "complementary_C_minus_A": c["cv_leave_campaign"] - a["cv_leave_campaign"],
            "complementary_C_minus_A_30x60": c["cv_30x60"] - a["cv_30x60"],
        }
        json.dump(report, open(os.path.join(M.EXP, "pathc_dem_analysis.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n[ok] wrote pathc_dem_analysis.json")


if __name__ == "__main__":
    main()
