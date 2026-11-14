# -*- coding: utf-8 -*-
"""C-7 分析：WorldCover 土地覆盖组成作为第二类独立静态量。

问题
  C-1 用 DEM 地形量（静态**几何**）提供了独立于 HR 影像的增量（留站点 +0.034）。
  C-5 用 WorldClim 气候量（静态**背景条件**）证明**无增量**（−0.010）。
  ⇒ 真正缺的是第三类：**地表组成**（土地覆盖类别比例）——
     它既不是影像强度纹理，也不是地形几何，而是"这个像元由什么构成"。

数据
  experiments/worldcover_2021.jsonl（ESA WorldCover 2021，10 m，1.6 km 窗口）
  派生：11 类占比 + 类别数 + Shannon 多样性 + 优势类占比 + 植被/水体/建成区汇总

判据
  1. 与 texture_hr、DEM 地形量的正交性
  2. 在 base / +tex / +tex+dem 之上是否有增量
  3. 与 C-1（几何）、C-5（背景条件）对比：三类静态量的边际价值排序

输出
  experiments/pathc_worldcover_static.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M  # noqa: E402
import pathc_dem_analysis as DA  # noqa: E402
import pathc_climate as C  # noqa: E402

WCF = os.path.join(M.EXP, "worldcover_2021.jsonl")
# 主分析只取信息量高、非共线的量（|r| 高的小类别会冗余）
WC_KEYS = ["wc_tree", "wc_grass", "wc_crop", "wc_built", "wc_bare",
           "wc_water_frac", "wc_nclass", "wc_entropy", "wc_dominant"]
DEM_KEYS = C.DEM_KEYS
BASE_NO_TEX = C.BASE_NO_TEX
BASE_TEX = C.BASE_TEX


def load_wc():
    out = {}
    if not os.path.exists(WCF):
        return out
    for line in open(WCF, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        out[r["cov"]] = r
    return out


def merge(rows):
    wc = load_wc()
    n = 0
    for r in rows:
        c = wc.get(r["cov"])
        if not c:
            continue
        n += 1
        for k in WC_KEYS:
            v = c.get(k)
            if v is not None:
                r[k] = float(v)
    return n, len(wc)


def main():
    ap = argparse.ArgumentParser()
    # --mask: 公共子集口径。wc=wc∩dem∩clim（C-7 主口径）；dem=dem∩clim（C-1/C-5 全量对照）
    ap.add_argument("--mask", default="wc", choices=["wc", "dem", "none"])
    ap.add_argument("--only", default="")          # 逗号分隔，只跑指定集合（分段续跑）
    ap.add_argument("--tag", default="")           # 输出文件名后缀，避免覆盖主结果
    a = ap.parse_args()

    rows = M.load()
    C.merge(rows)                    # WorldClim + DEM
    n_wc, tot_wc = merge(rows)       # WorldCover
    print(f"[load] rows={len(rows)}  worldcover={n_wc} (file has {tot_wc})", flush=True)
    y = np.array([r["delta_floor"] for r in rows], float)
    ok = np.isfinite(y)
    rows = [r for r, o in zip(rows, ok) if o]
    y = y[ok]

    OP = os.path.join(M.EXP, "pathc_worldcover_static%s.json" % a.tag)
    rep = {"n_rows": int(len(rows)), "n_worldcover": n_wc,
           "wc_keys": WC_KEYS, "n_feat_base": len(BASE_TEX),
           "mask": a.mask}
    if os.path.exists(OP):           # 续跑：先读回已有结果，避免重算
        try:
            old = json.load(open(OP, encoding="utf-8"))
            if old.get("mask") == a.mask:
                rep = old
        except Exception:
            pass

    # ---------- 1. 正交性 ----------
    tex = DA.matrix(rows, ["texture_hr"])[:, 0]
    dem = DA.matrix(rows, DEM_KEYS)
    print("\n[1] 正交性（Pearson r）")
    print(f"{'WorldCover 量':<16s} {'vs texture':>11s} {'vs DEM(max)':>12s} {'vs Δ_floor':>12s}")
    ortho = {}
    for k in WC_KEYS:
        v = DA.matrix(rows, [k])[:, 0]
        m = np.isfinite(v) & np.isfinite(dem).all(axis=1) & np.isfinite(tex) & np.isfinite(y)
        if m.sum() < 60:
            continue
        rt = stats.pearsonr(v[m], tex[m])[0]
        rd = np.array([stats.pearsonr(v[m], dem[m, q])[0] for q in range(dem.shape[1])])
        ry, py = stats.pearsonr(v[m], y[m])
        print(f"{k:<16s} {rt:+11.3f} {np.abs(rd).max():12.3f} "
              f"{ry:+12.3f} (p={py:.1e})")
        ortho[k] = {"r_vs_texture": float(rt),
                    "max_abs_r_vs_dem": float(np.abs(rd).max()),
                    "r_vs_dfloor": float(ry), "p_vs_dfloor": float(py),
                    "n": int(m.sum())}
    rep["orthogonality"] = ortho

    # ---------- 2. 特征集阶梯 ----------
    wc_ok = np.isfinite(DA.matrix(rows, WC_KEYS)).all(axis=1)
    dem_ok = np.isfinite(dem).all(axis=1)
    clim_ok = np.isfinite(DA.matrix(rows, C.CLIM_KEYS)).all(axis=1)
    if a.mask == "wc":
        common = wc_ok & dem_ok & clim_ok
    elif a.mask == "dem":
        common = dem_ok & clim_ok
    else:
        common = np.ones(len(rows), bool)
    print(f"\n[2] 特征集阶梯（mask={a.mask}，公共子集 n={common.sum()}，"
          f"留一战役 CV，GBDT 60 树）")
    sets = {
        "base(-tex,-dem,-clim)": BASE_NO_TEX,
        "+tex": BASE_TEX,
        "+tex+dem": BASE_TEX + DEM_KEYS,
        "+tex+wc": BASE_TEX + WC_KEYS,
        "+tex+dem+wc": BASE_TEX + DEM_KEYS + WC_KEYS,
        "+tex+dem+wc+clim": BASE_TEX + DEM_KEYS + WC_KEYS + C.CLIM_KEYS,
    }
    fit = M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)
    print(f"{'集合':<24s} {'p':>3s} {'in':>6s} {'CV30x60':>8s} {'leaveCamp':>10s}")
    res = rep.setdefault("featsets", {})
    wanted = [s.strip() for s in a.only.split(",") if s.strip()]
    for name, keys in sets.items():
        if wanted and name not in wanted:
            continue
        if name in res and res[name].get("n") == int(common.sum()):
            print(f"{name:<24s} {len(keys):3d}  [skip 已有]", flush=True)
            continue
        X = DA.matrix(rows, keys)
        okm = np.isfinite(X).all(axis=1) & common
        Xo, yo = X[okm], y[okm]
        if len(yo) < 60:
            print(f"{name:<24s} {len(keys):3d}  子集不足({len(yo)})", flush=True)
            continue
        sub = [r for r, o in zip(rows, okm) if o]
        ins = float(M.r2_of(yo, fit(Xo, yo, Xo)))
        c30 = M.cv_spatial(sub, Xo, yo, 30, 60, fit)[0]
        lc, per = M.cv_group(sub, Xo, yo, "campaign", fit)
        print(f"{name:<24s} {len(keys):3d} {ins:6.3f} {c30:8.3f} {lc:10.3f}", flush=True)
        res[name] = {"p": len(keys), "n": int(len(yo)), "in_r2": ins,
                     "cv_30x60": c30, "cv_leave_campaign": lc,
                     "per_campaign": per, "keys": keys}
        rep["featsets"] = res
        json.dump(rep, open(OP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------- 3. 三类静态量的边际价值排序 ----------
    if "+tex+dem" in res and "+tex+wc" in res and "+tex+dem+wc" in res:
        g = lambda k: res[k]["cv_leave_campaign"]   # noqa: E731
        b = g("+tex")
        print("\n[3] 在 texture 之上的边际增量（留一战役 R²）")
        print(f"  base(+tex)            {b:.3f}")
        print(f"  + 地形 DEM            {g('+tex+dem') - b:+.3f}  → {g('+tex+dem'):.3f}")
        print(f"  + 土地覆盖 WorldCover {g('+tex+wc') - b:+.3f}  → {g('+tex+wc'):.3f}")
        print(f"  + 两者并用             {g('+tex+dem+wc') - b:+.3f}  → "
              f"{g('+tex+dem+wc'):.3f}")
        print(f"  + 气候 WorldClim（C-5） −0.010")
        rep["verdict"] = {
            "base_tex": b,
            "inc_dem": g("+tex+dem") - b,
            "inc_wc": g("+tex+wc") - b,
            "inc_dem_wc": g("+tex+dem+wc") - b,
            "inc_clim_c5": -0.010,
            "subset_n": int(res["+tex+dem"]["n"]),
        }
        json.dump(rep, open(OP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[ok] wrote pathc_worldcover_static.json")


if __name__ == "__main__":
    main()
