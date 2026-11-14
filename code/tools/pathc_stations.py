# -*- coding: utf-8 -*-
"""C-2a：站点细化 —— 留国家 / 留区域 CV。

问题
  WorldStrat 只有 4 个采集战役（campaign），留一战役只剩 3 个训练组，
  "留站点 CV" 的自由度太低（4 折），结论不稳。

观察
  UNHCR 的 scene id 形如 `UNHCR-<ISO3><序列号>`，内含**国家码**：实测 87 国。
  Landcover / ASMSpotter / Amnesty POI 的 id 无国家信息，但可由经纬度划到
  地理网格，构成等价的"区域站点"。

设计
  station 三档：
    S4   = campaign（4 组，基线）
    S87  = UNHCR 按 ISO3 国家 + 其余按 10° 经纬格（≈ 100+ 组）
    S_geo= 全部按 10° 经纬格（纯地理，≈ 100+ 组）
  对每档做 leave-one-station-out CV（GBDT，leak-free 特征）。

  判据：细化站点后 R² 若仍 ≥ 0.4，说明"跨采集体制泛化"结论不依赖
  某几个大战役，且不是靠站点内部的空间自相关撑起来的。

输出
  experiments/pathc_stations.json
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M  # noqa: E402

CAMP = re.compile(r"^(ASMSpotter|Amnesty POI|Landcover|UNHCR)")
DEM_KEYS = ["dem_std", "slope_mean", "tri_mean", "relief", "roughness", "dem_mean"]


def load_dem():
    p = os.path.join(M.EXP, "dem_terrain.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" not in r:
            out[r["cov"]] = r
    return out


def append_dem(rows, X, keys):
    """把 DEM 地形量接到特征矩阵右侧（C-1 的独立地表量）。"""
    dem = load_dem()
    add = np.full((len(rows), len(DEM_KEYS)), np.nan)
    for i, r in enumerate(rows):
        d = dem.get(r["cov"])
        if not d:
            continue
        for j, k in enumerate(DEM_KEYS):
            v = d.get(k)
            if v is not None:
                add[i, j] = float(v)
    return np.column_stack([X, add]), list(keys) + DEM_KEYS


def station_of(cov, kind):
    cov = str(cov).strip()
    m = CAMP.match(cov)
    camp = m.group(1) if m else "OTHER"
    if kind == "campaign":
        return camp
    if kind == "country":
        if camp == "UNHCR":
            y = cov.split("-", 1)[1] if "-" in cov else cov
            m2 = re.match(r"([A-Z]{3})", y)
            # 极少数 id 无国家码（实测 2/981，纯数字），退回 campaign
            return "UNHCR-" + m2.group(1) if m2 else camp
        return camp          # 无国家码者退回 campaign
    raise ValueError(kind)


def geo_station(lat, lon, d=10.0):
    return f"G{int((lat + 90) // d)}_{int((lon + 180) // d)}"


def cv_stations(rows, X, y, stations, fitter, min_tr=80, min_n=10):
    """leave-one-station-out。返回总体 R²、逐站 R²、被留出的站点数。"""
    pred = np.full(len(y), np.nan)
    per = {}
    for g in sorted(set(stations)):
        te = np.array([i for i, s in enumerate(stations) if s == g])
        if len(te) < min_n:
            continue                      # 太小的站点不单独留出（统计不可靠）
        tr = np.array([i for i, s in enumerate(stations) if s != g])
        if len(tr) < min_tr:
            continue
        pred[te] = fitter(X[tr], y[tr], X[te])
        per[g] = float(M.r2_of(y[te], pred[te])) if len(te) > 3 else np.nan
    m = np.isfinite(pred)
    if m.sum() < 50:
        return float("nan"), int(m.sum()), per, 0
    return (float(M.r2_of(y[m], pred[m])), int(m.sum()), per,
            len([v for v in per.values() if np.isfinite(v)]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="all",
                    choices=["all", "S4", "S87", "S_geo", "S_mixed"])
    ap.add_argument("--model", default="gbdt", choices=["gbdt", "ridge", "knn"])
    ap.add_argument("--dem", action="store_true", help="并入独立 DEM 地形量（C-1）")
    a = ap.parse_args()

    rows = M.load()
    y = np.array([r["delta_floor"] for r in rows], float)
    X, yf, keep, keys = M.build(rows, y, 1)
    sub = [rows[i] for i in keep]
    if a.dem:
        X, keys = append_dem(sub, X, keys)
        ok = np.isfinite(X).all(axis=1)
        X, yf, sub = X[ok], yf[ok], [r for r, o in zip(sub, ok) if o]
    print(f"[load] n={len(sub)}  p={X.shape[1]}  dem={a.dem}", flush=True)
    fit = {"gbdt": M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1),
           "ridge": M.make_ridge(10.0), "knn": M.naive_knn(15)}[a.model]

    OUTP = os.path.join(M.EXP, "pathc_stations.json")
    try:
        out = json.load(open(OUTP, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        out = {}
    sfx = "_dem" if a.dem else ""
    out["_meta" + sfx] = {"n": len(sub), "n_feat": int(X.shape[1]),
                          "keys": keys, "model": a.model, "dem": bool(a.dem)}

    def dump():
        json.dump(out, open(OUTP, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    want = lambda k: a.kind in ("all", k)  # noqa: E731

    # ---- S4：campaign（基线，复用 cv_group）----
    if want("S4"):
        lc, per4 = M.cv_group(sub, X, yf, "campaign", fit)
        print(f"\n[S4  campaign=4]  R2={lc:.3f}   " +
              " ".join(f"{k}={v:.2f}" for k, v in per4.items()), flush=True)
        out["S4_campaign" + sfx] = {"n_stations": 4, "cv_r2": lc, "per_station": per4}
        dump()

    # ---- S87：UNHCR 用国家，其余用 campaign ----
    if want("S87"):
        st_c = [station_of(r["cov"], "country") for r in sub]
        n_st = len(set(st_c))
        r2c, ncov, perc, nused = cv_stations(sub, X, yf, st_c, fit)
        cnt = {}
        for s in st_c:
            cnt[s] = cnt.get(s, 0) + 1
        used = [k for k in perc if np.isfinite(perc[k])]
        print(f"\n[S87 country/region]  stations={n_st} (used {nused}, n>=10)  "
              f"R2={r2c:.3f}  n_covered={ncov}", flush=True)
        print("   最大 10 站: " + ", ".join(f"{k}({cnt[k]})" for k, _ in
                                            sorted(cnt.items(), key=lambda x: -x[1])[:10]),
              flush=True)
        if used:
            v = np.array([perc[k] for k in used], float)
            print(f"   逐站 R2: median={np.median(v):.3f}  "
                  f"IQR=[{np.percentile(v, 25):.3f},{np.percentile(v, 75):.3f}]  "
                  f"frac>0={(v > 0).mean():.2f}", flush=True)
        out["S87_country" + sfx] = {"n_stations": n_st, "n_used": nused, "cv_r2": r2c,
                              "per_station": perc,
                              "station_sizes": dict(sorted(cnt.items(),
                                                           key=lambda x: -x[1]))}
        dump()

    # ---- S_geo：全部按 10° 经纬格 ----
    if want("S_geo"):
        st_g = [geo_station(r["lat"], r["lon"], 10.0) for r in sub]
        n_g = len(set(st_g))
        r2g, ncovg, perg, nusedg = cv_stations(sub, X, yf, st_g, fit)
        print(f"\n[S_geo 10deg-grid]  stations={n_g} (used {nusedg})  R2={r2g:.3f}",
              flush=True)
        out["S_geo" + sfx] = {"n_stations": n_g, "n_used": nusedg, "cv_r2": r2g,
                        "per_station": perg}
        dump()

    # ---- S_mixed：UNHCR 国家 + 其余 10° 格（最细粒度）----
    if want("S_mixed"):
        st_m = []
        for r in sub:
            m = CAMP.match(r["cov"])
            camp = m.group(1) if m else "OTHER"
            st_m.append(station_of(r["cov"], "country") if camp == "UNHCR"
                        else geo_station(r["lat"], r["lon"], 10.0))
        n_m = len(set(st_m))
        r2m, ncovm, perm, nusedm = cv_stations(sub, X, yf, st_m, fit)
        print(f"\n[S_mixed]  stations={n_m} (used {nusedm})  R2={r2m:.3f}", flush=True)
        out["S_mixed" + sfx] = {"n_stations": n_m, "n_used": nusedm, "cv_r2": r2m,
                          "per_station": perm}
        dump()

    print("\n[ok] wrote pathc_stations.json")


if __name__ == "__main__":
    main()
