# -*- coding: utf-8 -*-
"""C-7c：增量在 Δ_floor 分布上的**分层定位**——DEM 到底在哪一段起作用？

动机（来自 C-7b）
  原始口径：DEM 增量 R² +0.034，RMSE −0.113（P=1.000）但 MedAE −0.004（P=0.634）
  缩尾口径：R² −0.001（CI 跨 0），MAE 反而 +0.042（P=0.001）
  ⇒ 假设：**DEM 的收益集中在 Δ_floor 极端尾段，对分布主体不降反升**。
  这里不用再拟合，直接对留一战役 OOF 预测做分层（按真值 Δ_floor 五等分）。

输出
  experiments/pathc_increment_strata.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M          # noqa: E402
import pathc_dem_analysis as DA   # noqa: E402
import pathc_climate as C         # noqa: E402
import pathc_worldcover_analysis as W  # noqa: E402

NQ = 5
B = 2000
SEED = 7


def oof(rows, keys, y, fit):
    X = DA.matrix(rows, keys)
    good = np.isfinite(X).all(axis=1)
    sub = [r for r, g in zip(rows, good) if g]
    _, _, p = M.cv_group(sub, X[good], y[good], "campaign", fit, ret_pred=True)
    full = np.full(len(y), np.nan)
    full[np.array(good)] = p
    return full


def main():
    rows = M.load()
    C.merge(rows)
    W.merge(rows)
    y0 = np.array([r["delta_floor"] for r in rows], float)
    ok = np.isfinite(y0)
    rows = [r for r, o in zip(rows, ok) if o]
    y = y0[ok]
    dem_ok = np.isfinite(DA.matrix(rows, C.DEM_KEYS)).all(axis=1)
    clim_ok = np.isfinite(DA.matrix(rows, C.CLIM_KEYS)).all(axis=1)
    good = dem_ok & clim_ok
    rows = [r for r, g in zip(rows, good) if g]
    y = y[good]
    print(f"[load] n={len(rows)}  Δ_floor mean={y.mean():.2f} std={y.std():.2f}")

    fit = M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)
    pred = {"+tex": oof(rows, C.BASE_TEX, y, fit),
            "+tex+dem": oof(rows, C.BASE_TEX + C.DEM_KEYS, y, fit),
            "+tex+wc": oof(rows, C.BASE_TEX + W.WC_KEYS, y, fit)}
    r2 = {k: float(M.r2_of(y[np.isfinite(v)], v[np.isfinite(v)])) for k, v in pred.items()}
    print("  " + "  ".join(f"{k}: R²={v:.3f}" for k, v in r2.items()))

    q = np.quantile(y, np.linspace(0, 1, NQ + 1))
    q[0], q[-1] = -np.inf, np.inf
    rng = np.random.default_rng(SEED)
    print(f"\n[分层] Δ_floor 五等分（Q1=最负=信息预算枯竭，Q5=最接近 0）")
    print(f"{'层':<4s} {'Δ_floor 区间':>18s} {'n':>5s} {'ΔRMSE(DEM)':>12s} "
          f"{'ΔMAE(DEM)':>11s} {'P(改善)':>9s}")
    out = {"r2": r2, "strata": []}
    for j in range(NQ):
        m = (y > q[j]) & (y <= q[j + 1]) & np.isfinite(pred["+tex"]) \
            & np.isfinite(pred["+tex+dem"])
        n = int(m.sum())
        yy = y[m]
        ea = np.abs(yy - pred["+tex"][m])
        eb = np.abs(yy - pred["+tex+dem"][m])
        dr = float(np.sqrt((eb ** 2).mean()) - np.sqrt((ea ** 2).mean()))
        dm = float(eb.mean() - ea.mean())
        idx = rng.integers(0, n, size=(B, n))
        db = eb[idx].mean(axis=1) - ea[idx].mean(axis=1)
        lo, hi = np.percentile(db, [2.5, 97.5])
        pimp = float((db < 0).mean())
        lo_s = "-inf" if j == 0 else f"{q[j]:.1f}"
        hi_s = "+inf" if j == NQ - 1 else f"{q[j+1]:.1f}"
        print(f"Q{j+1:<3d} [{lo_s:>7s},{hi_s:>7s}] {n:5d} {dr:+12.3f} {dm:+11.3f} "
              f"{pimp:9.3f}   MAE 95%CI [{lo:+.3f},{hi:+.3f}]")
        out["strata"].append({"q": f"Q{j+1}", "range": [lo_s, hi_s], "n": n,
                              "delta_rmse": dr, "delta_mae": dm,
                              "mae_ci": [float(lo), float(hi)], "p_improve": pimp})
    json.dump(out, open(os.path.join(M.EXP, "pathc_increment_strata.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[ok] wrote pathc_increment_strata.json")


if __name__ == "__main__":
    main()
