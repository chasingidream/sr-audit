# -*- coding: utf-8 -*-
"""C-7b：三类静态量增量的**置信区间**（配对 bootstrap）+ 缩尾稳健口径。

背景（必须解决的自相矛盾）
  同一份代码、同一 seed、同一特征列顺序，只换子集：
    全量 n=3927（mask=dem∩clim）   +tex 0.474 → +tex+dem 0.508   ⇒ DEM 增量 +0.034
    n=3831（mask=wc∩dem∩clim）     +tex 0.508 → +tex+dem 0.495   ⇒ DEM 增量 −0.013
  诊断：缺失的 96 景是 Δ_floor 的**极端尾部**（均值 −8.56 vs −3.56，std 10.40 vs
  4.47，Δ_floor<−20 dB 占 19.8% vs 1.3%，|lat| 中位 36.8° vs 26.9°）。
  ⇒ pooled R² 由 2.4% 的极端景主导，点估计不可靠。

做法
  1. 留一战役 CV 取出**逐样本 OOF 预测**（不重采样也要重拟合，故只重采样误差项）。
  2. 对 (i, i, ...) 配对 bootstrap B 次，ΔR² = R²_B − R²_A，取 2.5/97.5 分位。
     ⇔ 两模型在**同一批重采样样本**上比，抵消样本构成的影响。
  3. 缩尾口径：y 在 1%/99% 分位截断后重拟合重评估（保留全部样本，只削尾部杠杆）。

输出
  experiments/pathc_increment_ci.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M          # noqa: E402
import pathc_dem_analysis as DA   # noqa: E402
import pathc_climate as C         # noqa: E402
import pathc_worldcover_analysis as W  # noqa: E402

SETS = {
    "base": C.BASE_NO_TEX,
    "+tex": C.BASE_TEX,
    "+tex+dem": C.BASE_TEX + C.DEM_KEYS,
    "+tex+wc": C.BASE_TEX + W.WC_KEYS,
    "+tex+clim": C.BASE_TEX + C.CLIM_KEYS,
}
# (对照, 处理, 标签)
CONTRASTS = [("base", "+tex", "texture"),
             ("+tex", "+tex+dem", "dem"),
             ("+tex", "+tex+wc", "worldcover"),
             ("+tex", "+tex+clim", "climate")]


def err_boot(y, pa, pb, idx):
    """RMSE / MAE / MedAE（对尾部不敏感，用来判定增量是否只来自极端景）。"""
    yy = y[idx]
    ea, eb = np.abs(yy - pa[idx]), np.abs(yy - pb[idx])
    return (np.sqrt((ea ** 2).mean()), np.sqrt((eb ** 2).mean()),
            ea.mean(), eb.mean(), np.median(ea), np.median(eb))


def r2_boot(y, pa, pb, idx):
    """在同一批重采样样本上算两个模型的 R²（SST 共用）。"""
    yy, aa, bb = y[idx], pa[idx], pb[idx]
    sst = ((yy - yy.mean()) ** 2).sum()
    if sst <= 0:
        return np.nan, np.nan
    return (1 - ((yy - aa) ** 2).sum() / sst,
            1 - ((yy - bb) ** 2).sum() / sst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", default="dem", choices=["dem", "wc"])
    ap.add_argument("--winsor", default="")     # 给 "1,99" 则缩尾
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    rows = M.load()
    C.merge(rows)
    W.merge(rows)
    y0 = np.array([r["delta_floor"] for r in rows], float)
    ok = np.isfinite(y0)
    rows = [r for r, o in zip(rows, ok) if o]
    y0 = y0[ok]

    dem_ok = np.isfinite(DA.matrix(rows, C.DEM_KEYS)).all(axis=1)
    clim_ok = np.isfinite(DA.matrix(rows, C.CLIM_KEYS)).all(axis=1)
    wc_ok = np.isfinite(DA.matrix(rows, W.WC_KEYS)).all(axis=1)
    common = (dem_ok & clim_ok & wc_ok) if a.mask == "wc" else (dem_ok & clim_ok)

    rows = [r for r, c in zip(rows, common) if c]
    y = y0[common]
    wl = wh = None
    if a.winsor:
        wl, wh = [float(v) for v in a.winsor.split(",")]
        lo, hi = np.percentile(y, wl), np.percentile(y, wh)
        y = np.clip(y, lo, hi)
    print(f"[load] mask={a.mask}  n={len(rows)}  winsor={a.winsor or 'none'} "
          f"y: mean={y.mean():.2f} std={y.std():.2f}", flush=True)

    fit = M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)
    pred, r2s = {}, {}
    for name, keys in SETS.items():
        X = DA.matrix(rows, keys)
        good = np.isfinite(X).all(axis=1)
        Xo, yo = X[good], y[good]
        sub = [r for r, g in zip(rows, good) if g]
        r2, per, p = M.cv_group(sub, Xo, yo, "campaign", fit, ret_pred=True)
        full = np.full(len(y), np.nan)
        full[np.array(good)] = p
        pred[name] = full
        r2s[name] = float(r2)
        print(f"  {name:<10s} p={len(keys):2d} n={int(good.sum()):4d} "
              f"leaveCamp R²={r2:.3f}  per={ {k: round(v,3) for k,v in per.items()} }",
              flush=True)

    rng = np.random.default_rng(a.seed)

    print(f"\n[bootstrap] B={a.B}（每个对比用自己的公共交集，点估计与 CI 同集合）")
    print(f"{'增量':<12s} {'n':>5s} {'点估计':>8s} {'95% CI':>18s} {'P(Δ>0)':>8s}")
    out = {}
    for base, treat, tagname in CONTRASTS:
        # ⚠ 每个对比单独取交集：否则 wc 的 96 景缺失会把 dem 对比也拖到 n=3831
        gm = np.isfinite(pred[base]) & np.isfinite(pred[treat])
        yy = y[gm]
        pa, pb = pred[base][gm], pred[treat][gm]
        n = int(gm.sum())
        idxs = rng.integers(0, n, size=(a.B, n))
        d = np.empty(a.B)
        for b in range(a.B):
            ra, rb = r2_boot(yy, pa, pb, idxs[b])
            d[b] = rb - ra
        pt = float(M.r2_of(yy, pb) - M.r2_of(yy, pa))
        lo, hi = np.percentile(d, [2.5, 97.5])
        ppos = float((d > 0).mean())
        print(f"{tagname:<12s} {n:5d} {pt:+8.3f}   [{lo:+.3f}, {hi:+.3f}]   {ppos:8.3f}")
        # ---- 误差口径（Δ<0 = 更优）----
        dm = np.empty((a.B, 3))
        for b in range(a.B):
            ra, rb, ma, mb, qa, qb = err_boot(yy, pa, pb, idxs[b])
            dm[b] = (rb - ra, mb - ma, qb - qa)
        pt_m = [float(np.sqrt(((yy - pb) ** 2).mean()) - np.sqrt(((yy - pa) ** 2).mean())),
                float(np.abs(yy - pb).mean() - np.abs(yy - pa).mean()),
                float(np.median(np.abs(yy - pb)) - np.median(np.abs(yy - pa)))]
        ci_m = np.percentile(dm, [2.5, 97.5], axis=0)
        for j, mn in enumerate(("RMSE", "MAE", "MedAE")):
            print(f"    {mn:<6s} Δ={pt_m[j]:+7.4f}  "
                  f"[{ci_m[0, j]:+.4f}, {ci_m[1, j]:+.4f}]  "
                  f"P(改善)={float((dm[:, j] < 0).mean()):.3f}")
        out[tagname] = {"point": float(pt), "ci_lo": float(lo), "ci_hi": float(hi),
                        "p_positive": ppos, "n": n,
                        "base_r2": float(M.r2_of(yy, pa)),
                        "treat_r2": float(M.r2_of(yy, pb)),
                        "delta_rmse": {"point": pt_m[0], "ci": [float(ci_m[0, 0]), float(ci_m[1, 0])],
                                       "p_improve": float((dm[:, 0] < 0).mean())},
                        "delta_mae": {"point": pt_m[1], "ci": [float(ci_m[0, 1]), float(ci_m[1, 1])],
                                      "p_improve": float((dm[:, 1] < 0).mean())},
                        "delta_medae": {"point": pt_m[2], "ci": [float(ci_m[0, 2]), float(ci_m[1, 2])],
                                        "p_improve": float((dm[:, 2] < 0).mean())}}
        json.dump({"args": vars(a), "r2": r2s, "increments": out},
                  open(os.path.join(M.EXP, "pathc_increment_ci%s.json" % a.tag),
                       "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[ok] wrote pathc_increment_ci%s.json" % a.tag)


if __name__ == "__main__":
    main()
