# -*- coding: utf-8 -*-
"""C-6：格网级 CV 的正确实现 —— K-fold by cell（GBDT）。

背景
  C-4 用"留一格网 CV"得到 0.427~0.430，但那是用 **ridge** 算的
  （2,273 个格网逐个重训 GBDT 超出沙箱 CPU 上限）。ridge 在同等留出口径下
  系统性偏低（CV 30°×60°：GBDT 0.684 vs ridge 0.354，差 0.33），
  故 0.43 只是**保守下界**，不是产品精度。

修正
  "留一格网"并非必需。防泄漏的实质是"**同一格网的样本不同时出现在训练与
  留出集中**"，这只需把格网划分为互斥的 K 组即可满足 —— 即 **K-fold by cell**。
  格网数 2,273 时只需 K 次重训（K=10 → 15 s），可轻松用 GBDT。

  ⚠ 与"留一格网"的差别仅在方差（K-fold 的留出集是 1/K 的格网，
    训练集更大 ⇒ 略偏乐观），但**泄漏被完全切断**，这才是关键。

对标口径（同一 18 维特征集、同一 GBDT60）
  CV 30°×60° 空间块 CV      0.684
  buffered 30° CV            0.631
  留战役 CV                  0.508
  **K-fold by cell (K=10)**  ← 本脚本给出产品级估计

输出
  experiments/pathc_cellcv.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M  # noqa: E402
import pathc_dem_analysis as DA  # noqa: E402
import pathc_climate as C  # noqa: E402
import pathc_atlas as A  # noqa: E402

K = 10
SEED = 7


def main():
    rows, X, y = A.build()
    print(f"[build] n={len(rows)} p={X.shape[1]}", flush=True)
    fit = M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)

    rep = {"n_scenes": int(len(rows)), "n_feat": int(X.shape[1]),
           "K": K, "seed": SEED, "model": "GBDT60", "keys": A.KEYS, "grids": {}}

    for res in (5.0, 1.0, 0.5):
        keys = np.array([A.cell_id(r["lat"], r["lon"], res) for r in rows],
                        dtype=np.int64)
        uniq = sorted(set(map(tuple, keys.tolist())))
        kid = {k: i for i, k in enumerate(uniq)}
        cell = np.array([kid[k] for k in map(tuple, keys.tolist())])
        # 把格网分配到 K 折（按格网随机，**格网整体**不可拆分）
        rng = np.random.RandomState(SEED)
        fold_of_cell = rng.permutation(len(uniq)) % K
        fold = fold_of_cell[cell]

        pred = np.full(len(y), np.nan)
        for k in range(K):
            te = np.where(fold == k)[0]
            tr = np.where(fold != k)[0]
            pred[te] = fit(X[tr], y[tr], X[te])
        m = np.isfinite(pred)
        r2 = float(M.r2_of(y[m], pred[m]))
        rmse = float(np.sqrt(((y[m] - pred[m]) ** 2).mean()))

        # 格网均值层面的 R²（图集产品实际以格网均值发布）
        gm, gp = [], []
        for gi in range(len(uniq)):
            s = cell == gi
            gm.append(y[s].mean())
            gp.append(pred[s].mean())
        gm, gp = np.array(gm), np.array(gp)
        r2_cellmean = float(M.r2_of(gm, gp))

        # 基准：同折但不做格网分组（普通 K-fold，衡量"格网分组"本身的代价）
        pred_naive = np.full(len(y), np.nan)
        rng2 = np.random.RandomState(SEED)
        f2 = rng2.permutation(len(y)) % K
        for k in range(K):
            te = np.where(f2 == k)[0]
            tr = np.where(f2 != k)[0]
            pred_naive[te] = fit(X[tr], y[tr], X[te])
        assert np.isfinite(pred_naive).all(), "普通 K-fold 应覆盖全部样本"
        r2_naive = float(M.r2_of(y, pred_naive))

        print(f"\n[{res}°] 格网数={len(uniq)}  K={K}", flush=True)
        print(f"  **K-fold by cell  R² = {r2:.3f}**  (RMSE={rmse:.3f} dB, "
              f"n={int(m.sum())})", flush=True)
        print(f"      格网均值层面 R² = {r2_cellmean:.3f}  ({len(uniq)} 个格网)")
        print(f"      基准：普通 K-fold（不按格网分组）R² = {r2_naive:.3f}", flush=True)
        print(f"      ⇒ 格网分组的代价 = {r2 - r2_naive:+.3f}", flush=True)
        rep["grids"][f"{res}"] = {
            "n_cells": int(len(uniq)), "K": K,
            "kfold_by_cell_r2": r2, "rmse_db": rmse, "n_pred": int(m.sum()),
            "cellmean_r2": r2_cellmean,
            "naive_kfold_r2": r2_naive,
            "grouping_cost": r2 - r2_naive,
        }
        json.dump(rep, open(os.path.join(M.EXP, "pathc_cellcv.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n[ok] wrote pathc_cellcv.json")


if __name__ == "__main__":
    main()
