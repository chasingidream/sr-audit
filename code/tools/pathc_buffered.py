# -*- coding: utf-8 -*-
"""C-3：buffered 空间留出 —— 彻底切断空间自相关。

问题
  现有 CV 是"硬留块"：留出块 k 的全部样本，但**紧邻块 k 的训练样本仍在训练集里**。
  由于 Δ_floor 具空间自相关（相邻景共享同一传感器、同一轨、同一地物），
  硬留块仍可能高估泛化能力。这就是审稿人"空间自相关造出假拟合"的质疑。

做法
  留出块 k 时，训练集额外排除**距任一留出样本 ≤ buffer 的样本**（缓冲带），
  强制模型跨过一段空间距离才能外推。扫描 buffer：

    buffer = 0°   ≡ 现有硬留块（基线）
    buffer = 0.5° / 1° / 2° / 5°   （≈ 55 / 110 / 220 / 550 km）

  场景最近邻距离中位数 0.402°（≈44.6 km），故 0.5° 已切掉最近的邻居，
  5° 是最严苛（跨气候带/国界）。

  ⚠ 必须同时报告**有效训练样本数 n_tr**：buffer 增大必然减少训练量，
    R² 的变化含"样本变少"与"空间泄漏被切断"两个竞争效应。

判据
  * R² 随 buffer **基本保持** → 空间自相关不是虚高的来源，图集可外推；
  * R² 随 buffer **快速下降** → 硬留块确实高估，精度声明须收紧。

输出
  experiments/pathc_buffered_cv.json
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M  # noqa: E402

BUFFERS = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
BLOCKS = [(10, 10), (20, 20), (30, 60)]
# GBDT 逐折重训：10x10=242 块 x 5 缓冲 = 1210 折会超时，
# 故树模型只跑 30x60（31 块 x 5 = 155 折 ≈ 4 min），全扫描用 ridge。


def cv_buffered(rows, X, y, dlat, dlon, buffer_deg, fitter, min_tr=60):
    """留块 + 缓冲带留出。返回 (R², n_pred, n_tr_mean, n_tr_min)。"""
    keys = np.array([(int((r["lat"] + 90) // dlat), int((r["lon"] + 180) // dlon))
                     for r in rows], dtype=np.int64)
    P = np.column_stack([np.array([r["lat"] for r in rows], float),
                         np.array([r["lon"] for r in rows], float)])
    tree = cKDTree(P)
    pred = np.full(len(y), np.nan)
    n_trs = []
    for k in sorted(set(map(tuple, keys.tolist()))):
        te = np.where((keys[:, 0] == k[0]) & (keys[:, 1] == k[1]))[0]
        if len(te) == 0:
            continue
        if buffer_deg > 0:
            # 到该块任一样本的最小距离；<= buffer 的样本一并排除出训练集。
            # 块内样本通常 <20 个，直接向量化算最小距离比 KD-tree 更快。
            dm = np.min(np.stack([np.hypot(P[:, 0] - P[i, 0],
                                           P[:, 1] - P[i, 1]) for i in te]), axis=0)
            tr = np.where((keys[:, 0] != k[0]) | (keys[:, 1] != k[1]))[0]
            tr = tr[dm[tr] > buffer_deg]
        else:
            tr = np.where((keys[:, 0] != k[0]) | (keys[:, 1] != k[1]))[0]
        if len(tr) < min_tr:
            continue
        n_trs.append(len(tr))
        pred[te] = fitter(X[tr], y[tr], X[te])
    m = np.isfinite(pred)
    if m.sum() < 50 or not n_trs:
        return float("nan"), int(m.sum()), 0, 0
    return (float(M.r2_of(y[m], pred[m])), int(m.sum()),
            int(np.mean(n_trs)), int(np.min(n_trs)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ridge", choices=["ridge", "gbdt"])
    ap.add_argument("--blocks", default="", help="逗号分隔，如 20x20,30x60；空=按模型默认")
    ap.add_argument("--buffers", default="", help="逗号分隔的 buffer 半径；空=默认全部")
    a = ap.parse_args()
    global BUFFERS
    if a.buffers:
        BUFFERS = [float(v) for v in a.buffers.split(",")]
    if a.blocks:
        bl = [tuple(int(v) for v in b.split("x")) for b in a.blocks.split(",")]
    elif a.model == "gbdt":
        bl = [(30, 60)]
    else:
        bl = BLOCKS
    rows = M.load()
    y = np.array([r["delta_floor"] for r in rows], float)
    X, yf, keep, keys = M.build(rows, y, 1)
    sub = [rows[i] for i in keep]
    print(f"[load] n={len(sub)}  p={X.shape[1]}", flush=True)

    fit = (M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)
           if a.model == "gbdt" else M.make_ridge(10.0))

    out = {"n": int(len(sub)), "n_feat": int(X.shape[1]), "buffers_deg": BUFFERS,
           "model": a.model,
           "blocks": {}}
    print(f"\n[buffered CV]  R² 随缓冲半径的变化（{a.model}, leak-free, p={X.shape[1]}）")
    for dlat, dlon in bl:
        key = f"{dlat}x{dlon}"
        out["blocks"][key] = {}
        print(f"\n  块 {key}  ({len(set((int((r['lat']+90)//dlat), int((r['lon']+180)//dlon)) for r in sub))} 块)")
        print(f"    {'buffer':>8s} {'R2':>7s} {'n_pred':>7s} {'n_tr均值':>9s} "
              f"{'n_tr最小':>9s} {'训练集剔除':>10s}")
        for b in BUFFERS:
            r2, npred, ntr, ntrm = cv_buffered(sub, X, yf, dlat, dlon, b, fit)
            frac = 1.0 - ntr / max(len(y), 1)
            print(f"    {b:7.1f}° {r2:7.3f} {npred:7d} {ntr:9d} {ntrm:9d} "
                  f"{frac * 100:8.1f}%", flush=True)
            out["blocks"][key][f"{b}"] = {"buffer_deg": b, "cv_r2": r2,
                                          "n_pred": npred, "n_tr_mean": ntr,
                                          "n_tr_min": ntrm,
                                          "train_removed_frac": round(frac, 4)}
        json.dump(out, open(os.path.join(M.EXP, "pathc_buffered_cv.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 判据 ----
    bk = f"{bl[-1][0]}x{bl[-1][1]}"
    print(f"\n[判据] 以 {bk} 为基准")
    b = out["blocks"][bk]
    base = b["0.0"]["cv_r2"]
    for k in [str(x) for x in BUFFERS[1:]]:
        d = b[k]["cv_r2"] - base
        rel = d / max(abs(base), 1e-9)
        print(f"  buffer={k:>5s}°  R2={b[k]['cv_r2']:.3f}  Δ={d:+.3f} ({rel:+.1%})  "
              f"n_tr {b['0.0']['n_tr_mean']}→{b[k]['n_tr_mean']}  "
              f"剔除 {b[k]['train_removed_frac'] * 100:.1f}%")
    json.dump(out, open(os.path.join(M.EXP, "pathc_buffered_cv.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[ok] wrote pathc_buffered_cv.json")


if __name__ == "__main__":
    main()
