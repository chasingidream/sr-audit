# -*- coding: utf-8 -*-
"""Path C 模型阶梯（分段执行版）—— 每次只跑一格，结果合并入 pathc_models.json。

沙箱对单次持续 CPU 有时间上限（实测 ~4 min 后 SIGTERM），故不可把 4 个
特征集 × 5 个模型 × 5 个 CV 网格塞进一次运行。本脚本用 --cell 选择要跑的组合，
每次增量写盘。

用法：
  python code/tools/pathc_cell.py --y delta_floor --level 1 --model gbdt60
  python code/tools/pathc_cell.py --list
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pathc_models as M  # noqa: E402

MODELS = {
    "ols":      lambda: M.make_ridge(0.0),
    "ridge10":  lambda: M.make_ridge(10.0),
    "ridge100": lambda: M.make_ridge(100.0),
    "knn15":    lambda: M.naive_knn(15),
    "gbdt60":   lambda: M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1),
    "gbdt80":   lambda: M.make_gbdt(n_trees=80, lr=0.12, depth=5, min_leaf=20, seed=2),
    "gbdt40":   lambda: M.make_gbdt(n_trees=40, lr=0.20, depth=6, min_leaf=25, seed=3),
}
# 树模型只跑粗网格（折数少）
GRIDS_TREE = [(20, 20), (30, 60)]
GRIDS_LIN = [(5, 5), (10, 10), (20, 20), (30, 60)]

OUT = os.path.join(M.EXP, "pathc_models.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y", default="delta_floor")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--model", default="gbdt60")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("models:", " ".join(MODELS))
        return

    rows = M.load()
    y = np.array([r.get(a.y, np.nan) for r in rows], float)
    Xf, yf, keep, keys = M.build(rows, y, a.level)
    sub = [rows[i] for i in keep]
    fitter = MODELS[a.model]()
    is_tree = M.spec_of(fitter)[0] == "gbdt"
    grids = GRIDS_TREE if is_tree else GRIDS_LIN

    ins = M.r2_of(yf, fitter(Xf, yf, Xf))
    cvs = {}
    for g in grids:
        cvs[f"{g[0]}x{g[1]}"] = M.cv_spatial(sub, Xf, yf, g[0], g[1], fitter)[0]
    lc, per = M.cv_group(sub, Xf, yf, "campaign", fitter)

    tag = f"L{a.level}|{a.model}"
    try:
        rep = json.load(open(OUT, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        rep = {"n": len(rows)}
    rep.setdefault(a.y, {})[tag] = {
        "p": int(Xf.shape[1]), "n": int(len(yf)), "in_r2": ins, "cv": cvs,
        "cv_leave_campaign": lc, "per_campaign": per, "keys": keys}
    json.dump(rep, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{a.y} {tag}] p={Xf.shape[1]} n={len(yf)} in={ins:.3f} " +
          " ".join(f"{k}={v:.3f}" for k, v in cvs.items()) +
          f" leaveCamp={lc:.3f}")
    print("   per-campaign: " + " ".join(f"{k}={v:.2f}" for k, v in per.items()
                                         if np.isfinite(v)))


if __name__ == "__main__":
    main()
