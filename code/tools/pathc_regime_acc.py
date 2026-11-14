"""M3 (2026-10-05): 图集的"决策单位"精度——leave-one-campaign-out 的符号（regime）正确率。

动机：图集此前只用 R² 报精度（0.73 / 0.51 …），但决策规则要的是**符号正确率**
（一个场景该判 Δ_floor<0 还是 >0）。R² 是连续精度；符号正确率才是"地图能否用于决策"的
诚实度量。本脚本复用 pathc_models 的 leak-free 特征与 GBDT，取 OOF 预测后算：
  整体符号正确率 / 平衡正确率（两类宏平均）/ AUC / 与"恒判负"平凡基线的对照。
输出 experiments/pathc_regime_acc.json。
"""
import os, json, numpy as np
import pathc_models as M

EXP = M.EXP


def auc(y_sign, score):
    """y_sign ∈ {0,1}；score 越大越倾向正。用秩公式算 AUC。"""
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = int(y_sign.sum()); n0 = len(y_sign) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r1 = ranks[y_sign == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def report_tier(name, y, pred):
    m = np.isfinite(pred)
    yv, pv = y[m], pred[m]
    ysign = (yv > 0).astype(int)          # 1=正（Δ_floor>0，预算尚存）
    psign = (pv > 0).astype(int)
    acc = float((ysign == psign).mean())
    pos, neg = int(ysign.sum()), int((1 - ysign).sum())
    tpr = float(((ysign == 1) & (psign == 1)).sum()) / pos if pos else float("nan")
    tnr = float(((ysign == 0) & (psign == 0)).sum()) / neg if neg else float("nan")
    bal = (tpr + tnr) / 2 if np.isfinite(tpr + tnr) else float("nan")
    base_neg = float((ysign == 0).mean())     # 恒判负的平凡正确率
    pos_frac = float(ysign.mean())
    return {
        "tier": name, "n": int(m.sum()),
        "pos_frac_true": pos_frac,
        "baseline_always_negative_acc": base_neg,
        "sign_accuracy": acc,
        "balanced_accuracy": bal,
        "sensitivity_tpr": tpr, "specificity_tnr": tnr,
        "auc": auc(ysign, pv),
        "acc_gain_over_baseline": acc - base_neg,
    }


def main():
    rows = M.load()
    y = np.array([r["delta_floor"] for r in rows], float)
    Xf, yf, keep, keys = M.build(rows, y, 1)          # level-1 leak-free (p=12)
    sub = [rows[i] for i in keep]
    print(f"[load] n={len(rows)}  leak-free p={Xf.shape[1]}  keys={keys}", flush=True)

    gbdt = M.make_gbdt(n_trees=60, lr=0.15, depth=4, min_leaf=30, seed=1)

    out = {"n": len(sub), "p": int(Xf.shape[1]), "tiers": {}}

    # 最严 tier：留一采集战役（体制外推）——决策诚实度最高的口径
    lc, per, pred_c = M.cv_group(sub, Xf, yf, "campaign", gbdt, ret_pred=True)
    r = report_tier("leave_one_campaign_out", np.array([s["delta_floor"] for s in sub], float), pred_c)
    r["r2"] = float(lc); r["per_campaign_r2"] = per
    out["tiers"]["leave_campaign"] = r
    print("[leave-campaign] ", json.dumps(r, ensure_ascii=False), flush=True)

    # 次严 tier：30°×60° 空间块（体制外推的另一口径）
    r2b, nb = M.cv_spatial(sub, Xf, yf, 30, 60, gbdt)
    out["tiers"]["spatial_30x60"] = {"r2": float(r2b), "n": int(nb),
                                     "note": "cv_spatial 不回传 OOF，仅报 R²"}
    print("[spatial 30x60] ", r2b, nb, flush=True)

    # 产品级 tier：1° 格网互斥 K-fold（K=10）——"同体制内查询"的决策正确率
    import pathc_atlas as A
    keys = np.array([A.cell_id(r["lat"], r["lon"], 1.0) for r in sub], dtype=np.int64)
    uniq = sorted(set(map(tuple, keys.tolist())))
    kid = {k: i for i, k in enumerate(uniq)}
    cell = np.array([kid[k] for k in map(tuple, keys.tolist())])
    rng = np.random.RandomState(7)
    fold_of_cell = rng.permutation(len(uniq)) % 10
    fold = fold_of_cell[cell]
    pred_cell = np.full(len(yf), np.nan)
    for k in range(10):
        te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
        pred_cell[te] = gbdt(Xf[tr], yf[tr], Xf[te])
    rcell = report_tier("kfold_by_cell_1deg", yf, pred_cell)
    m = np.isfinite(pred_cell)
    rcell["r2"] = float(M.r2_of(yf[m], pred_cell[m]))
    out["tiers"]["cell_1deg"] = rcell
    print("[cell 1deg] ", json.dumps(rcell, ensure_ascii=False), flush=True)

    # 落盘
    json.dump(out, open(os.path.join(EXP, "pathc_regime_acc.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[saved] experiments/pathc_regime_acc.json", flush=True)


if __name__ == "__main__":
    main()
