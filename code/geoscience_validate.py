# -*- coding: utf-8 -*-
"""
地学方法补强：用**完全独立于影像**的地表/气候描述量验证「可恢复信息预算是地表属性」。

数据
  experiments/stratified_floor_ws.json    n=3927 景：delta_floor + 影像代理量 + lat/lon + IPCC/cloud/delta_days
  experiments/climate_independent.jsonl   由 code/sample_worldclim_monthly.py 产出：
                                          WorldClim 2.1 10 arc-min 气候学（Fick & Hijmans 2017, 1970-2000）
                                          —— 气象站插值面，与本文任何传感器无关，且时间不变。

分析
  1. 描述性分层：De Martonne 干旱度 P/(T+10) 分级 -> Δ_floor（KW η²）
     （该指数在 T -> -10 °C 发散，故仅在 T >= -9 °C 的子集上定义；被排除的寒区场景单独报告）
  2. 相关性：Δ_floor vs 各独立量（Pearson / Spearman）
  3. 回归对照（样本内）：影像代理量 / 独立量 / 合并 / IPCC one-hot
  4. **空间留出交叉验证**（留一块法，块尺寸敏感性）：直接回应
     「用同一张影像预测同一张影像」与「空间自相关造出的假拟合」两种质疑。
输出
  experiments/geoscience_independent_validation.json
  experiments/figs_final/figS8_independent_surface.png / .pdf
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
FIGS = os.path.join(EXP, "figs_final")
os.makedirs(FIGS, exist_ok=True)

IMG_FEATS = ["ndvi_mean", "ndvi_std", "bright_lr", "spec_het", "bright_hr", "texture_hr"]
IND_FEATS = ["log_precip", "t_mean_ann", "t_seasonality", "precip_cv", "elevation", "abs_lat"]
ARID_MIN_D = 1.0          # De Martonne 分母下限（T >= -9 °C）

CV_BLOCK_SIZES = [(5, 5), (10, 10), (20, 20), (30, 60)]


def load():
    d = json.load(open(os.path.join(EXP, "stratified_floor_ws.json"), encoding="utf-8"))
    clim = {}
    for line in open(os.path.join(EXP, "climate_independent.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            clim[r["cov"]] = r
    rows = []
    for s in d["scenes"]:
        c = clim.get(s.get("cov"))
        if not c or s.get("delta_floor") is None:
            continue
        t, pr = c["t_mean_ann"], c["precip_ann"]
        arid = pr / (t + 10.0) if (t + 10.0) >= ARID_MIN_D else np.nan
        rec = {"cov": s["cov"], "df": float(s["delta_floor"]),
               "lat": s["lat"], "lon": s["lon"], "IPCC": s.get("IPCC"),
               "precip_ann": pr, "t_mean_ann": t,
               "log_precip": float(np.log10(max(pr, 0.1))),
               "t_seasonality": c["t_seasonality"], "precip_cv": c["precip_cv"],
               "elevation": c["elevation"], "abs_lat": abs(float(s["lat"])),
               "aridity": float(arid) if np.isfinite(arid) else np.nan}
        for f in IMG_FEATS:
            rec[f] = s.get(f)
        rows.append(rec)
    return rows


def ols(X, y):
    n, p = X.shape
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    Xd = np.column_stack([np.ones(n), Xz])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return b, float(r2), float(adj), pred


def fit(rows, feats, y):
    X = np.array([[r[f] if r[f] is not None else np.nan for f in feats] for r in rows], float)
    m = ~np.isnan(X).any(axis=1)
    b, r2, adj, _ = ols(X[m], y[m])
    return {"r2": r2, "adj_r2": adj, "n": int(m.sum()),
            "beta": {f: float(v) for f, v in zip(feats, b[1:])}}


def block_keys(rows, dlat, dlon):
    return [(int((r["lat"] + 90) // dlat), int((r["lon"] + 180) // dlon)) for r in rows]


def spatial_cv(rows, feats, y, dlat, dlon, min_tr=20):
    keys = block_keys(rows, dlat, dlon)
    X = np.array([[r[f] if r[f] is not None else np.nan for f in feats] for r in rows], float)
    pred = np.full(len(rows), np.nan)
    for k in sorted(set(keys)):
        te = np.array([i for i, kk in enumerate(keys) if kk == k])
        tr = np.array([i for i, kk in enumerate(keys) if kk != k])
        tr = tr[~np.isnan(X[tr]).any(axis=1)]
        te = te[~np.isnan(X[te]).any(axis=1)]
        if len(tr) < min_tr or len(te) == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        Xd = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        b, *_ = np.linalg.lstsq(Xd, y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(len(te)), (X[te] - mu) / sd]) @ b
    m = ~np.isnan(pred)
    if m.sum() < 30:
        return float("nan"), int(m.sum()), len(set(keys))
    r2 = 1 - ((y[m] - pred[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    return float(r2), int(m.sum()), len(set(keys))


def kw_eta2(groups):
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 2:
        return None, None, None
    H, p = stats.kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    return float(H), float(p), float((H - k + 1) / (n - k))


def aridity_class(a):
    if not np.isfinite(a):
        return None
    if a < 5:
        return "hyper-arid"
    if a < 10:
        return "arid"
    if a < 20:
        return "semi-arid"
    if a < 30:
        return "dry sub-humid"
    return "humid"


def main():
    rows = load()
    n = len(rows)
    y = np.array([r["df"] for r in rows], float)
    out = {"n": n, "n_cryosphere_excluded": int(np.isnan([r["aridity"] for r in rows]).sum()),
           "global_mean_df": float(y.mean()), "frac_negative": float((y < 0).mean())}
    print("[data] n=%d  (cryosphere T<-9C excluded from aridity strata: %d)"
          % (n, out["n_cryosphere_excluded"]))

    # ---- 1. 干旱度分层 ----
    order = ["hyper-arid", "arid", "semi-arid", "dry sub-humid", "humid"]
    cls = [aridity_class(r["aridity"]) for r in rows]
    for r, c in zip(rows, cls):
        r["arid_class"] = c
    groups = {c: y[[i for i, cc in enumerate(cls) if cc == c]] for c in order}
    H, p, eta2 = kw_eta2([g for g in groups.values() if len(g)])
    out["aridity_strat"] = {c: {"n": int(len(g)), "mean_df": float(g.mean()) if len(g) else None,
                                "frac_neg": float((g < 0).mean()) if len(g) else None}
                            for c, g in groups.items()}
    out["aridity_kw"] = {"H": H, "p": p, "eta2": eta2}
    print("[aridity] KW H=%.1f p=%.2g eta2=%.3f" % (H, p, eta2))
    for c in order:
        g = groups[c]
        if len(g):
            print("   %-14s n=%4d  mean_df=%+.2f  frac_neg=%.3f" % (c, len(g), g.mean(), (g < 0).mean()))

    # 干旱 vs 湿润 两分对比（与正文 vegetated-vs-built 同一口径）
    dry_set = np.concatenate([groups["hyper-arid"], groups["arid"]]) if groups["arid"].size else np.array([])
    wet_set = groups["humid"]
    if dry_set.size > 30 and wet_set.size > 30:
        U, pu = stats.mannwhitneyu(wet_set, dry_set, alternative="less")
        d = (wet_set.mean() - dry_set.mean()) / np.sqrt((wet_set.var(ddof=1) + dry_set.var(ddof=1)) / 2)
        out["arid_vs_humid"] = {"n_dry": int(dry_set.size), "mean_dry": float(dry_set.mean()),
                                "n_humid": int(wet_set.size), "mean_humid": float(wet_set.mean()),
                                "diff": float(wet_set.mean() - dry_set.mean()),
                                "mwu_p": float(pu), "cohens_d": float(d)}
        print("[arid vs humid] dry(%.2f, n=%d) vs humid(%.2f, n=%d): diff=%.2f dB, MWU p=%.2g, d=%.2f"
              % (dry_set.mean(), dry_set.size, wet_set.mean(), wet_set.size,
                 wet_set.mean() - dry_set.mean(), pu, d))

    # 寒区对照（被排除的 3%）
    m_cold = np.isnan([r["aridity"] for r in rows])
    out["cryosphere"] = {"n": int(m_cold.sum()), "mean_df": float(y[m_cold].mean()),
                         "warm_mean_df": float(y[~m_cold].mean()),
                         "frac_neg": float((y[m_cold] < 0).mean()),
                         "median_abs_lat": float(np.median([r["abs_lat"] for r, m in zip(rows, m_cold) if m]))}
    print("[cryosphere] n=%d mean_df=%+.2f (vs warm %+.2f), frac_neg=%.3f, median|lats|=%.1f"
          % (out["cryosphere"]["n"], out["cryosphere"]["mean_df"],
             out["cryosphere"]["warm_mean_df"], out["cryosphere"]["frac_neg"],
             out["cryosphere"]["median_abs_lat"]))

    # ---- 2. 相关性 ----
    corr = {}
    for f in IND_FEATS + ["aridity"]:
        v = np.array([r[f] if r[f] is not None else np.nan for r in rows], float)
        m = np.isfinite(v)
        if m.sum() > 30:
            pr_ = stats.pearsonr(v[m], y[m])
            sp_ = stats.spearmanr(v[m], y[m])
            corr[f] = {"pearson_r": float(pr_[0]), "pearson_p": float(pr_[1]),
                       "spearman_r": float(sp_[0]), "spearman_p": float(sp_[1]), "n": int(m.sum())}
            print("   corr %-16s pearson r=%+.3f (p=%.2g)  spearman=%+.3f (n=%d)"
                  % (f, pr_[0], pr_[1], sp_[0], m.sum()))
    out["correlations"] = corr

    # ---- 3. 样本内回归对照 ----
    cats = sorted({r["IPCC"] for r in rows if r.get("IPCC")})
    for i, c in enumerate(cats):
        for r in rows:
            r["ipcc_%d" % i] = 1.0 if r.get("IPCC") == c else 0.0
    cat_feats = ["ipcc_%d" % i for i in range(len(cats))]

    out["reg_image"] = fit(rows, IMG_FEATS, y)
    out["reg_independent"] = fit(rows, IND_FEATS, y)
    out["reg_combined"] = fit(rows, IMG_FEATS + IND_FEATS, y)
    out["reg_category"] = fit(rows, cat_feats, y)
    print("[reg in-sample] image R2=%.3f | independent R2=%.3f | combined R2=%.3f | IPCC one-hot R2=%.3f"
          % (out["reg_image"]["r2"], out["reg_independent"]["r2"],
             out["reg_combined"]["r2"], out["reg_category"]["r2"]))

    # ---- 4. 空间留出 CV（块尺度敏感性）----
    cv = {}
    for dlat, dlon in CV_BLOCK_SIZES:
        key = "%dx%d" % (dlat, dlon)
        cv[key] = {}
        for name, feats in [("image", IMG_FEATS), ("independent", IND_FEATS),
                            ("combined", IMG_FEATS + IND_FEATS), ("category", cat_feats)]:
            r2, nn, nb = spatial_cv(rows, feats, y, dlat, dlon)
            cv[key][name] = {"cv_r2": r2, "n": nn, "n_blocks": nb}
        print("[cv %-6s] image=%.3f independent=%.3f combined=%.3f category=%.3f (blocks=%s)"
              % (key, cv[key]["image"]["cv_r2"], cv[key]["independent"]["cv_r2"],
                 cv[key]["combined"]["cv_r2"], cv[key]["category"]["cv_r2"], key))
    out["spatial_cv"] = cv

    # ---- 4b. 暖区子集复核（排除寒区，确认结论不由 3% 极端景驱动）----
    warm = [r for r in rows if np.isfinite(r["aridity"])]
    yw = np.array([r["df"] for r in warm], float)
    key = "%dx%d" % CV_BLOCK_SIZES[-1]
    out["warm_subset"] = {"n": len(warm)}
    out["warm_subset"]["reg_image"] = fit(warm, IMG_FEATS, yw)
    out["warm_subset"]["reg_independent"] = fit(warm, IND_FEATS, yw)
    out["warm_subset"]["reg_combined"] = fit(warm, IMG_FEATS + IND_FEATS, yw)
    out["warm_subset"]["reg_category"] = fit(warm, cat_feats, yw)
    out["warm_subset"]["spatial_cv_%s" % key] = {}
    for name, feats in [("image", IMG_FEATS), ("independent", IND_FEATS),
                        ("combined", IMG_FEATS + IND_FEATS), ("category", cat_feats)]:
        r2, nn, nb = spatial_cv(warm, feats, yw, CV_BLOCK_SIZES[-1][0], CV_BLOCK_SIZES[-1][1])
        out["warm_subset"]["spatial_cv_%s" % key][name] = {"cv_r2": r2, "n": nn}
    print("[warm n=%d] in-sample image=%.3f indep=%.3f comb=%.3f cat=%.3f | CV(%s) image=%.3f indep=%.3f comb=%.3f cat=%.3f"
          % (len(warm), out["warm_subset"]["reg_image"]["r2"], out["warm_subset"]["reg_independent"]["r2"],
             out["warm_subset"]["reg_combined"]["r2"], out["warm_subset"]["reg_category"]["r2"], key,
             out["warm_subset"]["spatial_cv_%s" % key]["image"]["cv_r2"],
             out["warm_subset"]["spatial_cv_%s" % key]["independent"]["cv_r2"],
             out["warm_subset"]["spatial_cv_%s" % key]["combined"]["cv_r2"],
             out["warm_subset"]["spatial_cv_%s" % key]["category"]["cv_r2"]))

    json.dump(out, open(os.path.join(EXP, "geoscience_independent_validation.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[ok] wrote geoscience_independent_validation.json")

    make_fig(rows, y, groups, order, out)


def make_fig(rows, y, groups, order, out):
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    C = {"image": "#7a6aa8", "independent": "#e08a3c", "combined": "#4f8fbc", "category": "#9aa0a6"}

    # (a) 干旱度分级箱线（独立气候）
    ax = axes[0]
    labs = [c for c in order if len(groups[c])]
    ax.boxplot([groups[c] for c in labs], patch_artist=True, widths=0.6, showfliers=False,
               medianprops=dict(color="#222", lw=1.0),
               boxprops=dict(facecolor="#b9a7d1", edgecolor="#444", alpha=0.8))
    ax.axhline(0, color="#bbb", lw=0.8, ls="--")
    means = [groups[c].mean() for c in labs]
    ax.plot(range(1, len(labs) + 1), means, "o-", color="#d1611a", ms=3.2, lw=1.1, zorder=5)
    top = ax.get_ylim()[1]
    for i, c in enumerate(labs):
        ax.text(i + 1, top * 0.72, u"n=%d" % len(groups[c]), ha="center", fontsize=5.8, color="#555")
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels(labs, rotation=32, ha="right", fontsize=6.3)
    ax.set_ylabel(u"\u0394_floor (dB)")
    ax.set_title("(a) imagery-free aridity class", fontsize=7.5)
    ax.text(0.97, 0.04, u"KW \u03b7\u00b2=%.3f\np=%.1g" % (out["aridity_kw"]["eta2"], out["aridity_kw"]["p"]),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color="#444")

    # (b) Δ_floor vs aridity
    ax = axes[1]
    a = np.array([r["aridity"] for r in rows], float)
    m = np.isfinite(a)
    ax.scatter(a[m], y[m], s=3, c="#7a6aa8", alpha=0.22, edgecolors="none")
    co = np.polyfit(a[m], y[m], 1)
    xs = np.linspace(np.percentile(a[m], 1), np.percentile(a[m], 99), 50)
    ax.plot(xs, np.polyval(co, xs), color="#d1611a", lw=1.3)
    # 分箱均值（20 箱，等频）
    qs = np.quantile(a[m], np.linspace(0, 1, 21))
    bx, by = [], []
    for i in range(20):
        sel = (a[m] >= qs[i]) & (a[m] <= qs[i + 1])
        if sel.sum() >= 20:
            bx.append(a[m][sel].mean())
            by.append(y[m][sel].mean())
    ax.plot(bx, by, "-", color="#2b2b2b", lw=1.4, marker="o", ms=2.6, zorder=6)
    rr = out["correlations"]["aridity"]
    ax.text(0.03, 0.05, u"r=%.2f, p=%.0e" % (rr["pearson_r"], rr["pearson_p"]),
            transform=ax.transAxes, fontsize=6.2, color="#444")
    ax.set_ylim(np.percentile(y[m], 0.5), np.percentile(y[m], 99.5))
    ax.axhline(0, color="#bbb", lw=0.8, ls="--")
    ax.set_xlabel(u"aridity  P/(T+10)   (independent)")
    ax.set_ylabel(u"\u0394_floor (dB)")
    ax.set_title("(b) budget vs independent climate", fontsize=7.5)

    # (c) 空间 CV：块尺寸敏感性
    ax = axes[2]
    keys = [k for k in out["spatial_cv"]]
    xs = np.arange(len(keys))
    w = 0.2
    for j, nm in enumerate(["image", "independent", "combined", "category"]):
        vals = [out["spatial_cv"][k][nm]["cv_r2"] for k in keys]
        ax.bar(xs + (j - 1.5) * w, vals, width=w, color=C[nm], label=nm)
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([k.replace("x", u"\u00b0\u00d7") + u"\u00b0" for k in keys], fontsize=6.2)
    ax.set_ylabel(u"spatial-CV R\u00b2 (out-of-sample)")
    ax.set_title("(c) transfer to withheld regions", fontsize=7.5)
    ax.legend(fontsize=5.6, frameon=False, loc="upper right", ncol=1, handlelength=1.1)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figS8_independent_surface.png"), dpi=600)
    fig.savefig(os.path.join(FIGS, "figS8_independent_surface.pdf"))
    print("[ok] wrote figS8_independent_surface.{png,pdf}")


if __name__ == "__main__":
    main()
