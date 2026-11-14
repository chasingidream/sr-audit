# -*- coding: utf-8 -*-
"""Path A 分析 —— 「时间惩罚由日历时间设定，还是由地表动态设定？」

合并两批本地产物（零额外下载）：
  experiments/stratified_floor_ws.json  →  y = Δ_floor, t = |delta_days|, 类别/位置/云
  experiments/canopy_dynamics.jsonl     →  m = ndvi_amp（该像元实测 NDVI 动态幅度）

核心检验
  H1（朴素时钟假说）：Δ_floor 只随 |Δt| 变，与地表状态无关。
  H2（活地面假说）  ：同样 |Δt| 下，地表越活跃（ndvi_amp 越大）损失越重
                       ⇒ 交互项 Δ_floor ~ t + m + t×m 的 β_{t×m} < 0。

稳健性
  ① 类内（IPCC）重复；② 安慰剂用 |lat| / cloud 替代 m 做同形交互；
  ③ 云污染口径（ndvi_iqr 替代 ndvi_amp）；④ 暖区子集。

输出 experiments/canopy_analysis.json 与 experiments/figs_final/figS10_canopy_dynamics.{png,pdf}
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF = os.path.join(BASE, "experiments", "stratified_floor_ws.json")
CD = os.path.join(BASE, "experiments", "canopy_dynamics_lite.jsonl")
OUT = os.path.join(BASE, "experiments", "canopy_analysis.json")
FIG = os.path.join(BASE, "experiments", "figs_final", "figS10_canopy_dynamics.png")


# ------------------------------------------------------------------ statistics
def zs(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def ols_t(X, y):
    """最小二乘 + 普通 t 检验。X 含常数列。返回 beta, se, t, p。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtX = X.T @ X
    try:
        XtXi = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError:
        return None
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    dof = max(n - k, 1)
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(XtXi) * s2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 1e-15, beta / se, 0.0)
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    return {"beta": beta, "se": se, "t": t, "p": p, "n": int(n), "dof": int(dof)}


def onecol(x):
    return np.asarray(x, float)[:, None]


def interact(y, a, b, ctrls=None, names=("a", "b", "axb")):
    """标准化交互回归 y ~ a + b + a*b (+ ctrls)。返回系数字典。"""
    cols = [np.ones(len(y)), zs(a), zs(b), zs(a) * zs(b)]
    nm = ["const", names[0], names[1], names[2]]
    if ctrls:
        for k, (v, cn) in enumerate(ctrls):
            cols.append(zs(v))
            nm.append(cn)
    r = ols_t(np.column_stack(cols), zs(y))
    if r is None:
        return None
    return {n: {"beta": float(r["beta"][i]), "se": float(r["se"][i]),
                "t": float(r["t"][i]), "p": float(r["p"][i])}
            for i, n in enumerate(nm)} | {"n": r["n"]}


def slope_ci(x, y):
    """单变量 OLS 斜率 + 95% CI。返回 slope, lo, hi, r, p, n。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 6 or x.std() < 1e-12:
        return None
    lr = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, n - 2)
    half = tcrit * lr.stderr
    return {"slope": float(lr.slope), "lo": float(lr.slope - half),
            "hi": float(lr.slope + half), "r": float(lr.rvalue),
            "p": float(lr.pvalue), "n": int(n)}


def tertiles(x):
    x = np.asarray(x, float)
    q1, q2 = np.percentile(x, [100.0 / 3.0, 200.0 / 3.0])
    return q1, q2


# ---------------------------------------------------------------------- driver
def main():
    sf = json.load(open(SF, encoding="utf-8"))["scenes"]
    SFm = {s["cov"]: s for s in sf}
    cd = [json.loads(l) for l in open(CD, encoding="utf-8")]
    CDm = {r["cov"]: r for r in cd}

    rows = []
    for cov, s in SFm.items():
        c = CDm.get(cov)
        if c is None:
            continue
        if s.get("delta_floor") is None or s.get("delta_days") is None:
            continue
        amp = c.get("ndvi_amp")
        iqr = c.get("ndvi_iqr")
        if amp is None or not np.isfinite(amp):
            continue
        rows.append({
            "cov": cov, "y": float(s["delta_floor"]),
            "t": abs(float(s["delta_days"])),
            "tsign": float(s["delta_days"]),
            "amp": float(amp), "iqr": float(iqr),
            "amp_win": c.get("ndvi_amp_win"),
            "ndvi_n1": float(s["ndvi_mean"]),
            "ndvi_std_sp": float(s["ndvi_std"]),          # 空间 NDVI 异质性
            "spec_het": float(s["spec_het"]),             # 光谱异质性
            "texture_hr": float(s["texture_hr"]),         # HR 纹理能量
            "bright_lr": float(s["bright_lr"]),
            "IPCC": s.get("IPCC"), "lat": float(s.get("lat") or 0.0),
            "lon": float(s.get("lon") or 0.0),
            "cloud": float(s.get("cloud") or 0.0),
            "n_usable": int(c.get("n_usable") or 0),
        })
    print("[info] joined rows = %d  (sf=%d, cd=%d)" % (len(rows), len(SFm), len(CDm)))

    y = np.array([r["y"] for r in rows])
    t = np.array([r["t"] for r in rows])
    amp = np.array([r["amp"] for r in rows])
    lat = np.array([np.abs(r["lat"]) for r in rows])
    cloud = np.array([r["cloud"] for r in rows])
    n1 = np.array([r["ndvi_n1"] for r in rows])
    ndsp = np.array([r["ndvi_std_sp"] for r in rows])
    sphet = np.array([r["spec_het"] for r in rows])
    tex = np.array([r["texture_hr"] for r in rows])

    out = {"n": len(rows)}

    # ---- 0. 描述统计 ----
    def desc(v, nm):
        return {"name": nm, "min": float(v.min()), "p25": float(np.percentile(v, 25)),
                "med": float(np.median(v)), "p75": float(np.percentile(v, 75)),
                "max": float(v.max())}
    out["desc"] = [desc(y, "delta_floor"), desc(t, "abs_delta_days"),
                   desc(amp, "ndvi_amp"), desc(n1, "ndvi_n1"),
                   desc(ndsp, "ndvi_std_spatial"), desc(lat, "abs_lat"),
                   desc(cloud, "cloud")]
    out["corr"] = {"y_t": float(stats.pearsonr(y, t)[0]),
                   "y_amp": float(stats.pearsonr(y, amp)[0]),
                   "t_amp": float(stats.pearsonr(t, amp)[0]),
                   "amp_n1": float(stats.pearsonr(amp, n1)[0]),
                   "amp_ndsp": float(stats.pearsonr(amp, ndsp)[0]),
                   "amp_spec": float(stats.pearsonr(amp, sphet)[0]),
                   "amp_tex": float(stats.pearsonr(amp, tex)[0]),
                   "amp_lat": float(stats.pearsonr(amp, lat)[0])}
    print("[desc] corr y~|dt|=%.3f  y~amp=%.3f  |dt|~amp=%.3f" 
          % (out["corr"]["y_t"], out["corr"]["y_amp"], out["corr"]["t_amp"]))
    print("[confound] amp~NDVI=%.3f  amp~spatial-sd=%.3f  amp~spec-het=%.3f  "
          "amp~HR-texture=%.3f  amp~|lat|=%.3f"
          % (out["corr"]["amp_n1"], out["corr"]["amp_ndsp"], out["corr"]["amp_spec"],
             out["corr"]["amp_tex"], out["corr"]["amp_lat"]))

    # ---- 1. 总体交互回归 ----
    out["interaction_all"] = interact(y, t, amp, names=("abs_dt", "amp", "abs_dt_x_amp"))
    # 关键控制：空间异质性（异质地表本身就难重建，须与时间动态分离）
    out["interaction_all_ctrl"] = interact(
        y, t, amp, ctrls=[(lat, "abs_lat"), (cloud, "cloud"), (n1, "ndvi_n1"),
                          (ndsp, "ndvi_std_spatial"), (sphet, "spec_het")],
        names=("abs_dt", "amp", "abs_dt_x_amp"))
    for k in ("interaction_all", "interaction_all_ctrl"):
        d = out[k]
        if d:
            print("[inter %-18s] abs_dt=%+.3f (p=%.2g)  amp=%+.3f (p=%.2g)  "
                  "abs_dt_x_amp=%+.3f (p=%.2g)  n=%d"
                  % (k, d["abs_dt"]["beta"], d["abs_dt"]["p"],
                     d["amp"]["beta"], d["amp"]["p"],
                     d["abs_dt_x_amp"]["beta"], d["abs_dt_x_amp"]["p"], d["n"]))

    # ---- 2. 按 ndvi_amp 三分位分层 ----
    q1, q2 = tertiles(amp)
    out["amp_tertile_cuts"] = [float(q1), float(q2)]
    strata = []
    labels = ["dormant", "intermediate", "active"]
    for k, (lo, hi) in enumerate([(-np.inf, q1), (q1, q2), (q2, np.inf)]):
        sel = (amp >= lo) & (amp < hi)
        sl = slope_ci(t[sel], y[sel])
        strata.append({"label": labels[k], "n": int(sel.sum()),
                       "amp_lo": None if np.isinf(lo) else float(lo),
                       "amp_hi": None if np.isinf(hi) else float(hi),
                       "amp_med": float(np.median(amp[sel])) if sel.sum() else None,
                       "mean_y": float(y[sel].mean()) if sel.sum() else None, "slope": sl})
        if sl:
            print("[stratum %-12s] n=%4d  amp_med=%.3f  slope(y~|dt|)=%+.4f "
                  "[%+.4f,%+.4f]  r=%+.3f p=%.2g"
                  % (labels[k], sel.sum(), np.median(amp[sel]), sl["slope"],
                     sl["lo"], sl["hi"], sl["r"], sl["p"]))
    out["amp_strata"] = strata

    # ---- 3. 类内交互（IPCC）----
    by_class = {}
    for cl in sorted(set(r["IPCC"] for r in rows if r["IPCC"])):
        sel = np.array([r["IPCC"] == cl for r in rows])
        if sel.sum() < 60:
            continue
        d = interact(y[sel], t[sel], amp[sel],
                     ctrls=[(lat[sel], "abs_lat"), (n1[sel], "ndvi_n1"),
                            (ndsp[sel], "ndvi_std_spatial")],
                     names=("abs_dt", "amp", "abs_dt_x_amp"))
        sl_all = slope_ci(t[sel], y[sel])
        lo_sel = sel & (amp < q1)
        hi_sel = sel & (amp >= q2)
        d["slope_all"] = sl_all
        d["slope_dormant"] = slope_ci(t[lo_sel], y[lo_sel])
        d["slope_active"] = slope_ci(t[hi_sel], y[hi_sel])
        d["n_total"] = int(sel.sum())
        by_class[cl] = d
        if d:
            print("[class %-11s] n=%4d  abs_dt_x_amp=%+.3f (p=%.2g) | slope dormant %s vs active %s"
                  % (cl, sel.sum(), d["abs_dt_x_amp"]["beta"], d["abs_dt_x_amp"]["p"],
                     ("%+.3f" % d["slope_dormant"]["slope"]) if d["slope_dormant"] else "n/a",
                     ("%+.3f" % d["slope_active"]["slope"]) if d["slope_active"] else "n/a"))
    out["by_class"] = by_class

    # ---- 4. 安慰剂 / 空间异质性对照 ----
    #   若「空间异质性」的交互项同样显著，则本结果不能归因于时间动态。
    placebo = {}
    for nm, v in [("abs_lat", lat), ("cloud", cloud), ("ndvi_n1", n1),
                  ("ndvi_std_spatial", ndsp), ("spec_het", sphet),
                  ("texture_hr", tex)]:
        d = interact(y, t, v, names=("abs_dt", nm, "abs_dt_x_" + nm))
        placebo[nm] = d
        if d:
            key = "abs_dt_x_" + nm
            print("[placebo %-16s] %s=%+.3f (p=%.2g)" % (nm, key, d[key]["beta"], d[key]["p"]))
    out["placebo"] = placebo

    # ---- 5. 稳健口径：iqr 替代 amp ----
    iqr = np.array([r["iqr"] for r in rows])
    out["interaction_iqr"] = interact(y, t, iqr, names=("abs_dt", "ndvi_iqr", "abs_dt_x_iqr"))
    # 暖区子集（|lat|<45）避免寒区物候把 amp 拉大
    warm = lat < 45.0
    out["interaction_warm"] = interact(y[warm], t[warm], amp[warm],
                                       names=("abs_dt", "amp", "abs_dt_x_amp"))
    out["n_warm"] = int(warm.sum())
    # n_usable 门槛
    for thr in (6, 8):
        sel = np.array([r["n_usable"] >= thr for r in rows])
        out["interaction_nusable_ge%d" % thr] = interact(
            y[sel], t[sel], amp[sel], names=("abs_dt", "amp", "abs_dt_x_amp"))
        out["n_nusable_ge%d" % thr] = int(sel.sum())
    for k in ("interaction_iqr", "interaction_warm", "interaction_nusable_ge6",
              "interaction_nusable_ge8"):
        d = out.get(k)
        if d:
            key = [x for x in d if x.startswith("abs_dt_x")][0]
            print("[robust %-24s] %s=%+.3f (p=%.2g)  n=%d"
                  % (k, key, d[key]["beta"], d[key]["p"], d["n"]))

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[saved] %s" % OUT)

    make_fig(rows, out, q1, q2)


# ------------------------------------------------------------------------ fig
def make_fig(rows, out, q1, q2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": 0.8, "savefig.dpi": 400,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.05),
                           gridspec_kw={"width_ratios": [1.15, 1.0, 1.15], "wspace": 0.34})

    y = np.array([r["y"] for r in rows])
    t = np.array([r["t"] for r in rows])
    amp = np.array([r["amp"] for r in rows])
    lo_mask = amp < q1
    hi_mask = amp >= q2
    mid_mask = ~lo_mask & ~hi_mask

    # (a) y vs |dt| 按 amp 三分位
    A = ax[0]
    for msk, col, lab in [(lo_mask, "#2C6E9B", "dormant (low s.d.)"),
                          (mid_mask, "#9AA5AE", "intermediate"),
                          (hi_mask, "#B5402E", "active (high s.d.)")]:
        A.scatter(t[msk], y[msk], s=3, c=col, alpha=0.30, linewidths=0, rasterized=True)
        sl = slope_ci(t[msk], y[msk])
        if sl:
            xs = np.array([t[msk].min(), t[msk].max()])
            A.plot(xs, y[msk].mean() + sl["slope"] * (xs - t[msk].mean()), c=col, lw=1.6,
                   label="%s  (%+.3f/p=%.1g)" % (lab, sl["slope"], sl["p"]))
    A.axhline(0, c="0.35", lw=0.7, ls="--")
    A.set_xlabel("|acquisition offset|  |Δt|  (days)")
    A.set_ylabel(r"$\Delta_{\rm floor}$  (dB)")
    A.set_title("(a) loss vs. clock offset,\nstratified by surface dynamics", fontsize=8)
    A.legend(frameon=False, fontsize=6.0, loc="lower left", handlelength=1.1)
    A.set_ylim(-16, 4)

    # (b) 三分位斜率森林图
    B = ax[1]
    ys = []
    for k, (lab, col) in enumerate([("dormant\n(low s.d.)", "#2C6E9B"),
                                    ("intermediate", "#9AA5AE"),
                                    ("active\n(high s.d.)", "#B5402E")]):
        sl = out["amp_strata"][k]["slope"]
        if not sl:
            continue
        ys.append(k)
        B.errorbar(sl["slope"], k, xerr=[[sl["slope"] - sl["lo"]], [sl["hi"] - sl["slope"]]],
                   fmt="o", ms=5, c=col, ecolor=col, elinewidth=1.3, capsize=2.5)
        B.text(sl["slope"] + (sl["hi"] - sl["lo"]) * 0.28 + 0.006, k + 0.17,
               "n=%d" % sl["n"], fontsize=6, color="0.35")
    B.axvline(0, c="0.35", lw=0.7, ls=":")
    B.set_yticks([0, 1, 2])
    B.set_yticklabels(["dormant", "intermediate", "active"], fontsize=7.5)
    B.set_ylim(-0.6, 2.6)
    B.set_xlabel(r"slope  $\partial\Delta_{\rm floor}/\partial|\Delta t|$")
    B.set_title("(b) each extra day of offset\ncosts more on active surfaces", fontsize=8)

    # (c) 交互项系数（真 vs 安慰剂）
    C = ax[2]
    items = []
    da = out.get("interaction_all_ctrl") or out.get("interaction_all")
    if da:
        items.append(("temporal dynamics\n(NDVI range)", da["abs_dt_x_amp"], "#B5402E"))
    for nm, lab, col in [("ndvi_std_spatial", "placebo: spatial NDVI s.d.", "#7A7A7A"),
                         ("spec_het", "placebo: spectral heterogeneity", "#7A7A7A"),
                         ("ndvi_n1", "placebo: mean greenness", "#B08A3E"),
                         ("abs_lat", "placebo: |latitude|", "#7A7A7A"),
                         ("cloud", "placebo: cloud cover", "#7A7A7A")]:
        d = out["placebo"].get(nm)
        if d:
            items.append((lab, d["abs_dt_x_" + nm], col))
    for k, (lab, d, col) in enumerate(items):
        if not d:
            continue
        C.errorbar(d["beta"], k, xerr=1.96 * d["se"], fmt="o", ms=5, c=col,
                   ecolor=col, elinewidth=1.4, capsize=2.5)
        star = "*" if d["p"] < 0.05 else ""
        C.text(d["beta"], k + 0.20, "%+.2f%s" % (d["beta"], star), fontsize=6.2,
               color=col, ha="center")
    C.axvline(0, c="0.35", lw=0.7, ls=":")
    C.set_yticks(range(len(items)))
    C.set_yticklabels([i[0] for i in items], fontsize=6.6)
    C.invert_yaxis()
    C.set_xlabel(r"interaction  $\beta_{|\Delta t|\times m}$  (s.d. units)")
    C.set_title("(c) only surface dynamics\nmodulates the offset penalty", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print("[fig] %s" % FIG)


if __name__ == "__main__":
    main()
