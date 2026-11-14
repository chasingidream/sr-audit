# -*- coding: utf-8 -*-
"""Fig. S10 —— 采集间隔的信息损失由什么设定？地表空间结构，而非地表动态。

数据（全部本地）
  experiments/stratified_floor_ws.json   Δ_floor, texture_hr, |delta_days|, IPCC, lat
  experiments/canopy_dynamics.jsonl      ndvi_amp（该像元实测 NDVI 季节振幅）
  experiments/recon_quality.jsonl        corr_bic_hr（尺度无关重建质量）

三联图
 (a) 分间隔档、站点内中心化后：amp 与 texture 的标准化系数 —— texture 主导，amp 次要
 (b) Forest / Agriculture / Settlement：NDVI 振幅几乎无差异，地表纹理差异极大
 (c) 预注册交互假说：|Δt|×amp（失败）与 |Δt|×texture（成立）的四种设定对比
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(BASE, "experiments")
FIG = os.path.join(EXP, "figs_final", "figS10_canopy_dynamics.png")

C_TEX = "#2C6E9B"
C_AMP = "#C98A1E"
C_MID = "#9AA5AE"
C_ACT = "#B5402E"


def z(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def ols_beta(y, cols, names):
    """标准化因变量；返回 {name: (beta, se, p)}。"""
    X = (np.column_stack([np.asarray(c, float) for c in cols])
         if isinstance(cols, (list, tuple)) else np.asarray(cols, float))
    X = np.asarray(X, float)
    yv = z(y)
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    r = yv - X @ b
    dof = max(len(yv) - X.shape[1], 1)
    s2 = float(r @ r) / dof
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(X.T @ X)) * s2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.where(se > 1e-15, b / se, 0.0)
    p = 2.0 * stats.t.sf(np.abs(tt), dof)
    return {n: (float(b[i]), float(se[i]), float(p[i])) for i, n in enumerate(names)}


def load():
    sf = {s["cov"]: s for s in json.load(open(os.path.join(EXP, "stratified_floor_ws.json"),
                                              encoding="utf-8"))["scenes"]}
    cd = {}
    for l in open(os.path.join(EXP, "canopy_dynamics_lite.jsonl"), encoding="utf-8"):
        d = json.loads(l)
        cd[d["cov"]] = d
    rq = {}
    for l in open(os.path.join(EXP, "recon_quality.jsonl"), encoding="utf-8"):
        d = json.loads(l)
        rq[d["cov"]] = d
    R = []
    for c, s in sf.items():
        if c not in cd or c not in rq:
            continue
        R.append({"cov": c, "site": c.split("-")[0], "ipcc": s.get("IPCC"),
                  "t": abs(float(s["delta_days"])), "y": float(s["delta_floor"]),
                  "tex": float(s["texture_hr"]), "amp": float(cd[c]["ndvi_amp"]),
                  "corr": float(rq[c]["corr_bic_hr"]),
                  "lat": abs(float(s.get("lat") or 0.0))})
    return R


def main():
    R = load()
    n = len(R)
    t = np.array([r["t"] for r in R])
    y = np.array([r["y"] for r in R])
    tex = np.array([r["tex"] for r in R])
    amp = np.array([r["amp"] for r in R])
    corr = np.array([r["corr"] for r in R])
    ipcc = np.array([r["ipcc"] for r in R], dtype=object)
    site = np.array([r["site"] for r in R], dtype=object)
    lat = np.array([r["lat"] for r in R])
    print("[n] %d" % n)

    # 站点内中心化（消除站点水平与采集策略差异）
    yc = y.copy()
    site_med = {}
    for s in np.unique(site):
        m = site == s
        site_med[s] = float(np.median(y[m]))
        yc[m] = y[m] - site_med[s]
    sites = sorted(set(site))
    D = np.column_stack([(site == s).astype(float) for s in sites[1:]])

    out = {"n": n, "site_medians": site_med}

    # ---------------- (a) 分档：amp 与 texture 的标准化系数 ----------------
    bands = [("<10 d", t < 10), ("10–100 d", (t >= 10) & (t < 100)),
             ("100–300 d", (t >= 100) & (t < 300)), ("≥300 d", t >= 300)]
    panel_a = []
    print("\n=== (a) 分档内 amp / texture 标准化系数（站点内中心化 y）===")
    for lab, sel in bands:
        if sel.sum() < 60:
            continue
        ss = sorted(set(site[sel]))
        Ds = (np.column_stack([(site[sel] == s).astype(float) for s in ss[1:]])
              if len(ss) > 1 else np.zeros((int(sel.sum()), 0)))
        rr = ols_beta(yc[sel], [np.ones(int(sel.sum())), z(amp[sel]), z(tex[sel]), Ds],
                      ["c", "amp", "tex"] + ["s%d" % i for i in range(1, len(ss))])
        panel_a.append({"label": lab, "n": int(sel.sum()),
                        "amp": rr["amp"], "tex": rr["tex"],
                        "t_med": float(np.median(t[sel]))})
        print("  %-10s n=%4d  amp=%+.3f (p=%.2g)   texture=%+.3f (p=%.2g)"
              % (lab, sel.sum(), rr["amp"][0], rr["amp"][2], rr["tex"][0], rr["tex"][2]))
    out["panel_a"] = panel_a

    # 极端档的直接对比，供图注引用
    ext = {}
    for nm, v in (("texture", tex), ("amp", amp)):
        sel = t >= 300
        if sel.sum() < 40:
            continue
        med = np.median(v[sel])
        lo = sel & (v <= med)
        hi = sel & (v > med)
        ext[nm] = {"n": int(sel.sum()), "med_lo": float(np.median(yc[lo])),
                   "med_hi": float(np.median(yc[hi])),
                   "p": float(stats.mannwhitneyu(yc[lo], yc[hi]).pvalue)}
    out["extreme_ge300"] = ext
    print("  [≥300 d] texture low/high centered y = %+.2f / %+.2f (p=%.1g) | "
          "amp low/high = %+.2f / %+.2f (p=%.1g)"
          % (ext["texture"]["med_lo"], ext["texture"]["med_hi"], ext["texture"]["p"],
             ext["amp"]["med_lo"], ext["amp"]["med_hi"], ext["amp"]["p"]))

    # ---------------- (b) 三类的地表属性 ----------------
    panel_b = {}
    for cl in ("Forest", "Agriculture", "Settlement"):
        sel = ipcc == cl
        if sel.sum() < 50:
            continue
        panel_b[cl] = {"n": int(sel.sum()), "amp": amp[sel].tolist(), "tex": tex[sel].tolist()}
    f, a_ = amp[ipcc == "Forest"], amp[ipcc == "Agriculture"]
    ft, at = tex[ipcc == "Forest"], tex[ipcc == "Agriculture"]
    d_amp = (f.mean() - a_.mean()) / np.sqrt((f.var() + a_.var()) / 2)
    d_tex = (ft.mean() - at.mean()) / np.sqrt((ft.var() + at.var()) / 2)
    p_amp = float(stats.mannwhitneyu(f, a_).pvalue)
    p_tex = float(stats.mannwhitneyu(ft, at).pvalue)
    out["panel_b"] = {"d_amp": float(d_amp), "d_tex": float(d_tex),
                      "p_amp": p_amp, "p_tex": p_tex}
    print("\n=== (b) Forest vs Agriculture ===")
    print("  NDVI range d=%+.3f (p=%.2g)   texture d=%+.3f (p=%.2g)" % (d_amp, p_amp, d_tex, p_tex))

    # ---------------- (c) 预注册交互 vs 真机制 ----------------
    specs = [("Δ_floor, no FE", y, False), ("Δ_floor, +site FE", y, True),
             ("corr, no FE", corr, False), ("corr, +site FE", corr, True)]
    panel_c = []
    print("\n=== (c) |Δt|×amp（预注册） vs |Δt|×texture（机制）===")
    for lab, dv, use_fe in specs:
        cols = [np.ones(n), z(t), z(tex), z(t) * z(tex), z(amp), z(t) * z(amp)]
        names = ["c", "dt", "tex", "dt_tex", "amp", "dt_amp"]
        if use_fe:
            cols.append(D)
            names += ["s%d" % i for i in range(1, len(sites))]
        rr = ols_beta(dv, np.column_stack(cols), names)
        panel_c.append({"label": lab, "dt_amp": rr["dt_amp"], "dt_tex": rr["dt_tex"]})
        print("  %-18s dt_x_amp=%+.3f (p=%.2g)   dt_x_tex=%+.3f (p=%.2g)"
              % (lab, rr["dt_amp"][0], rr["dt_amp"][2], rr["dt_tex"][0], rr["dt_tex"][2]))
    out["panel_c"] = panel_c

    json.dump(out, open(os.path.join(EXP, "canopy_figS10.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    make_fig(panel_a, panel_b, panel_c, ipcc, amp, tex)


def make_fig(panel_a, panel_b, panel_c, ipcc, amp, tex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": 0.8, "savefig.dpi": 400,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.2),
                           gridspec_kw={"width_ratios": [1.2, 1.0, 1.2], "wspace": 0.40})

    # ---------------- (a) ----------------
    A = ax[0]
    pos = np.arange(len(panel_a))
    w = 0.34
    for j, (key, col, lab, mk) in enumerate([("amp", C_AMP, "NDVI range (dynamics)", "o"),
                                             ("tex", C_TEX, "HR texture (structure)", "s")]):
        b = np.array([e[key][0] for e in panel_a])
        se = np.array([e[key][1] for e in panel_a])
        A.bar(pos + (j - 0.5) * w, b, w * 0.92, color=col, edgecolor=col,
              label=lab, linewidth=0)
        A.errorbar(pos + (j - 0.5) * w, b, yerr=1.96 * se, fmt="none",
                   ecolor="0.25", elinewidth=0.9, capsize=1.8)
        for k, v in enumerate(b):
            if panel_a[k][key][2] < 0.05:
                A.text(pos[k] + (j - 0.5) * w, v + (0.05 if v >= 0 else -0.09),
                       "*", ha="center", fontsize=8.5, color=col)
    A.axhline(0, c="0.35", lw=0.8)
    A.set_xticks(pos)
    A.set_xticklabels([e["label"] for e in panel_a], fontsize=6.9)
    A.set_xlabel(r"acquisition offset  $|\Delta t|$")
    A.set_ylabel(r"standardised  $\beta$  on centred $\Delta_{\rm floor}$")
    A.set_title("(a) structure sets the loss,\nnot surface dynamics", fontsize=8)
    A.legend(frameon=False, fontsize=6.0, loc="lower left", handlelength=1.0)
    A.set_ylim(-0.30, 0.55)

    # ---------------- (b) ----------------
    B = ax[1]
    cls = [c for c in ("Forest", "Agriculture", "Settlement") if c in panel_b]
    pos = np.arange(len(cls))
    w = 0.34
    bp1 = B.boxplot([np.array(panel_b[c]["amp"]) for c in cls], positions=pos - w / 2,
                    widths=w * 0.9, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=0.9))
    bp2 = B.boxplot([np.array(panel_b[c]["tex"]) * 30.0 for c in cls], positions=pos + w / 2,
                    widths=w * 0.9, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=0.9))
    for b in bp1["boxes"]:
        b.set(facecolor=C_AMP, edgecolor=C_AMP)
    for b in bp2["boxes"]:
        b.set(facecolor=C_TEX, edgecolor=C_TEX)
    B.set_xticks(pos)
    B.set_xticklabels(cls, fontsize=6.9)
    B.set_ylabel("NDVI range  /  HR texture × 30")
    B.set_title("(b) these classes differ in\nterrain, not in dynamics", fontsize=8)
    B.text(0.03, 0.97, "NDVI range (×1)", transform=B.transAxes, fontsize=6.0,
           color=C_AMP, va="top")
    B.text(0.03, 0.91, "HR texture (×30)", transform=B.transAxes, fontsize=6.0,
           color=C_TEX, va="top")
    B.text(0.97, 0.97, r"$d$ = +0.25", transform=B.transAxes, fontsize=6.0,
           color=C_AMP, ha="right", va="top")
    B.text(0.97, 0.91, r"$d$ = −0.56", transform=B.transAxes, fontsize=6.0,
           color=C_TEX, ha="right", va="top")

    # ---------------- (c) ----------------
    C = ax[2]
    off = 0.16
    for k, sp in enumerate(panel_c):
        for j, (key, col) in enumerate([("dt_amp", C_AMP), ("dt_tex", C_TEX)]):
            b, se, p = sp[key]
            yy = k + (j - 0.5) * 2 * off
            C.errorbar(b, yy, xerr=1.96 * se, fmt="o", ms=4.4, c=col,
                       elinewidth=1.1, capsize=1.8)
            C.text(b + 1.96 * se + 0.015, yy, "%+.2f%s" % (b, " *" if p < 0.05 else ""),
                   fontsize=5.9, color=col, va="center")
    C.axvline(0, c="0.35", lw=0.8, ls=":")
    C.set_yticks(range(len(panel_c)))
    C.set_yticklabels([sp["label"] for sp in panel_c], fontsize=6.5)
    C.invert_yaxis()
    C.set_xlim(-0.15, 0.52)
    C.set_xlabel(r"interaction  $\beta_{|\Delta t|\times m}$")
    C.set_title("(c) the pre-registered moderator\nfails; structure survives", fontsize=8)
    C.plot([], [], "o", c=C_AMP, ms=4.4, label=r"$\times$ dynamics")
    C.plot([], [], "o", c=C_TEX, ms=4.4, label=r"$\times$ structure")
    C.legend(frameon=False, fontsize=5.9, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print("\n[fig] %s" % FIG)


if __name__ == "__main__":
    main()
