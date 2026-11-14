# -*- coding: utf-8 -*-
"""Fig. S11 —— 变量信息含量标度：一个变量在一个 LR 观测下还剩多少空间方差？

数据（全部本地，零下载）
  experiments/variable_scaling.jsonl   Δ_V = 1 − MSE(V_bic, V_HR)/Var(V_HR)
  SEN2VENµS FGMANAUS（129 对，含红边组 b5b6b7b8a，原生 4×）
  SEN2VENµS SO2     （738 对，仅 b2b3b4b8，原生 2×）

三联图
 (a) **匹配倍率**下的 Δ_V：用由 5 m HR 面积降采样得到的合成 LR，使两组波段（可见光
     vs 红边）在同一个倍率上可比。结论：匹配后红边 NDRE 并不吃亏，反而略优于 NDVI
     —— "叶绿素信息本质上需要更细尺度"被否证。
 (b) **跨传感器代价**：Δ_V(合成 LR) − Δ_V(原生 LR)，同倍率逐 patch 配对。真实
     跨传感器观测带来的变量级损失，按变量排序；暗波段主导的指数损失最大。
 (c) **机制**：Δ_V 与"该变量两个组成波段的相关系数"的关系。相关系数越高，指数的
     方差越集中在两波段的高频残差上，而降采样首先抹掉的就是它 —— 单调负关系。
"""
from __future__ import annotations

import json
import os
import sys
import collections
import io
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from scipy import stats  # noqa: E402

EXP = os.path.join(BASE, "experiments")
FIG = os.path.join(EXP, "figs_final", "figS11_variable_scaling.png")
SO2 = os.path.join(BASE, "data", "external", "sen2venus_hf", "SO2")
FG = os.path.join(BASE, "data", "external", "sen2venus", "FGMANAUS")

C_RE = "#C98A1E"       # 红边组 —— 橙
C_VIS = "#2C6E9B"      # 可见光-近红外组 —— 蓝
C_ACT = "#7B4FA8"      # 强调 —— 紫
C_MID = "#9AA5AE"
C_GRN = "#3E7D5A"

LABEL = {
    "ndvi": "NDVI", "ndwi": "NDWI", "greenness": "Greenness",
    "brightness": "Brightness", "evi": "EVI", "nir": "NIR", "red": "Red",
    "ndre": "NDRE (B5)", "ndre2": "NDRE (B6)", "ci_redge": "CI$_{red-edge}$",
    "b8a": "B8A",
}
GROUP = {"ndre": "re", "ndre2": "re", "ci_redge": "re", "b8a": "re"}


def load():
    rows = []
    with open(os.path.join(EXP, "variable_scaling.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def bucket(rows, protocol="direct"):
    d = collections.defaultdict(dict)
    for r in rows:
        if r["protocol"] != protocol:
            continue
        d[(r["src"], r["lrtag"], r["variable"])][r["patch"]] = r["delta_v"]
    return d


def band_corr_so2(n=40):
    fs = sorted(f for f in os.listdir(SO2) if f.endswith("_05m.zip"))[:n]
    M = np.zeros((4, 4))
    for f in fs:
        with zipfile.ZipFile(os.path.join(SO2, f)) as z:
            a = np.clip(tifffile.imread(io.BytesIO(z.read(z.namelist()[0]))).astype(np.float32) / 1e4, 0, 1)
        for i in range(4):
            for j in range(4):
                M[i, j] += np.corrcoef(a[i].ravel(), a[j].ravel())[0, 1]
    return M / len(fs)


FG_VIS_C = np.array([[1.0, .5993, .9281, .0248], [.5993, 1.0, .5756, .6171],
                     [.9281, .5756, 1.0, .0425], [.0248, .6171, .0425, 1.0]])
FG_RE_C = np.array([[1.0, .7633, .7034, .6947], [.7633, 1.0, .9895, .9825],
                    [.7034, .9895, 1.0, .9961], [.6947, .9825, .9961, 1.0]])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load()
    idx = bucket(rows)

    def vals(src, tag, var):
        return idx.get((src, tag, var), {})

    def desc(src, tag, var):
        v = np.array(list(vals(src, tag, var).values()), float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return v.mean(), v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0, v.size

    # ---------------------------------------------------------------- (a)
    re_vars = ["ndre", "ndre2", "ci_redge", "b8a"]
    vis_vars = ["ndvi", "red", "ndwi", "greenness", "evi", "nir", "brightness"]

    # ---------------------------------------------------------------- (b)
    def cost(src, tag_s, tag_n, var):
        a, b = vals(src, tag_s, var), vals(src, tag_n, var)
        ks = sorted(set(a) & set(b))
        x = np.array([a[k] for k in ks], float)
        y = np.array([b[k] for k in ks], float)
        m = np.isfinite(x) & np.isfinite(y)
        d = x[m] - y[m]
        if d.size < 3:
            return None
        se = d.std(ddof=1) / np.sqrt(d.size)
        tc = stats.t.ppf(0.975, d.size - 1)
        return d.mean(), se, tc * se, int(d.size)

    # ---------------------------------------------------------------- (c)
    so2c = band_corr_so2()
    # 变量 → 支配性波段对（在各自组内的 channel 下标）
    PAIR = {
        "fgmanaus|vis|ndvi": ("vis", 2, 3), "fgmanaus|vis|ndwi": ("vis", 1, 3),
        "fgmanaus|vis|greenness": ("vis", 1, 2), "fgmanaus|vis|evi": ("vis", 2, 3),
        "fgmanaus|re|ndre": ("re", 0, 3), "fgmanaus|re|ndre2": ("re", 1, 3),
        "fgmanaus|re|ci_redge": ("re", 2, 3),
        "so2|vis|ndvi": ("vis", 2, 3), "so2|vis|ndwi": ("vis", 1, 3),
        "so2|vis|greenness": ("vis", 1, 2), "so2|vis|evi": ("vis", 2, 3),
    }
    # 参照：同站点、同组的 NIR 型波段（vis→nir, re→b8a），用于扣掉站点整体难度
    REFBAND = {"vis": "nir", "re": "b8a"}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": 0.8, "savefig.dpi": 400,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8})
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.6),
                           gridspec_kw={"width_ratios": [1.15, 1.0, 1.05], "wspace": 0.42})

    # =========================================================== (a)
    a0 = ax[0]
    order = [("fgmanaus", "re", v) for v in re_vars] + [("fgmanaus", "vis", v) for v in vis_vars]
    ys = np.arange(len(order))
    h = 0.36
    # synth2 与 synth4 并排
    for k, (tag, col, dy, lbl) in enumerate([("synth2", C_VIS, +h / 2, "2$\\times$ (synth LR)"),
                                             ("synth4", C_ACT, -h / 2, "4$\\times$ (synth LR)")]):
        m = []
        e = []
        for (src, grp, v) in order:
            d = desc(src, tag, v)
            m.append(d[0] if d else np.nan)
            e.append(d[1] if d else 0)
        a0.barh(ys + dy, m, height=h, xerr=e, color=col, alpha=0.9, label=lbl,
                error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="#444444"))
    a0.set_yticks(ys)
    a0.set_yticklabels([LABEL[v] for (_, _, v) in order], fontsize=7.5)
    for i, (src, grp, v) in enumerate(order):
        if grp == "re":
            a0.get_yticklabels()[i].set_color(C_RE)
    a0.axvline(0, color="#333333", lw=0.8)
    a0.set_xlim(-0.05, 1.02)
    a0.set_xlabel("$\\Delta_V$  (fraction of HR variable variance recovered)")
    a0.set_title("(a) Matched-scale recovery, FGMANAUS (n=129)", fontsize=8, loc="left")
    a0.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=2, frameon=False, handlelength=1.4, columnspacing=1.2)
    a0.invert_yaxis()

    # =========================================================== (b)
    a1 = ax[1]
    spec = [("fgmanaus", [(v, "vis") for v in vis_vars] + [(v, "re") for v in re_vars]),
            ("so2", [(v, "vis") for v in vis_vars])]
    ys2, names, cols, means, errs, ns = [], [], [], [], [], []
    ypos = 0.0
    for src, vs in spec:
        for v, grp in vs:
            c = cost(src, "synth2", "native2", v) if grp == "vis" else cost(src, "synth4", "native4", v)
            if c is None:
                continue
            means.append(c[0]); errs.append(c[2]); ns.append(c[3])
            names.append(LABEL[v]); cols.append(C_RE if grp == "re" else C_VIS)
            ys2.append(ypos); ypos += 1.0
        ypos += 0.9
    a1.barh(ys2, means, height=0.68, xerr=errs, color=cols, alpha=0.9,
            error_kw=dict(elinewidth=0.7, capthick=0.7, ecolor="#444444"))
    a1.set_yticks(ys2)
    a1.set_yticklabels(names, fontsize=7.5)
    a1.invert_yaxis()
    a1.axvline(0, color="#333333", lw=0.8)
    a1.set_xlabel("$\\Delta_V$(synth) $-$ $\\Delta_V$(native)")
    a1.set_title("(b) Cost of real cross-sensor transfer", fontsize=8, loc="left")
    a1.text(0.98, 0.93, "upper block: FGMANAUS (n=129)\nlower block: SO2 (n=738)",
            transform=a1.transAxes, fontsize=6.4, ha="right", va="top", color="#555555")

    # =========================================================== (c)
    a2 = ax[2]
    pts = []
    for key, (grp, i, j) in PAIR.items():
        src, _, var = key.split("|")
        d = desc(src, "synth4", var)
        ref = desc(src, "synth4", REFBAND[grp])
        if d is None or ref is None:
            continue
        C = (FG_RE_C if grp == "re" else FG_VIS_C) if src == "fgmanaus" else so2c
        pts.append((abs(C[i, j]), d[0] - ref[0], d[0], src, var, grp))
    pts.sort()
    xs = np.array([p[0] for p in pts])
    ys3 = np.array([p[1] for p in pts])
    sl, ic, rv, pv, sev = stats.linregress(xs, ys3)
    xg = np.linspace(0.0, 1.02, 50)
    a2.axhline(0, color="#888888", lw=0.7, ls=":")
    a2.plot(xg, sl * xg + ic, color=C_MID, lw=1.0, ls="--", zorder=1)
    LBL_OFF = {"ndvi": (5, 3), "ci_redge": (-5, 5), "ndre2": (-32, 2),
               "ndre": (5, 2), "greenness": (4, -9)}
    for x, y, m, src, var, grp in pts:
        col = C_RE if grp == "re" else C_VIS
        mk = "o" if src == "fgmanaus" else "s"
        a2.plot(x, y, mk, ms=5.0, mfc=col, mec="white", mew=0.6, zorder=3)
        if var in LBL_OFF:
            a2.annotate(LABEL[var], (x, y), textcoords="offset points", xytext=LBL_OFF[var],
                        fontsize=6.3, color="#333333",
                        ha="right" if var == "ci_redge" else "left")
    # 分箱均值（|corr| < 0.5 / 0.5–0.9 / >0.9）
    for lo, hi in [(0.0, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        sel = [p[1] for p in pts if lo <= p[0] < hi]
        if sel:
            a2.plot(0.5 * (lo + min(hi, 1.0)), np.mean(sel), "D", ms=6.5,
                    mfc="white", mec="#222222", mew=0.9, zorder=4)
    a2.set_xlim(-0.02, 1.06)
    a2.set_ylim(-0.295, 0.115)
    a2.set_xlabel("|correlation| between the variable's two bands (5 m)")
    a2.set_ylabel("$\\Delta_V$ $-$ $\\Delta_V$(NIR band), at 4$\\times$")
    a2.set_title("(c) Mechanism: shared-band redundancy", fontsize=8, loc="left")
    a2.text(0.035, 0.97, f"$R^2$={rv**2:.2f}, p={pv:.1e}, n={len(pts)}\n"
                         "circle=FGMANAUS, square=SO2;  diamonds=bin means",
            transform=a2.transAxes, fontsize=6.4, color="#555555", va="top")

    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")

    print("wrote", FIG)
    print("\n(a) matched-scale Δ_V (synth2 / synth4)")
    for (src, grp, v) in order:
        d2, d4 = desc(src, "synth2", v), desc(src, "synth4", v)
        print(f"  {src:9s} {grp:4s} {v:11s} {d2[0]:.4f} / {d4[0]:.4f}")
    print("\n(b) cross-sensor cost")
    for src, vs in spec:
        for v, grp in vs:
            c = cost(src, "synth2", "native2", v) if grp == "vis" else cost(src, "synth4", "native4", v)
            if c:
                print(f"  {src:9s} {v:11s} {c[0]:+.4f} ±{c[2]:.4f}  n={c[3]}")
    print("\n(c) mechanism  (x=|corr of the two bands|, y=Δ_V − Δ_V(NIR ref))")
    for x, y, m, src, var, grp in pts:
        print(f"  |corr|={x:.4f}  d={y:+.4f}  (Δ_V={m:.4f})  {src}/{var}")
    for lo, hi in [(0.0, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        sel = [p[1] for p in pts if lo <= p[0] < hi]
        if sel:
            print(f"  bin |corr|∈[{lo},{min(hi,1.0)}]  mean d = {np.mean(sel):+.4f}  n={len(sel)}")
    print(f"  OLS: slope={sl:.3f}  R2={rv**2:.3f}  p={pv:.2e}  n={len(pts)}")


if __name__ == "__main__":
    main()
