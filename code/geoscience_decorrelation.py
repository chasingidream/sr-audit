# -*- coding: utf-8 -*-
"""
去相关选择性机制：把逐场景 Δ_floor 归因到「观测状态」（云量、|LR–HR 采集间隔|）
而非「地表类别 / 分辨率」。

为什么这一步决定了本文能否构成地球系统知识：
  若该缺口只是"传感器分辨率不足"，它属于工程/评测问题；
  若它在**同一土地覆盖类、同一纬度带、同一高程带内部**仍由"云"与"采集间隔"决定，
  且该效应对植被冠层大、对建成区为零，则它是一个**可观测性 (observability) 的物理量**——
  缺口由地表—大气—传感器三者耦合的去相关造成，主控项是"地表自身的时间不稳定性"。

输出：experiments/decorrelation_selectivity.json + figs_final/figS9_decorrelation_selectivity.{png,pdf}
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

CLASSES = ["Forest", "Settlement", "Agriculture"]
CTRL = ["abs_lat", "elev", "cloud"]


def load():
    d = json.load(open(os.path.join(EXP, "stratified_floor_ws.json"), encoding="utf-8"))
    clim = {json.loads(l)["cov"]: json.loads(l)
            for l in open(os.path.join(EXP, "climate_independent.jsonl"), encoding="utf-8")}
    rows = []
    for s in d["scenes"]:
        c = clim.get(s["cov"])
        if not c or s.get("cloud") is None or s.get("delta_days") is None:
            continue
        rows.append(dict(cov=s["cov"], df=float(s["delta_floor"]), IPCC=s.get("IPCC") or "NA",
                         cloud=float(s["cloud"]), lag=abs(float(s["delta_days"])),
                         abs_lat=abs(float(s["lat"])), elev=c["elevation"]))
    return rows


def std_reg(X, y):
    """标准化 OLS + t 检验。返回 (r2, beta, t, p)。"""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    Xd = np.column_stack([np.ones(len(Xz)), Xz])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    res = y - Xd @ b
    ss = float((res ** 2).sum())
    dof = len(y) - Xd.shape[1]
    s2 = ss / dof
    se = np.sqrt(s2 * np.linalg.inv(Xd.T @ Xd).diagonal())
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    r2 = 1 - ss / float(((y - y.mean()) ** 2).sum())
    return float(r2), b[1:], t[1:], p[1:]


def main():
    rows = load()
    print("n =", len(rows))
    out = {"n": len(rows), "by_class": {}}

    for cls in CLASSES:
        sub = [r for r in rows if r["IPCC"] == cls]
        if len(sub) < 80:
            continue
        y = np.array([r["df"] for r in sub], float)
        X = np.array([[r[c] for c in CTRL] + [np.log10(1 + r["lag"])] for r in sub], float)
        r2, b, t, p = std_reg(X, y)
        # 同样的模型但把 lag 换成 cloud，用于比较两条通道的相对强度
        X2 = np.array([[r[c] for c in ["abs_lat", "elev"]] + [np.log10(1 + r["lag"]), r["cloud"]]
                       for r in sub], float)
        names = ["abs_lat", "elev", "log10(1+lag)", "cloud"]
        rec = {"n": len(sub), "r2": float(r2)}
        for nm, bb, tt, pp in zip(names, b, t, p):
            rec[nm] = {"beta": float(bb), "t": float(tt), "p": float(pp)}
        # 分箱
        lg = np.array([r["lag"] for r in sub], float)
        qs = np.quantile(lg, np.linspace(0, 1, 6))
        bins = []
        for i in range(5):
            m = (lg >= qs[i]) & (lg <= qs[i + 1])
            if m.sum() >= 20:
                bins.append({"lo": float(qs[i]), "hi": float(qs[i + 1]), "n": int(m.sum()),
                             "mean_df": float(y[m].mean()), "lag_med": float(np.median(lg[m]))})
        rec["lag_bins"] = bins
        # |lat| 三分位内部复查
        la = np.array([r["abs_lat"] for r in sub], float)
        qt = np.quantile(la, [0, 1 / 3, 2 / 3, 1.0])
        terc = []
        for k in range(3):
            m = (la >= qt[k]) & (la <= qt[k + 1])
            if m.sum() >= 40:
                rr, pp = stats.pearsonr(lg[m], y[m])
                terc.append({"lo": float(qt[k]), "hi": float(qt[k + 1]), "n": int(m.sum()),
                             "r": float(rr), "p": float(pp)})
        rec["lat_tercile"] = terc
        out["by_class"][cls] = rec
        print("\n[%s] n=%d  R2=%.3f" % (cls, len(sub), r2))
        for nm, bb, tt, pp in zip(names, b, t, p):
            print("    %-15s beta=%+.3f  t=%+.2f  p=%.3g" % (nm, bb, tt, pp))
        print("    lag bins:", ["%.0f" % x["mean_df"] for x in bins])
        print("    lat terciles r:", ["%.3f" % x["r"] for x in terc])

    json.dump(out, open(os.path.join(EXP, "decorrelation_selectivity.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[ok] wrote decorrelation_selectivity.json")
    make_fig(out)
    return out


def make_fig(out):
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4))
    COL = {"Forest": "#2f7d4f", "Settlement": "#8a8a8a", "Agriculture": "#c9a227"}

    # (a) 类内 Δ_floor 随 |采集间隔| 的轨迹
    ax = axes[0]
    for cls in CLASSES:
        r = out["by_class"].get(cls)
        if not r:
            continue
        xs = [b["lag_med"] for b in r["lag_bins"]]
        ys = [b["mean_df"] for b in r["lag_bins"]]
        ax.plot(xs, ys, "o-", color=COL[cls], ms=3.4, lw=1.3, label=cls)
    ax.set_xscale("log")
    ax.axhline(0, color="#bbb", lw=0.8, ls="--")
    ax.set_xlabel(u"|LR\u2013HR acquisition gap|  (days)")
    ax.set_ylabel(u"\u0394_floor (dB)")
    ax.set_title("(a) within-class penalty vs temporal gap", fontsize=7.5)
    ax.legend(fontsize=6.2, frameon=False)

    # (b) 相对惩罚 = 最滞后箱均值 / 最即时刻箱均值
    ax = axes[1]
    names = [c for c in CLASSES if out["by_class"].get(c)]
    ratio = []
    for c in names:
        b = out["by_class"][c]["lag_bins"]
        ratio.append(b[-1]["mean_df"] / b[0]["mean_df"])
    bars = ax.bar(range(len(names)), ratio, width=0.55,
                  color=[COL[c] for c in names])
    ax.axhline(1.0, color="#888", lw=0.9, ls="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=6.5)
    ax.set_ylabel(u"relative information penalty")
    ax.set_title("(b) penalty scales with canopy dynamics", fontsize=7.5)
    for x, v in zip(range(len(names)), ratio):
        ax.text(x, v + 0.03, u"\u00d7%.2f" % v, ha="center", fontsize=6.4, color="#333")
    ax.set_ylim(0, max(ratio) * 1.22)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figS9_decorrelation_selectivity.png"), dpi=600)
    fig.savefig(os.path.join(FIGS, "figS9_decorrelation_selectivity.pdf"))
    print("[ok] wrote figS9_decorrelation_selectivity.{png,pdf}")


if __name__ == "__main__":
    main()
