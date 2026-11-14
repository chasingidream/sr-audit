# -*- coding: utf-8 -*-
"""Path A 深挖诊断 —— 判定「活地面假说被证伪」是真修正还是统计假象。

三个必须回答的问题
  Q1 联合模型：把全部候选调制量放进同一回归，谁胜出？(|Δt|×amp 是否被清除)
  Q2 饱和假象：texture 高（Δ_floor 本就很低）时 |Δt| 斜率变平，是机制还是 ceiling？
  Q3 类内符号反转：Forest(负) vs Agriculture(正) 的 |Δt|×amp 是否真实、可否归因？

输出 experiments/canopy_diagnostics.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF = os.path.join(BASE, "experiments", "stratified_floor_ws.json")
CD = os.path.join(BASE, "experiments", "canopy_dynamics_lite.jsonl")
RQ = os.path.join(BASE, "experiments", "recon_quality.jsonl")
OUT = os.path.join(BASE, "experiments", "canopy_diagnostics.json")


def zs(x):
    x = np.asarray(x, float)
    s = x.std(ddof=0)
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def ols(X, y, names):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    dof = max(n - k, 1)
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(XtXi) * s2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 1e-15, beta / se, 0.0)
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return {nm: {"beta": float(beta[i]), "se": float(se[i]), "t": float(t[i]),
                 "p": float(p[i])} for i, nm in enumerate(names)} | \
        {"_n": int(n), "_r2": r2}


def load():
    sf = {s["cov"]: s for s in json.load(open(SF, encoding="utf-8"))["scenes"]}
    cd = {}
    for l in open(CD, encoding="utf-8"):
        d = json.loads(l)
        cd[d["cov"]] = d
    rq = {}
    if os.path.exists(RQ):
        for l in open(RQ, encoding="utf-8"):
            d = json.loads(l)
            rq[d["cov"]] = d
    rows = []
    for cov, c in cd.items():
        s = sf.get(cov)
        r = rq.get(cov)
        if not s or s.get("delta_floor") is None or s.get("delta_days") is None:
            continue
        rec = {
            "cov": cov, "y": float(s["delta_floor"]),
            "t": abs(float(s["delta_days"])), "amp": float(c["ndvi_amp"]),
            "iqr": float(c["ndvi_iqr"]),
            "tex": float(s["texture_hr"]), "sphet": float(s["spec_het"]),
            "ndsp": float(s["ndvi_std"]), "n1": float(s["ndvi_mean"]),
            "cloud": float(s["cloud"]), "lat": float(abs(s["lat"] or 0.0)),
            "bright_hr": float(s["bright_hr"]),
            "IPCC": s.get("IPCC"),
        }
        if r:
            rec["psnr_bic"] = float(r["psnr_bic"])
            rec["psnr_mean"] = float(r["psnr_mean"])
            rec["corr_bic_hr"] = float(r["corr_bic_hr"])
        else:
            rec["psnr_bic"] = rec["psnr_mean"] = rec["corr_bic_hr"] = np.nan
        rows.append(rec)
    return rows


def main():
    rows = load()
    A = {k: np.array([r[k] for r in rows], float) for k in
         ("y", "t", "amp", "iqr", "tex", "sphet", "ndsp", "n1", "cloud", "lat",
          "bright_hr", "psnr_bic", "psnr_mean", "corr_bic_hr")}
    y = A["y"]
    print("[n] %d" % len(rows))
    out = {"n": len(rows)}

    # ================= Q1 联合模型 =================
    print("\n=== Q1 联合交互模型（标准化，全部 |Δt|×m 同时入模）===")
    cand = [("amp", A["amp"]), ("texture_hr", A["tex"]), ("spec_het", A["sphet"]),
            ("ndvi_std_spatial", A["ndsp"]), ("cloud", A["cloud"]),
            ("mean_greenness", A["n1"]), ("abs_lat", A["lat"]),
            ("bright_hr", A["bright_hr"])]
    cols = [np.ones(len(y)), zs(A["t"])]
    names = ["const", "abs_dt"]
    for nm, v in cand:
        cols.append(zs(v)); names.append(nm)
    for nm, v in cand:
        cols.append(zs(A["t"]) * zs(v)); names.append("dt_x_" + nm)
    r = ols(np.column_stack(cols), zs(y), names)
    out["joint"] = r
    print("  R2 = %.4f" % r["_r2"])
    for nm, _ in cand:
        d = r["dt_x_" + nm]
        star = "***" if d["p"] < 1e-3 else ("**" if d["p"] < 0.01 else ("*" if d["p"] < 0.05 else ""))
        print("    %-18s beta=%+.4f  se=%.4f  p=%.2g %s" % (nm, d["beta"], d["se"], d["p"], star))

    # ================= Q2 饱和（ceiling）检查 =================
    print("\n=== Q2 饱和检查：按 texture 四分位，Δ_floor 与 |Δt| 斜率 ===")
    q = np.percentile(A["tex"], [25, 50, 75])
    out["tex_quartile_slopes"] = []
    print("  %-10s %5s %9s %9s %22s %7s" % ("quartile", "n", "texture", "mean_y", "slope(y~|dt|)", "sat"))
    for k, (lo, hi) in enumerate([(-np.inf, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], np.inf)]):
        sel = (A["tex"] >= lo) & (A["tex"] < hi)
        if sel.sum() < 30:
            continue
        lr = stats.linregress(A["t"][sel], y[sel])
        # 饱和指标：该组 |Δt| 低分位与高分位的 mean_y 差
        tl = np.median(A["t"][sel])
        dlo = y[sel & (A["t"] <= tl)].mean()
        dhi = y[sel & (A["t"] > tl)].mean()
        out["tex_quartile_slopes"].append({
            "q": k, "n": int(sel.sum()), "tex_med": float(np.median(A["tex"][sel])),
            "mean_y": float(y[sel].mean()), "slope": float(lr.slope),
            "slope_p": float(lr.pvalue), "delta_lohi": float(dhi - dlo)})
        print("  Q%-9d %5d %9.4f %9.3f %22s %7.3f"
              % (k + 1, sel.sum(), np.median(A["tex"][sel]), y[sel].mean(),
                 "%+.5f (p=%.1g)" % (lr.slope, lr.pvalue), dhi - dlo))

    # ================= Q3 类内符号反转 =================
    print("\n=== Q3 类内 |Δt|×amp（含 ndvi 水平与空间异质性控制）===")
    out["by_class"] = {}
    for cl in sorted(set(r["IPCC"] for r in rows if r["IPCC"])):
        sel = np.array([r["IPCC"] == cl for r in rows])
        if sel.sum() < 150:
            continue
        X = np.column_stack([np.ones(sel.sum()), zs(A["t"][sel]), zs(A["amp"][sel]),
                             zs(A["t"][sel]) * zs(A["amp"][sel]),
                             zs(A["ndsp"][sel]), zs(A["n1"][sel]), zs(A["lat"][sel])])
        nm = ["const", "abs_dt", "amp", "dt_x_amp", "ndvi_std_spatial",
              "mean_greenness", "abs_lat"]
        rr = ols(X, zs(y[sel]), nm)
        # 该类的 amp 内部四分位斜率
        qa = np.percentile(A["amp"][sel], [25, 75])
        sub_lo = sel & (A["amp"] <= qa[0])
        sub_hi = sel & (A["amp"] >= qa[1])
        s_lo = stats.linregress(A["t"][sub_lo], y[sub_lo])
        s_hi = stats.linregress(A["t"][sub_hi], y[sub_hi])
        rr["_slope_lowamp"] = float(s_lo.slope)
        rr["_slope_highamp"] = float(s_hi.slope)
        rr["_amp_med"] = float(np.median(A["amp"][sel]))
        rr["_tex_med"] = float(np.median(A["tex"][sel]))
        rr["_n"] = int(sel.sum())
        out["by_class"][cl] = rr
        print("  %-12s n=%4d amp_med=%.3f tex_med=%.4f | dt_x_amp=%+.3f (p=%.2g) | "
              "slope lowamp %+.5f -> highamp %+.5f"
              % (cl, sel.sum(), rr["_amp_med"], rr["_tex_med"], rr["dt_x_amp"]["beta"],
                 rr["dt_x_amp"]["p"], s_lo.slope, s_hi.slope))

    # ================= Q3b 反转是否源于 texture 差异 =================
    print("\n=== Q3b 森林 vs 农业：是否在 texture 上不可比？ ===")
    for cl in ("Forest", "Agriculture", "Settlement"):
        sel = np.array([r["IPCC"] == cl for r in rows])
        if sel.sum() < 50:
            continue
        print("  %-12s texture p25/med/p75 = %.4f / %.4f / %.4f   "
              "amp p25/med/p75 = %.3f / %.3f / %.3f   cloud med = %.2f"
              % (cl, np.percentile(A["tex"][sel], 25), np.median(A["tex"][sel]),
                 np.percentile(A["tex"][sel], 75),
                 np.percentile(A["amp"][sel], 25), np.median(A["amp"][sel]),
                 np.percentile(A["amp"][sel], 75), np.median(A["cloud"][sel])))

    # 类内 texture 匹配后 amp 交互（Forest vs Agriculture 在共同 texture 区间）
    lo = np.percentile(A["tex"], 40); hi = np.percentile(A["tex"], 60)
    band = (A["tex"] >= lo) & (A["tex"] <= hi)
    print("\n  texture 匹配带 [%.4f, %.4f] (n=%d)：" % (lo, hi, band.sum()))
    out["tex_matched"] = {}
    for cl in ("Forest", "Agriculture", "Settlement"):
        sel = band & np.array([r["IPCC"] == cl for r in rows])
        if sel.sum() < 60:
            print("    %-12s n=%d  -- 样本不足" % (cl, sel.sum()))
            continue
        X = np.column_stack([np.ones(sel.sum()), zs(A["t"][sel]), zs(A["amp"][sel]),
                             zs(A["t"][sel]) * zs(A["amp"][sel]), zs(A["n1"][sel])])
        rr = ols(X, zs(y[sel]), ["const", "abs_dt", "amp", "dt_x_amp", "mean_greenness"])
        out["tex_matched"][cl] = {k: v for k, v in rr.items() if not k.startswith("_")} | \
            {"_n": int(sel.sum())}
        print("    %-12s n=%4d  dt_x_amp=%+.3f (p=%.2g)" %
              (cl, sel.sum(), rr["dt_x_amp"]["beta"], rr["dt_x_amp"]["p"]))

    # ================= Q5 尺度无关复核 =================
    print("\n=== Q5 因变量敏感性：dB 尺度（Δ_floor / PSNR）vs 尺度无关（corr）===")
    dvs = [("delta_floor", A["y"]), ("psnr_bic", A["psnr_bic"]),
           ("corr_bic_hr", A["corr_bic_hr"])]
    out["dv_sensitivity"] = {}
    print("  %-14s %-4s %7s %10s %11s %10s" % ("dep_var", "grp", "n", "mean", "slope", "p"))
    for nm, dv in dvs:
        ok = np.isfinite(dv)
        entry = {"by_texture_quartile": []}
        for k, (lo, hi) in enumerate([(-np.inf, q[0]), (q[0], q[1]),
                                      (q[1], q[2]), (q[2], np.inf)]):
            sel = ok & (A["tex"] >= lo) & (A["tex"] < hi)
            if sel.sum() < 30:
                continue
            lr = stats.linregress(A["t"][sel], dv[sel])
            entry["by_texture_quartile"].append(
                {"q": k + 1, "n": int(sel.sum()), "mean": float(dv[sel].mean()),
                 "slope": float(lr.slope), "p": float(lr.pvalue)})
            print("  %-14s Q%-3d %7d %10.4f %11.5f %10.2g"
                  % (nm, k + 1, sel.sum(), dv[sel].mean(), lr.slope, lr.pvalue))
        cols = [np.ones(ok.sum()), zs(A["t"][ok]), zs(A["amp"][ok]), zs(A["tex"][ok]),
                zs(A["t"][ok]) * zs(A["amp"][ok]), zs(A["t"][ok]) * zs(A["tex"][ok]),
                zs(A["ndsp"][ok]), zs(A["n1"][ok])]
        names = ["const", "abs_dt", "amp", "tex", "dt_x_amp", "dt_x_tex",
                 "ndvi_std_spatial", "mean_greenness"]
        rr = ols(np.column_stack(cols), zs(dv[ok]), names)
        entry["joint_dt_x_amp"] = {k: rr["dt_x_amp"][k] for k in ("beta", "p")}
        entry["joint_dt_x_tex"] = {k: rr["dt_x_tex"][k] for k in ("beta", "p")}
        entry["r2"] = rr["_r2"]
        out["dv_sensitivity"][nm] = entry
        print("       joint: dt_x_amp=%+.4f (p=%.2g)  dt_x_tex=%+.4f (p=%.2g)  R2=%.3f"
              % (rr["dt_x_amp"]["beta"], rr["dt_x_amp"]["p"],
                 rr["dt_x_tex"]["beta"], rr["dt_x_tex"]["p"], rr["_r2"]))

    # ================= Q6 站点固定效应 =================
    #   ⚠ |Δt| 与站点近乎共线（Landcover/UNHCR 中位 ~5 d vs ASMSpotter ~309 d），
    #     且 ASMSpotter 的 texture 低 3 倍 ⇒ 全局 |Δt| 效应与 "texture 效应" 都可能是站点效应。
    print("\n=== Q6 站点固定效应（消除站点混杂）===")
    sites = sorted(set(r["cov"].split("-")[0] for r in rows))
    sid = np.array([sites.index(r["cov"].split("-")[0]) for r in rows])
    D = np.column_stack([(sid == k).astype(float) for k in range(1, len(sites))])
    out["sites"] = sites
    out["site_fe"] = {}
    for nm, dv in [("delta_floor", A["y"]), ("psnr_bic", A["psnr_bic"]),
                   ("corr_bic_hr", A["corr_bic_hr"])]:
        ok = np.isfinite(dv)
        cols = [np.ones(ok.sum()), zs(A["t"][ok]), zs(A["amp"][ok]),
                zs(A["t"][ok]) * zs(A["amp"][ok]), zs(A["ndsp"][ok]),
                zs(A["n1"][ok]), zs(A["tex"][ok])]
        names = ["const", "abs_dt", "amp", "dt_x_amp", "ndvi_std_spatial",
                 "mean_greenness", "texture_hr"]
        cols.append(D[ok])
        names += ["site%d" % k for k in range(1, len(sites))]
        rr = ols(np.column_stack(cols), zs(dv[ok]), names)
        out["site_fe"][nm] = {k: rr[k] for k in ("abs_dt", "dt_x_amp", "texture_hr")}
        out["site_fe"][nm]["_r2"] = rr["_r2"]
        print("  %-12s abs_dt=%+.4f (p=%.2g)  dt_x_amp=%+.4f (p=%.2g)  "
              "texture=%+.4f (p=%.2g)  R2=%.3f"
              % (nm, rr["abs_dt"]["beta"], rr["abs_dt"]["p"],
                 rr["dt_x_amp"]["beta"], rr["dt_x_amp"]["p"],
                 rr["texture_hr"]["beta"], rr["texture_hr"]["p"], rr["_r2"]))

    # ================= Q6b 纯站内（within）分层 =================
    print("\n  纯站内：各站点内 |Δt| 高/低半 与 amp 高/低半 的 Δ_floor 差（between-site 已消）")
    out["within_site_grid"] = {}
    for st in sites:
        sel = sid == sites.index(st)
        if sel.sum() < 120:
            continue
        tm = np.median(A["t"][sel]); am = np.median(A["amp"][sel])
        g = {}
        for tn, tl in [("lo", A["t"][sel] <= tm), ("hi", A["t"][sel] > tm)]:
            for an, al in [("lo", A["amp"][sel] <= am), ("hi", A["amp"][sel] > am)]:
                m = tl & al
                g["%s_%s" % (tn, an)] = float(A["y"][sel][m].mean()) if m.sum() > 5 else np.nan
                g["n_%s_%s" % (tn, an)] = int(m.sum())
        # 高amp−低amp 的 |Δt| 斜率差（站内）
        s_lo = stats.linregress(A["t"][sel & (A["amp"] <= am)],
                                A["y"][sel & (A["amp"] <= am)])
        s_hi = stats.linregress(A["t"][sel & (A["amp"] > am)],
                                A["y"][sel & (A["amp"] > am)])
        g["slope_lowamp"] = float(s_lo.slope)
        g["slope_highamp"] = float(s_hi.slope)
        g["n"] = int(sel.sum())
        out["within_site_grid"][st] = g
        print("    %-12s n=%4d | slope lowamp %+.5f  highamp %+.5f  Δ=%+.5f"
              % (st, sel.sum(), s_lo.slope, s_hi.slope, s_hi.slope - s_lo.slope))

    # ================= Q4 预注册口径汇总 =================
    print("\n=== Q4 汇总 ===")
    verdict = {
        "H2_supported": bool(out["joint"]["dt_x_amp"]["p"] < 0.05 and
                             out["joint"]["dt_x_amp"]["beta"] < 0),
        "joint_dt_x_amp_beta": out["joint"]["dt_x_amp"]["beta"],
        "joint_dt_x_amp_p": out["joint"]["dt_x_amp"]["p"],
        "strongest_moderator": max(
            [(nm, r["dt_x_" + nm]["beta"], r["dt_x_" + nm]["p"]) for nm, _ in cand
             if nm != "amp"], key=lambda z: abs(z[1])),
    }
    out["verdict"] = verdict
    print("  H2（活地面假说）成立？ %s" % verdict["H2_supported"])
    print("  联合模型中 |Δt|×amp: beta=%+.4f p=%.3g" %
          (verdict["joint_dt_x_amp_beta"], verdict["joint_dt_x_amp_p"]))
    print("  最强调制量: %s  beta=%+.4f p=%.2g" % verdict["strongest_moderator"])

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[saved] %s" % OUT)


if __name__ == "__main__":
    main()
