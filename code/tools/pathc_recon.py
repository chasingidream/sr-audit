# -*- coding: utf-8 -*-
"""Path C：可观测性预算（Δ_floor）的可预测性 —— 特征重要性 + 站点 FE + 诚实 CV。

数据（零训练、零下载，全部读既有 JSON/CSV）
  experiments/stratified_floor_ws.json  3,927 景：delta_floor + 6 物理预测子
                                        + lat/lon/IPCC/SMOD/cloud/delta_days + cov
  experiments/recon_quality.jsonl       同 3,927 景：psnr_bic/psnr_mean/corr_bic_hr
                                        /var_hr_n/contrast_lr
  data/raw/metadata.csv                 WorldStrat 官方人口级元数据（n=62,848）：
                                        area(Km²) / LCCS(连续码) / bounds

站点 = cov.rsplit("-", 3)[0] ∈ {ASMSpotter, Amnesty POI, Landcover, UNHCR}
      （实测 3,927 景 → 4 个 campaign；ASMSpotter 350 / Amnesty POI 189 / …）

因变量两口径
  y1 = delta_floor   （dB，含 PSNR(mean) 基线项 —— 与 texture 有定义层耦合）
  y2 = corr_bic_hr   （尺度无关，Path A 教训：跨场景回归应以它复核）

设计
  * OLS 样本内 R²/adj（用 t 统计量做"有效自由度"，避免 p≥n 的虚假拟合）
  * 空间留块 CV（Δlat×Δlon 四档）
  * 留站点 CV（leave-one-campaign-out）—— 全球制图的实际泛化口径
  * 分组 df 修正 R²（仅算组间解释力）
"""
import csv
import json
import os
import re

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "experiments")
META = os.path.join(ROOT, "data", "raw", "metadata.csv")


CAMP_RE = re.compile(r"^(ASMSpotter|Amnesty POI|Landcover|UNHCR)")


def campaign(cov):
    """WorldStrat 采集战役（4 个），非 scene 级子串。

    ⚠ 不能用 rsplit("-", 3)：Landcover/Amnesty 的 scene id 与目录名不同构，
    会把 3,390 个景各切成一个"组"（留组 CV 退化成 3,390 折）。
    """
    m = CAMP_RE.match(str(cov).strip())
    return m.group(1) if m else "OTHER"


def load_meta():
    """官方 metadata：area(Km²) / LCCS 连续码 / bounds → lat/lon 跨度。"""
    out = {}
    with open(META, "r", encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        idx = {n: i for i, n in enumerate(hdr)}
        for row in rd:
            if not row:
                continue
            sid = row[0].strip()
            if not sid or sid in out:
                continue
            rec = {}
            try:
                rec["area_km2"] = float(row[idx["area"]])
            except (KeyError, ValueError, IndexError):
                pass
            try:
                rec["lccs"] = float(row[idx["LCCS"]])
            except (KeyError, ValueError, IndexError):
                pass
            try:
                b = json.loads(row[idx["bounds"]])
                rec["lon_span"] = abs(b[2] - b[0])
                rec["lat_span"] = abs(b[3] - b[1])
            except (KeyError, ValueError, IndexError, json.JSONDecodeError):
                pass
            out[sid] = rec
    return out


def load():
    q = {}
    for line in open(os.path.join(EXP, "recon_quality.jsonl"), encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            q[r["cov"]] = r
    d = json.load(open(os.path.join(EXP, "stratified_floor_ws.json"), encoding="utf-8"))
    meta = load_meta()
    rows = []
    for s in d["scenes"]:
        cov = s["cov"]
        r = dict(s)
        r["campaign"] = campaign(cov)
        qq = q.get(cov)
        if qq:
            r.update({k: qq[k] for k in
                      ("psnr_bic", "psnr_mean", "var_hr_n", "corr_bic_hr", "contrast_lr")})
        m = meta.get(cov)
        if m:
            r.update(m)
        rows.append(r)
    return rows


def feat_vector(r, name):
    """由原始字段派生特征；返回 None 表示缺失。"""
    if name == "log_area":
        v = r.get("area_km2")
        return None if v is None else np.log10(max(v, 1e-3))
    if name == "log_lccs":
        v = r.get("lccs")
        return None if v is None else np.log10(max(v, 1.0))
    if name == "log_tex":
        v = r.get("texture_hr")
        return None if v is None else np.log10(max(v, 1e-12))
    if name == "log_var_hr":
        v = r.get("var_hr_n")
        return None if v is None or v <= 0 else np.log10(v)
    if name == "log_contrast":
        v = r.get("contrast_lr")
        return None if v is None or v <= 0 else np.log10(v)
    if name == "log_psnr_mean":
        v = r.get("psnr_mean")
        return None if v is None or v <= 0 else np.log10(v)
    if name == "abs_dt_log":
        v = r.get("delta_days")
        return None if v is None else np.log10(abs(v) + 1.0)
    if name == "abs_lat":
        v = r.get("lat")
        return None if v is None else abs(float(v))
    v = r.get(name)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def design(rows, feats, cats=None):
    X, ok = [], []
    for r in rows:
        vals = [feat_vector(r, f) for f in feats]
        if any(v is None for v in vals):
            ok.append(False)
            X.append([np.nan] * len(feats))
        else:
            ok.append(True)
            X.append(vals)
    X = np.array(X, float)
    if cats is not None:
        levels = sorted({r[cats] for r in rows})
        D = np.zeros((len(rows), len(levels) - 1))
        for i, r in enumerate(rows):
            for j, lv in enumerate(levels[1:]):
                D[i, j] = 1.0 if r[cats] == lv else 0.0
        X = np.column_stack([X, D])
    return X, np.array(ok)


def ols(X, y):
    n, p = X.shape
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xd = np.column_stack([np.ones(n), (X - mu) / sd])
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ b
    resid = y - pred
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / sst
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    # 有效自由度（t 统计量）：粗但稳健的正则化指标
    try:
        sigma2 = (resid ** 2).sum() / max(n - p - 1, 1)
        cov_b = sigma2 * np.linalg.pinv(Xd.T @ Xd)
        t = np.abs(b / np.sqrt(np.maximum(np.diag(cov_b), 1e-30)))
        df_eff = int(min(p, np.sum(t > 2.0)))
    except np.linalg.LinAlgError:
        df_eff = p
    return float(r2), float(adj), df_eff, pred


def cv_blocks(rows, X, y, dlat, dlon, min_tr=20, min_te=1):
    keys = [(int((r["lat"] + 90) // dlat), int((r["lon"] + 180) // dlon)) for r in rows]
    pred = np.full(len(rows), np.nan)
    for k in sorted(set(keys)):
        te = np.array([i for i, kk in enumerate(keys) if kk == k])
        tr = np.array([i for i, kk in enumerate(keys) if kk != k])
        ok = ~np.isnan(X).any(axis=1)
        tr, te = tr[ok[tr]], te[ok[te]]
        if len(tr) < min_tr or len(te) < min_te:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        b, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd]), y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(len(te)), (X[te] - mu) / sd]) @ b
    m = ~np.isnan(pred)
    if m.sum() < 30:
        return float("nan"), int(m.sum())
    r2 = 1 - ((y[m] - pred[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    return float(r2), int(m.sum())


def cv_leave_group(rows, X, y, key="campaign"):
    pred = np.full(len(rows), np.nan)
    lv = sorted({r[key] for r in rows})
    per = {}
    for g in lv:
        te = np.array([i for i, r in enumerate(rows) if r[key] == g])
        tr = np.array([i for i, r in enumerate(rows) if r[key] != g])
        ok = ~np.isnan(X).any(axis=1)
        tr, te = tr[ok[tr]], te[ok[te]]
        if len(tr) < 20 or len(te) == 0:
            per[g] = float("nan")
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        b, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd]), y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(len(te)), (X[te] - mu) / sd]) @ b
        per[g] = float(1 - ((y[te] - pred[te]) ** 2).sum() /
                       ((y[te] - y[te].mean()) ** 2).sum())
    m = ~np.isnan(pred)
    if m.sum() < 30:
        return float("nan"), int(m.sum()), per
    r2 = 1 - ((y[m] - pred[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    return float(r2), int(m.sum()), per


IMG = ["ndvi_mean", "ndvi_std", "bright_lr", "spec_het", "bright_hr", "texture_hr"]
ENV = ["abs_lat", "cloud", "abs_dt_log"]          # 观测状态（Path A 的 2 观测变量 + lat）
GEO = ["log_area", "log_lccs", "lon_span", "lat_span"]
SCALE = ["log_var_hr", "log_contrast", "log_psnr_mean"]   # 尺度陷阱组

FEATSETS = [
    ("image6 (Path A 原版)", IMG),
    ("image6+log_tex", IMG + ["log_tex"]),
    ("env3", ENV),
    ("geo4", GEO),
    ("image+env+geo", IMG + ENV + GEO),
    ("image+env+geo+scale", IMG + ENV + GEO + SCALE),
]


def main():
    rows = load()
    print(f"[load] n={len(rows)}")
    camps = {}
    for r in rows:
        camps.setdefault(r["campaign"], []).append(r)
    print("[campaign] " + "  ".join(f"{k}={len(v)}" for k, v in
                                    sorted(camps.items(), key=lambda x: -len(x[1]))))
    for f in ["area_km2", "lccs", "lat_span", "corr_bic_hr", "var_hr_n"]:
        n = sum(1 for r in rows if r.get(f) is not None)
        print(f"  cov {f:<14s} {n}/{len(rows)}")

    ys = {
        "delta_floor": np.array([r["delta_floor"] for r in rows], float),
        "corr_bic_hr": np.array([r.get("corr_bic_hr", np.nan) for r in rows], float),
    }

    report = {"n": len(rows),
              "campaigns": {k: len(v) for k, v in camps.items()},
              "featsets": {}}

    for yname, y in ys.items():
        print("\n" + "=" * 92)
        print(f"### y = {yname}   n_valid={np.isfinite(y).sum()}")
        print(f"{'feature set':<26s} {'p':>3s} {'n':>5s} {'in-R2':>7s} {'adj':>7s} "
              f"{'dfEff':>5s} {'CV5x5':>7s} {'CV10':>7s} {'CV20':>7s} {'CV30x60':>8s} "
              f"{'leave-campaign':>14s}")
        res = {}
        for name, feats in FEATSETS:
            X, ok = design(rows, feats)
            if X.shape[1] > len(rows):
                continue
            m = ok & np.isfinite(y)
            Xo, yo = X[m], y[m]
            if len(yo) < 50:
                continue
            r2, adj, dfe, _ = ols(Xo, yo)
            cvs = {f"{a}x{b}": cv_blocks(rows, X, y, a, b)[0]
                   for a, b in [(5, 5), (10, 10), (20, 20), (30, 60)]}
            lc, lcn, per = cv_leave_group(rows, X, y)
            Xf, okf = design(rows, feats, cats="campaign")
            mf = okf & np.isfinite(y)
            r2fe = ols(Xf[mf], y[mf])[0] if mf.sum() > 50 else np.nan
            print(f"{name:<26s} {len(feats):3d} {len(yo):5d} {r2:7.3f} {adj:7.3f} "
                  f"{dfe:5d} {cvs['5x5']:7.3f} {cvs['10x10']:7.3f} {cvs['20x20']:7.3f} "
                  f"{cvs['30x60']:8.3f} {lc:14.3f}")
            res[name] = {"p": len(feats), "n": int(len(yo)), "r2": r2, "adj_r2": adj,
                         "df_eff": dfe, "cv": cvs, "cv_leave_campaign": lc,
                         "r2_with_campaign_fe": r2fe,
                         "per_campaign_cv_r2": per}
        report["featsets"][yname] = res

    json.dump(report, open(os.path.join(EXP, "pathc_recon.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n[ok] wrote pathc_recon.json")


if __name__ == "__main__":
    main()
