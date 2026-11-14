#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主张 A —— Δ_floor 的地表条件分层（stratified recoverable-information budget）。

目的（回应 CEE 审稿人视角复评的 MR-1）：把 Δ_floor 从"数据属性"升级为
"**地表条件的函数**"——在哪些土地覆盖 / 哪一季 / 云量多少下，跨传感器可恢复预算先崩塌。

零训练：只计算 data-layer 的 Δ_floor = PSNR(bicubic(LR),GT) − PSNR(mean(GT),GT)，
逐场景取值后按 WorldStrat 元数据（LCCS / IPCC Class / SMOD / 月份 / 云量）分层。

口径与主文 `code/audit_axis.py` 完全一致：
  * per-scene 归一化（LR 侧 (mu,sd) 同时标准化 HR，无 clamp）；但 Δ_floor 对**全局**
    线性重标不变，故分层均值不依赖该约定（见 Methods §Invariance）。
  * bicubic align_corners=False（与 sr_audit.bicubic_up 一致）。
  * 逐样本 PSNR 平均 + 配对 t 检验。

用法：
  python code/stratify_floor.py --extract          # 抽特征（慢，~半小时，需 GPU? 否，纯 CPU/np）
  python code/stratify_floor.py --analyze          # 分层统计 + 出图
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import numpy as np  # noqa: E402

WS_ROOT = os.environ.get("WORLDSTRAT_ROOT") or os.path.expanduser(os.path.join("~", "worldstrat"))
META_CSV = os.path.join(BASE, "data", "raw", "metadata.csv")
OUT_JSON = os.path.join(BASE, "experiments", "stratified_floor_ws.json")
FIG_DIR = os.path.join(BASE, "experiments", "figs_final")

# 与主文外部配置一致
LR_SIZE = 64
HR_OUT = 424          # 6.63× 时 HR 侧窗口（64*6.63≈424）
T = 5
DIV_HR = 10000.0      # SPOT _ps.tiff: DN = 反射率×10000


# --------------------------------------------------------------------- I/O
def read_tif(path, bands=None, div=None):
    import tifffile
    arr = tifffile.imread(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1] // 2:
            pass
        else:
            arr = np.transpose(arr, (2, 0, 1))
    if bands is not None:
        # ⚠ 必须用 list（fancy indexing）而非 tuple：tuple 会被 numpy 当作
        # 逐轴整数索引，直接把 (C,H,W) 索引成标量。
        arr = arr[list(bands)]
    if div is not None:
        arr = arr / float(div)
    elif arr.max() > 1.0:
        arr = arr / 65535.0
    return np.clip(arr, 0.0, 1.0)


def bicubic_np(img, out_hw):
    """numpy 双三次上采样（与 torch F.interpolate align_corners=False 数值近似）。
    为避免依赖 torch（纯 CPU 环境更快），用 cv2 优先，退化到 PIL。"""
    import cv2
    c, h, w = img.shape
    out = np.empty((c, out_hw[0], out_hw[1]), dtype=np.float32)
    for i in range(c):
        out[i] = cv2.resize(img[i], (out_hw[1], out_hw[0]),
                            interpolation=cv2.INTER_CUBIC)
    return out


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


# ------------------------------------------------------------------ manifest
def build_manifest():
    """与 dataset.build_manifest_worldstrat_zenodo 同口径：coverage 目录名 = scene id。"""
    hr_ps = glob.glob(os.path.join(WS_ROOT, "**", "*_ps.tiff"), recursive=True)
    hr_rgbn = glob.glob(os.path.join(WS_ROOT, "**", "*_rgbn.tiff"), recursive=True)
    lr_files = glob.glob(os.path.join(WS_ROOT, "**", "L2A", "*L2A_data.tiff"), recursive=True)

    def hr_cov(p):
        return os.path.basename(os.path.dirname(p))

    def lr_cov(p):
        return os.path.basename(os.path.dirname(os.path.dirname(p)))

    hr_by = {}
    for p in sorted(hr_ps):
        hr_by.setdefault(hr_cov(p), p)
    for p in sorted(hr_rgbn):
        hr_by.setdefault(hr_cov(p), p)
    lr_by = {}
    for p in sorted(lr_files):
        lr_by.setdefault(lr_cov(p), []).append(p)

    man = []
    for cov, hr in hr_by.items():
        lrs = lr_by.get(cov, [])
        if lrs:
            man.append({"cov": cov, "hr": hr, "lr": sorted(lrs)})
    return man


def load_metadata():
    """读取 WorldStrat 官方 metadata.csv。

    ⚠ 该 CSV 首行形如 `,n,bounds,...`——**首列名为空**，`csv.DictReader`
    会用空串 `''` 作 key，故不能用 `r["Unnamed: 0"]`。此处显式按位置取首列。
    """
    rows = {}
    with open(META_CSV, "r", encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        # 各列索引（按名定位，容忍首列空名）
        idx = {name: i for i, name in enumerate(header)}
        c_sid = 0                                   # 首列 = scene id（名称为空）
        c_lat = idx.get("lat")
        c_lon = idx.get("lon")
        c_lccs = idx.get("LCCS")
        c_ipcc = idx.get("IPCC Class")
        c_smod = idx.get("SMOD Class")
        c_cloud = idx.get("cloud_cover")
        c_hrd = idx.get("highres_date")
        c_lrd = idx.get("lowres_date")
        c_delta = idx.get("delta")
        for row in rd:
            if not row:
                continue
            sid = row[c_sid].strip()
            if not sid or sid in rows:
                continue                            # 同一 sid 多行，保留首行
            rows[sid] = {
                "lat": row[c_lat] if c_lat is not None else None,
                "lon": row[c_lon] if c_lon is not None else None,
                "LCCS": row[c_lccs] if c_lccs is not None else None,
                "IPCC": row[c_ipcc] if c_ipcc is not None else None,
                "SMOD": row[c_smod] if c_smod is not None else None,
                "cloud": row[c_cloud] if c_cloud is not None else None,
                "hr_date": row[c_hrd] if c_hrd is not None else None,
                "lr_date": row[c_lrd] if c_lrd is not None else None,
                "delta_days": row[c_delta] if c_delta is not None else None,
            }
    return rows


# ------------------------------------------------------------------- extract
def extract():
    man = build_manifest()
    meta = load_metadata()
    print(f"[info] manifest pairs={len(man)}  metadata ids={len(meta)}")
    out = []
    miss = 0
    for k, item in enumerate(man):
        cov = item["cov"]
        rec = {"cov": cov}
        try:
            hr_full = read_tif(item["hr"], bands=(0, 1, 2), div=DIV_HR)
            # 用 frame0 的 LR 尺寸定窗口（同 dataset._pair_box 逻辑）
            lr0 = read_tif(item["lr"][0], bands=(3, 2, 1))   # True-RGB indices (S2 bands 4,3,2)
            Hr, Wr = hr_full.shape[-2], hr_full.shape[-1]
            Hl, Wl = lr0.shape[-2], lr0.shape[-1]
            ratio = float(Hr) / float(Hl)                    # ≈ 6.63
            hr_size = min(HR_OUT, Hr, Wr)
            lr_win = int(round(hr_size / ratio))
            lr_win = max(2, min(lr_win, Hl, Wl))
            # 确定性窗口（居中，避免与训练切片相关）
            i = max(0, (Hl - lr_win) // 2)
            j = max(0, (Wl - lr_win) // 2)
            hi = min(max(0, int(round(i * ratio))), max(0, Hr - hr_size))
            hj = min(max(0, int(round(j * ratio))), max(0, Wr - hr_size))
            hr = hr_full[:, hi:hi + hr_size, hj:hj + hr_size]
            lr = lr0[:, i:i + lr_win, j:j + lr_win]

            # per-scene 归一化：以 LR 的 (mu,sd) 同变换 HR（与 dataset norm='per_scene' 同口径）
            mu = float(lr.mean())
            sd = float(lr.std()) + 1e-6
            hr_n = (hr - mu) / sd

            # per-sample Δ_floor
            bic = bicubic_np(lr, (hr_size, hr_size))
            mean_img = np.full_like(hr_n, hr_n.mean())
            d_floor = psnr(bic, hr_n) - psnr(mean_img, hr_n)

            rec["delta_floor"] = round(d_floor, 4)
            rec["n_lr_frames"] = len(item["lr"])

            # ---- 物理预测子（A′）：来自 LR 全 12 波段 + HR 的连续地表物理量 ----
            # S2 12 波段 = [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12]（无 B10）
            #   真彩 RGB = B4,B3,B2 = 索引 3,2,1；近红外 NIR = B8 = 索引 7。
            lr_full = read_tif(item["lr"][0])           # (12,H,W)，已是 0–1 反射率
            lrw = lr_full[:, i:i + lr_win, j:j + lr_win]
            red, nir = lrw[3], lrw[7]
            ndvi = (nir - red) / (nir + red + 1e-6)     # 植被/水分代理
            rec["ndvi_mean"] = round(float(ndvi.mean()), 4)
            rec["ndvi_std"] = round(float(ndvi.std()), 4)
            rec["bright_lr"] = round(float(lrw[[3, 2, 1]].mean()), 4)   # 可见光亮度
            rec["spec_het"] = round(float(lrw.std(axis=0).mean()), 4)   # 光谱异质性
            lum = hr.mean(axis=0)                        # HR 亮度（SPOT RGB 均值）
            gy_, gx_ = np.gradient(lum)
            rec["bright_hr"] = round(float(lum.mean()), 4)
            rec["texture_hr"] = round(float(np.sqrt(gx_ ** 2 + gy_ ** 2).mean()), 4)  # 纹理能量

            md = meta.get(cov)
            if md:
                rec["lat"] = float(md["lat"])
                rec["lon"] = float(md["lon"])
                rec["LCCS"] = int(float(md["LCCS"])) if md["LCCS"] else None
                rec["IPCC"] = md["IPCC"]
                rec["SMOD"] = md["SMOD"]
                rec["cloud"] = float(md["cloud"])
                rec["hr_date"] = md["hr_date"]
                rec["lr_date"] = md["lr_date"]
                rec["delta_days"] = int(float(md["delta_days"]))
            else:
                miss += 1
            out.append(rec)
        except Exception as e:
            print(f"[warn] {cov}: {e}")
        if (k + 1) % 100 == 0:
            print(f"  ..{k+1}/{len(man)}")
    json.dump({"n": len(out), "n_meta_missing": miss, "scenes": out},
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[saved] {OUT_JSON}  n={len(out)} (meta missing {miss})")


# ------------------------------------------------------------------- analyze
def analyze():
    d = json.load(open(OUT_JSON, encoding="utf-8"))
    sc = d["scenes"]
    vals = np.array([s["delta_floor"] for s in sc if s.get("delta_floor") is not None])
    print(f"[all] n={len(vals)}  mean Δ_floor = {vals.mean():+.3f} dB  "
          f"(std {vals.std():.3f})")
    print(f"      negative fraction = {(vals < 0).mean()*100:.1f}%")

    def report(key, label, min_n=8, stat=True):
        print(f"\n=== by {label} ({key}) ===")
        groups = {}
        for s in sc:
            if s.get("delta_floor") is None:
                continue
            g = s.get(key)
            if g is None:
                continue
            groups.setdefault(g, []).append(s["delta_floor"])
        rows = []
        for g, v in groups.items():
            v = np.array(v)
            if len(v) < min_n:
                continue
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            ci = 1.96 * se
            rows.append([g, int(len(v)), float(v.mean()), float(v.std(ddof=1)),
                         float((v < 0).mean()), float(ci), float(se)])
        rows.sort(key=lambda r: r[2])
        for g, n, m, sd, negf, ci, se in rows:
            print(f"  {str(g):<38} n={n:<5} Δ_floor={m:+.3f} ± {sd:.3f} "
                  f"(95%CI ±{ci:.2f})  neg={negf*100:.0f}%")
        # Kruskal–Wallis + η²（组间平方和 / 总平方和）
        if stat and len(rows) >= 3:
            from scipy import stats as _st
            arrays = [np.array(groups[r[0]]) for r in rows]
            try:
                H, p = _st.kruskal(*arrays)
                gm = float(vals.mean())
                ss_b = sum(len(a) * (a.mean() - gm) ** 2 for a in arrays)
                ss_t = float(((vals - gm) ** 2).sum())
                eta2 = ss_b / ss_t if ss_t > 0 else 0.0
                print(f"  [KW] H={H:.1f}  p={p:.3e}   η²(组间/总)= {eta2:.3f}")
                return {"rows": rows, "kruskal_H": float(H),
                        "kruskal_p": float(p), "eta2": float(eta2)}
            except Exception as e:
                print(f"  [KW] failed: {e}")
        return {"rows": rows}

    # 月份（物候）
    for s in sc:
        if s.get("hr_date"):
            s["month"] = int(s["hr_date"][5:7])
    # 云量分箱
    for s in sc:
        c = s.get("cloud")
        if c is None:
            continue
        s["cloud_bin"] = ("clear (<5%)" if c < 5 else
                          "mixed (5–30%)" if c < 30 else "cloudy (≥30%)")
    # 时相差分箱
    for s in sc:
        dt = s.get("delta_days")
        if dt is None:
            continue
        s["lag_bin"] = ("≤30 d" if abs(dt) <= 30 else
                        "31–180 d" if abs(dt) <= 180 else ">180 d")

    # ---- 复合检验：植被/自然地表 vs 建成区 ----
    from scipy import stats as _st
    VEG = {"Forest", "Agriculture", "Grassland", "Wetland"}
    gv = np.array([s["delta_floor"] for s in sc
                   if s.get("IPCC") in VEG and s.get("delta_floor") is not None])
    gb = np.array([s["delta_floor"] for s in sc
                   if s.get("IPCC") == "Settlement" and s.get("delta_floor") is not None])
    if len(gv) and len(gb):
        U, pu = _st.mannwhitneyu(gv, gb, alternative="less")
        d_cohen = (gb.mean() - gv.mean()) / np.sqrt((gv.var(ddof=1) + gb.var(ddof=1)) / 2)
        print(f"\n=== vegetated/natural (n={len(gv)}) vs built-up (n={len(gb)}) ===")
        print(f"  vegetated Δ_floor={gv.mean():+.3f} ± {gv.std(ddof=1):.3f} | "
              f"built-up Δ_floor={gb.mean():+.3f} ± {gb.std(ddof=1):.3f}")
        print(f"  Mann–Whitney U p={pu:.2e}  Cohen d={d_cohen:+.3f}  "
              f"Δ={gb.mean()-gv.mean():+.3f} dB")
    else:
        U = pu = d_cohen = None

    out = {
        "all": {"n": int(len(vals)), "mean": float(vals.mean()),
                "std": float(vals.std()), "neg_frac": float((vals < 0).mean())},
        "by_IPCC": report("IPCC", "IPCC land-cover class"),
        "by_LCCS": None,
        "by_SMOD": report("SMOD", "settlement density (SMOD)"),
        "by_month": report("month", "acquisition month"),
        "by_cloud": report("cloud_bin", "cloud cover"),
        "by_lag": report("lag_bin", "LR–HR temporal lag"),
        "veg_vs_built": {
            "n_veg": int(len(gv)), "n_built": int(len(gb)),
            "veg_mean": float(gv.mean()) if len(gv) else None,
            "built_mean": float(gb.mean()) if len(gb) else None,
            "mannwhitney_p": float(pu) if pu is not None else None,
            "cohen_d": float(d_cohen) if d_cohen is not None else None},
    }
    json.dump(out, open(os.path.join(BASE, "experiments", "stratified_floor_summary.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[saved] stratified_floor_summary.json")


# ------------------------------------------------------------------- regress
# A′：把 Δ_floor 从"类别关联"升级为"连续物理量的函数"
FEATS = ["ndvi_mean", "ndvi_std", "bright_lr", "spec_het", "bright_hr", "texture_hr"]
FEAT_LABEL = {
    "ndvi_mean": "NDVI (mean)",
    "ndvi_std": "NDVI (spatial s.d.)",
    "bright_lr": "LR visible brightness",
    "spec_het": "spectral heterogeneity (12-band s.d.)",
    "bright_hr": "HR brightness",
    "texture_hr": "HR texture energy (|∇|)",
}


def regress():
    """OLS: per-scene Δ_floor ~ [NDVI, texture, brightness, spectral heterogeneity];
    与 IPCC 类别 one-hot 模型的解释力对照。零训练。"""
    from scipy import stats
    d = json.load(open(OUT_JSON, encoding="utf-8"))
    sc = [s for s in d["scenes"]
          if s.get("delta_floor") is not None and s.get("ndvi_mean") is not None]
    y = np.array([s["delta_floor"] for s in sc], dtype=float)
    X = np.array([[float(s[f]) for f in FEATS] for s in sc], dtype=float)
    n, p = X.shape
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)          # 标准化 → β 即标准化系数
    yc = y - y.mean()
    beta, *_ = np.linalg.lstsq(Z, yc, rcond=None)
    yhat = Z @ beta
    resid = yc - yhat
    ss_res = float(resid @ resid)
    ss_tot = float(yc @ yc)
    r2 = 1.0 - ss_res / ss_tot
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    sigma2 = ss_res / (n - p - 1)
    se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * sigma2)
    tstat = beta / se
    pval = 2.0 * stats.t.sf(np.abs(tstat), n - p - 1)

    print(f"\n=== A′ Δ_floor ~ 物理量（OLS, n={n}）===")
    print(f"  R² = {r2:.3f}   adj.R² = {adj:.3f}")
    print(f"  {'feature':<40} {'β(std)':>9} {'t':>7} {'p':>10} {'r(Pearson)':>11} {'ρ(Spearman)':>12}")
    rows = []
    for c, f in enumerate(FEATS):
        rp = float(np.corrcoef(X[:, c], y)[0, 1])
        rho = float(stats.spearmanr(X[:, c], y).statistic)
        rows.append({"feature": f, "label": FEAT_LABEL[f],
                     "beta_std": round(float(beta[c]), 4),
                     "t": round(float(tstat[c]), 3), "p": float(pval[c]),
                     "pearson": round(rp, 4), "spearman": round(rho, 4)})
        print(f"  {FEAT_LABEL[f]:<40} {beta[c]:>+9.3f} {tstat[c]:>7.2f} "
              f"{pval[c]:>10.2e} {rp:>+11.3f} {rho:>+12.3f}")

    # --- 对照：IPCC 类别 one-hot 模型的解释力 ---
    cats = sorted(set(s.get("IPCC", "") for s in sc))
    Y = np.zeros((n, len(cats)))
    for i, s in enumerate(sc):
        Y[i, cats.index(s.get("IPCC", ""))] = 1.0
    Yd = Y[:, 1:]                                      # 丢一列避免共线
    b2, *_ = np.linalg.lstsq(Yd, yc, rcond=None)
    r2_cat = 1.0 - float(((yc - Yd @ b2) ** 2).sum()) / ss_tot
    p_cat = Yd.shape[1]
    adj_cat = 1.0 - (1.0 - r2_cat) * (n - 1) / (n - p_cat - 1)
    print(f"\n  [对照] IPCC 类别 one-hot（{len(cats)} 类，{p_cat} 自由度）: "
          f"R² = {r2_cat:.3f}  adj.R² = {adj_cat:.3f}")

    # --- 逐步最简模型：只留 |t| 最大的三个特征 ---
    order = np.argsort(-np.abs(tstat))
    keep = [FEATS[i] for i in order[:3]]
    Xs = np.array([[float(s[f]) for f in keep] for s in sc])
    Zs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
    bs, *_ = np.linalg.lstsq(Zs, yc, rcond=None)
    r2_s = 1.0 - float(((yc - Zs @ bs) ** 2).sum()) / ss_tot
    print(f"  [简模型] 仅 {{{', '.join(FEAT_LABEL[k] for k in keep)}}}: R² = {r2_s:.3f}")

    out = {"n": int(n), "features": rows, "r2": round(r2, 4),
           "adj_r2": round(adj, 4), "ipcc_r2": round(r2_cat, 4),
           "ipcc_adj_r2": round(adj_cat, 4), "n_ipcc_classes": len(cats),
           "compact_features": keep, "compact_r2": round(r2_s, 4)}
    json.dump(out, open(os.path.join(BASE, "experiments",
                                     "floor_predictors_regression.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[saved] floor_predictors_regression.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap scenes")
    a = ap.parse_args()
    if a.limit:
        globals()["_LIMIT"] = a.limit
    if a.extract:
        extract()
    if a.analyze:
        analyze()
    if a.regress:
        regress()
    if not (a.extract or a.analyze or a.regress):
        print("pass --extract / --analyze / --regress")


if __name__ == "__main__":
    main()
