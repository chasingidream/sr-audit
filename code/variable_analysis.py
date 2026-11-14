#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path B 分析 —— 变量信息含量标度的排序与机制。

读 experiments/variable_scaling.jsonl，回答三问：

Q1 在**匹配倍率**下（synth2 / synth4，两组变量同源同倍率），红边变量的 Δ_V 是否
   系统性低于可见光-近红外变量？——即"叶绿素信息本质上需要更细尺度"是否成立。

Q2 若 Q1 不成立，那么排序由什么决定？候选机制：变量对**谱邻波段残差**的依赖度。
   对 (a-b)/(a+b) 型指数，若 a、b 高度相关，V 的方差主要由 a-b 的小残差驱动，
   而残差恰恰是降采样最先抹掉的高频。用 corr(a,b) 作该机制的代理量。

Q3 原生（真实跨传感器）LR 下的表现，与合成 LR 的差距 = 跨传感器代价。

统计口径与主文一致：逐 patch 取值，配对 t 检验，不池化。
输出 experiments/variable_analysis.json
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

IN = os.path.join(BASE, "experiments", "variable_scaling.jsonl")
OUT = os.path.join(BASE, "experiments", "variable_analysis.json")
FG_ROOT = os.path.join(BASE, "data", "external", "sen2venus", "FGMANAUS")

DEG_LAG1 = 0.30          # lag-1 空间自相关低于此 ⇒ 变量在 5 m 上本身即噪声，剔除


def load():
    rows = []
    with open(IN, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def index(rows, protocol="direct"):
    d = collections.defaultdict(dict)      # (src,lrtag,group,var) -> patch -> delta_v
    for r in rows:
        if r["protocol"] != protocol:
            continue
        d[(r["src"], r["lrtag"], r["group"], r["variable"])][r["patch"]] = r["delta_v"]
    return d


def get(idx, src, lrtag, var, group=None):
    """按 (src,lrtag,var) 取，group 缺省自动推断。"""
    if group is not None:
        return idx.get((src, lrtag, group, var))
    for k, v in idx.items():
        if k[0] == src and k[1] == lrtag and k[3] == var:
            return v
    return None


def describe(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), sem=float("nan"))
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                sem=float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)


def paired(a, b):
    """a、b 为同 patch 的两组值；返回 (mean diff, t, p, ci_lo, ci_hi, n)。"""
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] for k in ks], float)
    y = np.array([b[k] for k in ks], float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return None
    d = x - y
    t, p = stats.ttest_rel(x, y)
    se = d.std(ddof=1) / np.sqrt(d.size)
    tc = stats.t.ppf(0.975, d.size - 1)
    return dict(n=int(d.size), diff=float(d.mean()), t=float(t), p=float(p),
                ci_lo=float(d.mean() - tc * se), ci_hi=float(d.mean() + tc * se))


# ---------------------------------------------------------------- 机制代理量
def band_pair_corr(src="fgmanaus", group="vis", n_patch=40):
    """在 5 m HR 上测组内相邻波段的平均相关系数（谱邻程度）。"""
    lines = [l.rstrip("\n") for l in open(os.path.join(FG_ROOT, "FGMANAUS", "index.csv")) if l.strip()]
    hdr = lines[0].split("\t")[1:]
    rows = [l.split("\t") for l in lines[1:]]

    def rd(ri, c):
        rel = rows[ri][hdr.index(c) + 1]
        zn, mem = rel.split("/", 1)
        with zipfile.ZipFile(os.path.join(FG_ROOT, zn)) as z:
            return np.clip(tifffile.imread(io.BytesIO(z.read(mem))).astype(np.float32) / 1e4, 0, 1)

    c = "b2b3b4b8_05m" if group == "vis" else "b5b6b7b8a_05m"
    M = np.zeros((4, 4))
    for ri in range(min(n_patch, len(rows))):
        a = rd(ri, c)
        for i in range(4):
            for j in range(4):
                M[i, j] += np.corrcoef(a[i].ravel(), a[j].ravel())[0, 1]
    return M / min(n_patch, len(rows))


def main():
    rows = load()
    print(f"loaded {len(rows)} rows")

    # 退化剔除统计
    deg = collections.Counter()
    for r in rows:
        if r["protocol"] != "direct":
            continue
        if r["lag1"] < DEG_LAG1:
            deg[(r["src"], r["group"], r["variable"])] += 1
    print("\n### 退化变量（lag1 < %.2f 的 patch 计数 / 该变量总 patch 数）" % DEG_LAG1)
    allc = collections.Counter()
    for r in rows:
        if r["protocol"] == "direct":
            allc[(r["src"], r["group"], r["variable"])] += 1
    for k in sorted(deg, key=lambda k: -deg[k])[:8]:
        print(f"  {str(k):44s} {deg[k]:4d} / {allc[k]}")

    idx = index(rows, "direct")
    out = {"degen": {str(k): deg[k] for k in deg}, "desc": {}, "paired": {}}

    print("\n### Q1/Q3  Δ_V 描述统计（direct 口径）")
    print(f"  {'src':9s} {'lrtag':8s} {'group':4s} {'variable':11s} {'mean':>8s} {'sd':>7s} {'n':>5s}")
    for k in sorted(idx):
        s = describe(list(idx[k].values()))
        out["desc"]["|".join(k)] = s
        print(f"  {k[0]:9s} {k[1]:8s} {k[2]:4s} {k[3]:11s} {s['mean']:8.4f} {s['sd']:7.4f} {s['n']:5d}")

    # ---- 匹配倍率下的红边 vs 可见光 配对比较（仅 FGMANAUS 同时有两组）
    print("\n### Q1 匹配倍率下 红边变量 vs NDVI 的配对比较（FGMANAUS, synth LR）")
    for lrtag in ("synth2", "synth4", "native4"):
        ref = get(idx, "fgmanaus", lrtag, "ndvi")
        if not ref:
            continue
        for v in ("ndre", "ndre2", "ci_redge", "b8a"):
            tgt = get(idx, "fgmanaus", lrtag, v)
            if not tgt:
                continue
            r = paired(tgt, ref)
            if r:
                out["paired"][f"fgmanaus|{lrtag}|{v}-vs-ndvi"] = r
                print(f"  {lrtag:8s} {v:9s} - ndvi :  Δ={r['diff']:+.4f} "
                      f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p={r['p']:.2e}  n={r['n']}")

    # ---- 可见光组内：比值指数 vs 原始波段（同一倍率）
    print("\n### 附：同组内 比值指数 vs 原始波段（FGMANAUS synth4）")
    for ref_v in ("nir",):
        ref = get(idx, "fgmanaus", "synth4", ref_v)
        for v in ("ndvi", "ndwi", "greenness", "brightness"):
            tgt = get(idx, "fgmanaus", "synth4", v)
            r = paired(tgt, ref)
            if r:
                out["paired"][f"fgmanaus|synth4|{v}-vs-{ref_v}"] = r
                print(f"  {v:10s} - {ref_v:5s} :  Δ={r['diff']:+.4f} "
                      f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p={r['p']:.2e}")

    # ---- Q3 跨传感器代价：native vs synth（同倍率）
    print("\n### Q3 跨传感器代价 = Δ_V(synth) − Δ_V(native)，同倍率配对")
    for src, lg_s, lg_n in [("fgmanaus", "synth2", "native2"), ("so2", "synth2", "native2"),
                            ("fgmanaus", "synth4", "native4")]:
        for v in sorted({k[3] for k in idx if k[0] == src and k[1] in (lg_s, lg_n)}):
            a, b = get(idx, src, lg_s, v), get(idx, src, lg_n, v)
            if not a or not b:
                continue
            r = paired(a, b)
            if r:
                out["paired"][f"{src}|{v}|synth-minus-native@{lg_n[-1]}"] = r
                print(f"  {src:9s} {v:11s} @{lg_n:8s} : Δ={r['diff']:+.4f} "
                      f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  p={r['p']:.2e}  n={r['n']}")

    # ---- Q2 机制代理：组内波段相关
    print("\n### Q2 机制代理：5 m HR 组内波段相关系数（谱邻程度）")
    for g in ("vis", "re"):
        M = band_pair_corr(group=g)
        out[f"bandcorr_{g}"] = M.tolist()
        print(f"  {g}: off-diagonal 平均 = {(M.sum()-np.trace(M))/12:.4f}")
        for i in range(4):
            print("     " + "".join(f"{M[i, j]:+8.4f}" for j in range(4)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nwrote → {OUT}")

    # ---- lite 版：短键 + 去掉可导出字段（mse=(1−Δ_V)·var_hr），17.6 MB → ~5 MB
    # 键名对照：s=src p=patch d=date g=group v=variable t=protocol k=scale
    #           L=lrtag D=delta_v V=var_hr c=corr_lr_hr o=frac_oob l=lag1
    lite = os.path.join(BASE, "experiments", "variable_scaling_lite.jsonl")
    KEY = {"src": "s", "patch": "p", "date": "d", "group": "g", "variable": "v",
           "protocol": "t", "scale": "k", "lrtag": "L", "delta_v": "D",
           "var_hr": "V", "corr_lr_hr": "c", "frac_oob": "o", "lag1": "l"}
    n = 0
    with open(lite, "w", encoding="utf-8") as f:
        for r in rows:
            o = {}
            for k, short in KEY.items():
                val = r.get(k)
                if isinstance(val, float):
                    val = float(f"{val:.6g}") if val == val else None
                o[short] = val
            f.write(json.dumps(o, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    print(f"wrote → {lite}  ({n} rows, {os.path.getsize(lite)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
