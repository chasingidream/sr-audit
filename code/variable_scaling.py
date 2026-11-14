#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path B —— 变量信息含量标度（variable information-content scaling）。

科学问题
--------
SEN2VENµS 的原生产品结构本身携带着一条信息：同一次 VENµS / Sentinel-2 配对里，

    可见光-近红外组  b2b3b4b8 : HR 5 m  ↔  LR 10 m   （2×）
    红边组            b5b6b7b8a : HR 5 m  ↔  LR 20 m   （4×）

即 **红边（叶绿素）波段的下行观测分辨率，天生比可见光低一倍**。主文前四轴全部是
PSNR / 保真度导向，缺少"变量语义层"的标度。本脚本回答：

    在统一的 5 m HR 真值网格上，一个光谱变量（植被指数）的 **空间方差** 有多大
    比例能被 LR 观测 + bicubic 恢复？

        Δ_V = 1 − MSE(V_pred, V_HR) / Var(V_HR)

    Δ_V = 1  → 完美恢复
    Δ_V = 0  → 与"只预测该 patch 的变量均值"等价（bicubic 未带来任何空间信息）
    Δ_V < 0  → 比常数预测还差

两种上采样口径（非线性比值指数不与插值交换，必须同报）
------------------------------------------------------
    direct  : 先在 LR 网格算变量 V_lr，再 bicubic(V_lr) → HR 网格
    bandwise: 先把 LR 各波段 bicubic → HR 网格，再算变量（= 业务 / CDSE 口径）

由于 Δ_V 用 Var(V_HR) 归一，它跨变量可比（NDVI ~0.5、NDRE ~0.3、CI_green ~1–3）。

因子设计：把"变量类型"与"倍率"分开（本脚本的关键）
--------------------------------------------------
只有 native 产品时，vis 组永远 2×、red-edge 组永远 4×，两者与"变量类型"完全共线，
无法判断红边变量 Δ_V 低是"因为它粗一倍"还是"因为叶绿素信息本身更细碎"。为此额外
构造**由 5 m HR 面积降采样得到的合成 LR（synth）**，使两组变量都能在同一倍率下评估：

    scale_protocol               vis (b2b3b4b8)          red-edge (b5b6b7b8a)
    native2  真实 S2 10 m LR             ✓                        ✗
    native4  真实 S2 20 m LR      ✓（由 native 10 m 再降）         ✓
    synth2   area↓ HR → 10 m            ✓                        ✓   ← 匹配 2×
    synth4   area↓ HR → 20 m            ✓                        ✓   ← 匹配 4×

于是 {2×,4×} × {vis, red-edge} 的 2×2 完全因子设计成立。判读：
  * 若匹配倍率下红边变量的 Δ_V 仍系统性更低 → "叶绿素信息本质上需要更细尺度" 成立；
  * 若匹配后差异消失 → 原生差异纯粹是倍率差异（红边只是被 4× 采样，变量本身不特殊）。

诚实边界：synth LR 是 HR 的干净面积平均（无传感器噪声 / MTF / 配准差异），会系统性
高估绝对 Δ_V；但它**对两组、两倍率施加完全相同**，故因子对比有效，绝对值不可外推。

反射率口径：raw/10000 后 **clip 到 [0,1]**（与 band 级 loader 的"不 clip"有意分歧）。
理由：比值指数要求分母不穿零，clip 后 a,b>=0 自动给出 |(a-b)/(a+b)| <= 1。
实测 FGMANAUS 可见光 30–56% 像元为负（红均值仅 0.003，贴噪声底），SO2 为 0%；
该处理对两站点一致施加，不引入站点间偏置。比值指数另设**物理分母下限**
OOB_MIN = 1% 反射率，低于此的像元剔除并计入 frac_oob。

输出
----
experiments/variable_scaling.jsonl。每行 =
  {"src","patch","date","group","variable","protocol","scale","lrtag",
   "delta_v","mse","var_hr","corr_lr_hr","frac_oob","lag1","n_pix"}
其中 lag1 = HR 变量的 lag-1 空间自相关，用于识别"变量本身就是噪声"的退化情形
（此时 Δ_V 无意义，须在分析阶段剔除而非取平均）。

逐 patch 落盘（不池化），统计口径与主文一致：以 patch 为采样单元做配对检验。

用法
----
  python code/variable_scaling.py --probe 3          # 小样本口径自查
  python code/variable_scaling.py --workers 8        # 全量抽取
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import tifffile  # noqa: E402
import io  # noqa: E402

from stratify_floor import bicubic_np  # noqa: E402  （项目统一 bicubic 口径）

DIV = 10000.0          # SEN2VENµS 反射率定标（与 datasets_external.Sen2VenusDataset 一致）
EPS = 1e-6             # 数值下限（防 0 除）
OOB_MIN = 0.01         # 比值指数分母的**物理**下限 = 1% 反射率。低于此，比值由
                       # 反演噪声主导；clip 后分母非负，故直接 den >= OOB_MIN 判有效。
HR_OUT = 256           # 5 m HR 网格（原生即 256×256）

FG_ROOT = os.path.join(BASE, "data", "external", "sen2venus", "FGMANAUS")
SO2_ROOT = os.path.join(BASE, "data", "external", "sen2venus_hf", "SO2")
OUT_JSONL = os.path.join(BASE, "experiments", "variable_scaling.jsonl")


# ---------------------------------------------------------------- 变量定义
# arr: (C,H,W) float32 反射率（/10000 后，不 clip —— 与 loader 契约一致）
# 返回 (V, denom)；denom 为 None 表示无退化风险，否则 |denom| < EPS 的像元出界。

def _vis_ndvi(a):
    R, NIR = a[2], a[3]
    d = NIR + R
    return (NIR - R) / np.where(np.abs(d) < EPS, EPS, d), d


def _vis_greenness(a):
    G, R = a[1], a[2]
    d = G + R
    return (G - R) / np.where(np.abs(d) < EPS, EPS, d), d


def _vis_ndwi(a):
    G, NIR = a[1], a[3]
    d = G + NIR
    return (G - NIR) / np.where(np.abs(d) < EPS, EPS, d), d


def _vis_brightness(a):
    return a.mean(axis=0), None


def _vis_evi(a):
    B, R, NIR = a[0], a[2], a[3]
    d = NIR + 6.0 * R - 7.5 * B + 1.0
    dd = np.where(np.abs(d) < EPS, np.where(d >= 0, EPS, -EPS), d)
    return 2.5 * (NIR - R) / dd, d


def _vis_nir(a):
    return a[3].copy(), None


def _vis_red(a):
    return a[2].copy(), None


def _re_ndre(a):
    B5, B8A = a[0], a[3]
    d = B8A + B5
    return (B8A - B5) / np.where(np.abs(d) < EPS, EPS, d), d


def _re_ndre2(a):
    B6, B8A = a[1], a[3]
    d = B8A + B6
    return (B8A - B6) / np.where(np.abs(d) < EPS, EPS, d), d


def _re_ci_redge(a):
    B7, B8A = a[2], a[3]
    d = B7
    return B8A / np.where(np.abs(d) < EPS, EPS, d) - 1.0, d


def _re_b8a(a):
    return a[3].copy(), None


VIS_VARS = [
    ("ndvi", _vis_ndvi), ("greenness", _vis_greenness), ("ndwi", _vis_ndwi),
    ("brightness", _vis_brightness), ("evi", _vis_evi),
    ("nir", _vis_nir), ("red", _vis_red),
]
# 注：CI_green = B8A/B5 − 1 未纳入——其分母是**单波段** B5，而 B5 在 FGMANAUS 有
# 84% 像元低于 0.05（红边组本身暗），分母贴近噪声底。改用 CI_red-edge = B8A/B7 − 1
# （同一 chlorophyll-index 构造，分母 B7≈0.26 稳健）。
RE_VARS = [
    ("ndre", _re_ndre), ("ndre2", _re_ndre2),
    ("ci_redge", _re_ci_redge), ("b8a", _re_b8a),
]
VARS = {"vis": VIS_VARS, "re": RE_VARS}


# ---------------------------------------------------------------- 工具

def area_down(x, hw):
    """面积平均降采样（抗混叠），逐通道。x: (C,H,W) → (C,h,w)。"""
    c = x.shape[0]
    out = np.empty((c, hw[0], hw[1]), dtype=np.float32)
    for i in range(c):
        out[i] = cv2.resize(x[i], (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return out


def _read_bytes(raw):
    a = tifffile.imread(io.BytesIO(raw))
    if a.ndim == 2:
        a = a[None]
    return np.moveaxis(a, 0, -1) if (a.ndim == 3 and a.shape[0] <= 12) else a


def refl(raw):
    """反射率 = raw/DIV，并 clip 到 [0,1]。

    ⚠ 与 datasets_external.Sen2VenusDataset.load 的"不 clip"契约**有意分歧**：
    band 级 PSNR 可以容忍负反射率，但**比值类指数不能**——负值让 (a-b)/(a+b) 的
    分母穿过零点，指数发散（实测 FGMANAUS greenness Var 冲到 8.4e4，lag-1 自相关
    塌到 0.009）。clip 后 a,b>=0 保证 |(a-b)/(a+b)| <= 1，指数天然有界。
    实测 FGMANAUS 可见光波段 30-56% 像元为负（红波段均值仅 0.003，贴着噪声底），
    SO2 则 0%——故该处理对两站点同样施加，不引入站点间偏置。
    """
    a = _read_bytes(raw)
    a = a if a.ndim == 3 else a[None]
    a = np.moveaxis(a, -1, 0).astype(np.float32) / DIV    # (C,H,W)
    return np.clip(a, 0.0, 1.0)


_ZIP_CACHE: dict = {}


def _open_zip(path):
    z = _ZIP_CACHE.get(path)
    if z is None:
        z = zipfile.ZipFile(path)
        _ZIP_CACHE[path] = z
    return z


# ---------------------------------------------------------------- 单变量评估

def eval_one(vname, fn, hr_bands, lr_bands, protocol):
    """在 HR 网格上评估一个变量。

    hr_bands : (C,Hh,Wh)  5 m 真值反射率
    lr_bands : (C,Hl,Wl)  LR 反射率
    返回 dict（已按 HR 有效掩膜配平）。
    """
    Hh, Wh = hr_bands.shape[1], hr_bands.shape[2]
    V_hr, den_hr = fn(hr_bands)

    # LR 侧变量（direct 口径的预测源；同时用于 corr 诊断）
    V_lr, den_lr = fn(lr_bands)

    # 有效掩膜：HR 分母健康 **且 LR 侧同样健康**。
    # 关键：比值指数在 LR 上可能因分母穿零而发散（ci_redge = B8A/B7−1 最甚），
    # 再经 bicubic 传播到 HR 网格。若只在 HR 侧判有效，被污染的预测值会进 MSE
    # （实测把 ci_redge 的 Δ_V 打到 −9e3）。故三件事一起做：
    #   (1) 在 LR 网格上判有效（den_lr >= OOB_MIN）；
    #   (2) **上采样之前**把失效处的 V_lr 填中性值，杜绝插值从极端值取数；
    #   (3) 把失效掩膜**膨胀 2 个 LR 像元**（覆盖 bicubic 的 4×4 支撑域）后
    #       最近邻上采样，与 HR 掩膜取交集——保证 MSE 只在两侧都可信处评估。
    Hl, Wl = lr_bands.shape[1], lr_bands.shape[2]
    v_lr = np.ones((Hl, Wl), dtype=bool)
    bad_lr = np.zeros((Hl, Wl), dtype=np.uint8)
    if den_lr is not None:
        v_lr &= den_lr >= OOB_MIN
        if (~v_lr).any():
            fill = float(np.median(V_lr[v_lr])) if v_lr.any() else 0.0
            V_lr = np.where(v_lr, V_lr, fill)
            bad_lr = cv2.dilate((~v_lr).astype(np.uint8), np.ones((5, 5), np.uint8))

    if protocol == "direct":
        V_pred = bicubic_np(V_lr[None], (Hh, Wh))[0]
        den_up = None
    else:  # bandwise：先上采样波段再算变量
        up = bicubic_np(lr_bands, (Hh, Wh))
        V_pred, den_up = fn(up)

    valid = np.ones((Hh, Wh), dtype=bool)
    if den_hr is not None:
        valid &= den_hr >= OOB_MIN
    if bad_lr.any():
        valid &= cv2.resize(1 - bad_lr, (Wh, Hh),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    if den_up is not None:                      # bandwise：上采样波段自身的分母
        valid &= den_up >= OOB_MIN

    n_pix = int(valid.sum())
    frac_oob = float(1.0 - valid.mean())
    if n_pix < 64:
        return None

    diff = (V_pred - V_hr)[valid]
    mse = float(np.mean(diff * diff))
    var_hr = float(np.var(V_hr[valid]))
    delta_v = float(1.0 - mse / var_hr) if var_hr > 1e-12 else float("nan")

    # lag1：HR 变量自身的空间连续性。接近 0 ⇒ 该变量在 5 m 网格上本就是逐像元噪声，
    # 此时 Δ_V 无意义（分母是被噪声撑起来的方差）。用于后续剔除退化变量。
    x = V_hr - V_hr.mean()
    lag1 = float((np.mean(x[:, :-1] * x[:, 1:]) + np.mean(x[:-1, :] * x[1:, :]))
                 / (2.0 * np.var(V_hr) + 1e-12))

    # corr_lr_hr：LR 观测本身能"看见"多少变量的空间格局（上采样无法超越的天花板）
    corr = float("nan")
    try:
        V_hr_dn = area_down(V_hr[None], (Hl, Wl))[0]
        m = v_lr.copy()
        a1, a2 = V_lr[m], V_hr_dn[m]
        if a1.size >= 64 and a1.std() > 1e-9 and a2.std() > 1e-9:
            corr = float(np.corrcoef(a1, a2)[0, 1])
    except Exception:
        pass

    return {"delta_v": delta_v, "mse": mse, "var_hr": var_hr,
            "corr_lr_hr": corr, "frac_oob": frac_oob, "lag1": lag1,
            "n_pix": n_pix}


def eval_patch(group, hr_bands, lr_variants):
    """lr_variants: [(lrtag, scale, lr_bands), ...] → 每变量每口径每变体一行。"""
    rows = []
    for lrtag, scale, lr_bands in lr_variants:
        if lr_bands is None:
            continue
        for vname, fn in VARS[group]:
            for protocol in ("direct", "bandwise"):
                r = eval_one(vname, fn, hr_bands, lr_bands, protocol)
                if r is None:
                    continue
                rows.append({"group": group, "variable": vname,
                             "protocol": protocol, "scale": scale,
                             "lrtag": lrtag, **r})
    return rows


# ---------------------------------------------------------------- 数据驱动

def fgmanaus_tasks(limit=None):
    idx = os.path.join(FG_ROOT, "FGMANAUS", "index.csv")
    lines = [l.rstrip("\n") for l in open(idx, encoding="utf-8") if l.strip()]
    hdr = lines[0].split("\t")[1:]
    rows = [l.split("\t") for l in lines[1:]]
    tasks = []
    for r in rows[:limit] if limit else rows:
        rels = {hdr[j]: r[j + 1] for j in range(len(hdr))}
        base = os.path.basename(rels[hdr[0]])
        p = base.split("_")
        tasks.append({"src": "fgmanaus", "patch": r[0], "date": p[2],
                      "rels": rels, "kind": "fg"})
    return tasks


def so2_tasks(limit=None):
    fs = sorted(f for f in os.listdir(SO2_ROOT) if f.endswith(".zip"))
    by_key = {}
    for f in fs:
        p = f[:-4].split("_")        # SO2_{idx}_{date}_31TCJ_b2b3b4b8_{res}
        key = (p[1], p[2])
        by_key.setdefault(key, {})[p[-1]] = f
    tasks = []
    keys = sorted(by_key)[:limit] if limit else sorted(by_key)
    for (pi, date) in keys:
        d = by_key[(pi, date)]
        if "05m" not in d or "10m" not in d:
            continue
        tasks.append({"src": "so2", "patch": f"{pi}_{date}", "date": date,
                      "rels": d, "kind": "so2"})
    return tasks


def run_task(t):
    """读一个 patch，构造 HR + 全部 LR 变体，评估。"""
    try:
        if t["kind"] == "fg":
            rels = t["rels"]

            def rd(c):
                rel = rels[c]
                zname, member = rel.split("/", 1)
                return refl(_open_zip(os.path.join(FG_ROOT, zname)).read(member))

            hr_v = rd("b2b3b4b8_05m")
            lr_v = rd("b2b3b4b8_10m")
            hr_r = rd("b5b6b7b8a_05m")
            lr_r = rd("b5b6b7b8a_20m")
        else:
            rels = t["rels"]

            def rd2(res):
                zp = os.path.join(SO2_ROOT, rels[res])
                z = _open_zip(zp)
                return refl(z.read(z.namelist()[0]))

            hr_v = rd2("05m")
            lr_v = rd2("10m")
            hr_r, lr_r = None, None

        Hh, Wh = HR_OUT, HR_OUT
        out = []

        # ---- vis 组（b2b3b4b8）
        if hr_v.shape[1] != Hh:
            hr_v = bicubic_np(hr_v, (Hh, Wh))
        base_lr = lr_v
        variants_v = [
            ("native2", 2, base_lr),
            ("native4", 4, area_down(base_lr, (Hh // 4, Wh // 4))),
            ("synth2", 2, area_down(hr_v, (Hh // 2, Wh // 2))),
            ("synth4", 4, area_down(hr_v, (Hh // 4, Wh // 4))),
        ]
        out += eval_patch("vis", hr_v, variants_v)

        # ---- red-edge 组（b5b6b7b8a）—— 仅 FGMANAUS 有
        if hr_r is not None:
            if hr_r.shape[1] != Hh:
                hr_r = bicubic_np(hr_r, (Hh, Wh))
            variants_r = [
                ("native4", 4, lr_r),
                ("synth2", 2, area_down(hr_r, (Hh // 2, Wh // 2))),
                ("synth4", 4, area_down(hr_r, (Hh // 4, Wh // 4))),
            ]
            out += eval_patch("re", hr_r, variants_r)

        for row in out:
            row.update({"src": t["src"], "patch": t["patch"], "date": t["date"]})
        return out
    except Exception as e:  # noqa: BLE001
        return {"__error__": f"{t['src']}/{t['patch']}: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0, help="每源只跑 N 个 patch（口径自查）")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=OUT_JSONL)
    ap.add_argument("--sources", default="fgmanaus,so2")
    args = ap.parse_args()

    srcs = [s.strip() for s in args.sources.split(",") if s.strip()]
    tasks = []
    if "fgmanaus" in srcs:
        tasks += fgmanaus_tasks(args.probe or None)
    if "so2" in srcs:
        tasks += so2_tasks(args.probe or None)
    print(f"[variable_scaling] tasks={len(tasks)} workers={args.workers}", flush=True)

    t0 = time.time()
    rows, errs, done = [], [], 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_task, t) for t in tasks]
            for f in as_completed(futs):
                r = f.result()
                done += 1
                if isinstance(r, dict):
                    errs.append(r["__error__"])
                else:
                    rows += r
                if done % 25 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} patches  rows={len(rows)} "
                          f"errs={len(errs)}  {time.time()-t0:.0f}s", flush=True)
    else:
        for t in tasks:
            r = run_task(t)
            done += 1
            if isinstance(r, dict):
                errs.append(r["__error__"])
            else:
                rows += r
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} patches  rows={len(rows)} "
                      f"errs={len(errs)}  {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[variable_scaling] wrote {len(rows)} rows → {args.out}  "
          f"({time.time()-t0:.0f}s, {len(errs)} errors)", flush=True)
    if errs:
        print("  first errors:", errs[:5], flush=True)

    # ---- probe 摘要
    if args.probe and rows:
        import collections
        agg = collections.defaultdict(list)
        for r in rows:
            if r["protocol"] == "direct":
                agg[(r["src"], r["group"], r["variable"], r["lrtag"])].append(r["delta_v"])
        print("\n[probe] Δ_V direct 口径 均值（n=patch 数）")
        print(f"  {'src':9s} {'group':4s} {'variable':11s} {'lrtag':8s} {'Δ_V':>8s}  n")
        for k in sorted(agg):
            v = np.array(agg[k], float)
            print(f"  {k[0]:9s} {k[1]:4s} {k[2]:11s} {k[3]:8s} {v.mean():8.4f}  {len(v)}")


if __name__ == "__main__":
    main()
