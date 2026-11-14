# -*- coding: utf-8 -*-
"""
Path A —— 冠层动态专题（完全本地，零下载）。

目的：把「预期冠层动态」从 **IPCC 类别**（森林/建成区）换成**该像元自身实测的 NDVI 时间变化**，
      检验「采集间隔惩罚」是否由**真实冠层变化**而非**日历时间**设定。

数据（全部来自已解包的 WorldStrat 归档；根目录取环境变量 `WORLDSTRAT_ROOT`，默认 `~/worldstrat`）
  每个覆盖 <cov>/L2A/ 下有 16 个 S2 L2A 时相：
    <cov>-<N>-L2A_data.tiff  (H,W,12) 0–1 反射率，B4=idx3, B8=idx7
    <cov>-<N>-CLM.tiff       云掩膜（1=云）
    <cov>-<N>.metadata       {"target_date": HR 日期, "datetime": LR 日期, "delta", "cloud_cover"}

裁剪窗口**严格复现** `code/stratify_floor.py --extract` 的口径（同一中心窗），
使 ΔNDVI 与 Δ_floor 落在同一批像元上。

输出 experiments/canopy_dynamics.jsonl（每覆盖一行）

用法：
  python code/canopy_dynamics.py --probe 20      # 小样本试跑
  python code/canopy_dynamics.py --workers 8     # 全量
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import tifffile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# WorldStrat 数据根：优先取环境变量，否则退回用户主目录下的 worldstrat/~（可移植默认）
WS_ROOT = os.environ.get("WORLDSTRAT_ROOT") or os.path.expanduser(os.path.join("~", "worldstrat"))
OUT = os.path.join(BASE, "experiments", "canopy_dynamics.jsonl")
SRC_JSON = os.path.join(BASE, "experiments", "stratified_floor_ws.json")

HR_OUT = 424
RED_I, NIR_I = 3, 7          # S2 12 波段 = [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12]
CLM_BAD = 0.25               # 窗内云像元占比上限（超过则该帧不用于状态量）
AMP_HALF_WINDOW = 200        # 振幅统计窗口（天）


def _date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def parse_meta(path):
    """兼容 WorldStrat 的两套逐帧 metadata schema：

      A) ASMSpotter / Amnesty POI —— 5 字段旧版
         {"cloud_cover":0.3,"target_date":"2016-10-18","delta":"309","area":2.5,"datetime":"2017-08-23"}
         target_date = HR 日期，datetime = LR(S2) 日期。
      B) Landcover / UNHCR —— 12 字段新版
         {"n":1,"lon":..,"lat":..,"bounds":"..","lowres_date":"2017-04-21",
          "highres_date":"2017-04-10","delta":11,"rank":46.5,"cloud_cover":0.0,
          "area":2.5,"LCCS":200,"SMOD":"..","IPCC":".."}
         highres_date = HR 日期，lowres_date = LR(S2) 日期。
    """
    try:
        d = json.loads(open(path, encoding="utf-8").read())
    except Exception:
        return None
    td = _date(d.get("target_date") or d.get("highres_date"))     # HR
    dt = _date(d.get("datetime") or d.get("lowres_date"))         # LR (S2)
    if td is None or dt is None:
        return None
    return {"target": td, "acq": dt,
            "cloud_meta": float(d.get("cloud_cover", np.nan))}


def _shape(path):
    """复现 `stratify_floor.read_tif` 的轴判读：返回 (H, W)。

    read_tif 后 shape[-2:] 恒为 (H, W)，故需先按其转置规则还原：
      (C,H,W) 且 C∈{1,3} 且 C < H//2 → 保持，H,W = s[1],s[2]
      否则（含 (H,W,C)）            → 转置，H,W = s[0],s[1]
    """
    with tifffile.TiffFile(path) as tf:
        s = tuple(int(x) for x in tf.series[0].shape)
    if len(s) == 2:
        return s[0], s[1]
    if len(s) == 3:
        if s[0] in (1, 3) and s[0] < s[1] // 2:
            return s[1], s[2]
        return s[0], s[1]
    return None


def hr_shape(cov):
    """优先 `_ps.tiff`（与 stratify_floor.build_manifest 的 setdefault 顺序一致）。"""
    d = os.path.join(WS_ROOT, cov)
    try:
        fs = os.listdir(d)
    except OSError:
        return None
    cand = [f for f in fs if f.endswith("_ps.tiff")] or \
           [f for f in fs if f.endswith("_rgbn.tiff")]
    if not cand:
        return None
    try:
        return _shape(os.path.join(d, cand[0]))
    except Exception:
        return None


def load_l2a(path):
    """逐字复现 `stratify_floor.read_tif(path)`（无 div 分支），返回 (C,H,W) float32 ∈[0,1]。

    ⚠ 必须做 `max>1 → /65535` 与 `clip(0,1)`：部分覆盖的 L2A 含负值/nodata 哨兵，
    不裁剪会让 NDVI 爆出 [−1,1]（实测 ndvi_amp 最高 3.8，12.6% 的覆盖 NDVI 不匹配）。
    """
    arr = tifffile.imread(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1] // 2:
            pass
        else:
            arr = np.transpose(arr, (2, 0, 1))
    if arr.max() > 1.0:
        arr = arr / 65535.0
    return np.clip(arr, 0.0, 1.0)


def ndvi_window(arr, i, j, w):
    """arr: (C,H,W)。返回窗内 NDVI 均值。

    ⚠ 必须**逐字复现** `stratify_floor.extract` 的算式
        ndvi = (nir − red) / (nir + red + 1e-6)   →  .mean()
    不得改用 nanmean / 不得把 den≈0 的像元置 NaN：那样会在含水体/暗像元的窗口上
    产生系统性偏差（实测最大 0.0085）。
    """
    if arr.ndim != 3:
        return np.nan
    sub = arr[:, i:i + w, j:j + w]
    red = sub[RED_I].astype(np.float32)
    nir = sub[NIR_I].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    with np.errstate(invalid="ignore"):
        return float(np.mean(ndvi))


def one_cov(cov):
    d = os.path.join(WS_ROOT, cov, "L2A")
    if not os.path.isdir(d):
        return None
    files = os.listdir(d)
    metas = {}
    for f in files:
        m = re.match(r"^%s-(\d+)\.metadata$" % re.escape(cov), f)
        if m:
            info = parse_meta(os.path.join(d, f))
            if info:
                metas[int(m.group(1))] = info
    if not metas:
        return None
    hs = hr_shape(cov)
    if hs is None:
        return None
    Hr, Wr = hs

    frames = []
    anchor_dims = None
    for n in sorted(metas):
        tif = os.path.join(d, "%s-%d-L2A_data.tiff" % (cov, n))
        if not os.path.exists(tif):
            continue
        arr = load_l2a(tif)                       # (C,H,W) ∈[0,1]，同 read_tif
        if anchor_dims is None:
            Hl, Wl = arr.shape[-2], arr.shape[-1]
            ratio = float(Hr) / float(Hl)
            hr_size = min(HR_OUT, Hr, Wr)
            lr_win = max(2, min(int(round(hr_size / ratio)), Hl, Wl))
            i = max(0, (Hl - lr_win) // 2)
            j = max(0, (Wl - lr_win) // 2)
            anchor_dims = (i, j, lr_win)
        i, j, lr_win = anchor_dims
        if arr.shape[-2] < i + lr_win or arr.shape[-1] < j + lr_win:
            continue
        nd = ndvi_window(arr, i, j, lr_win)
        # 云掩膜
        clm_frac = np.nan
        cp = os.path.join(d, "%s-%d-CLM.tiff" % (cov, n))
        if os.path.exists(cp):
            try:
                clm = tifffile.imread(cp)
                if clm.ndim == 3:
                    clm = clm[..., 0]
                sub = clm[i:i + lr_win, j:j + lr_win]
                clm_frac = float((sub > 0.5).mean())
            except Exception:
                pass
        frames.append({"n": n, "acq": metas[n]["acq"].strftime("%Y-%m-%d"),
                       "hr": metas[n]["target"].strftime("%Y-%m-%d"),
                       "ndvi": nd, "clm_frac": clm_frac,
                       "cloud_meta": metas[n]["cloud_meta"],
                       "dt_target": (metas[n]["acq"] - metas[n]["target"]).days})

    if not frames:
        return None
    # ⚠ 口径对齐：主项目 `stratify_floor.py` 的 LR = `sorted(lrs)[0]`，即 **n=1** 帧。
    #   其 delta_days（官方 metadata.csv）= n=1 的 (LR − HR) 天数。
    #   因此「配对帧」必须是 n=1，否则 ΔNDVI 与 Δ_floor 不在同一帧上。
    n1 = next((f for f in frames if f["n"] == 1), None)
    if n1 is None:
        return None

    usable = [f for f in frames if np.isfinite(f["ndvi"]) and
              (not np.isfinite(f["clm_frac"]) or f["clm_frac"] <= CLM_BAD)]
    if len(usable) < 2:
        return None

    # 地表动态幅度：全部可用帧的 NDVI 分位跨度（表征「这块地有多活跃」）
    #   主口径 ndvi_amp = P95−P05；抗污染口径 ndvi_iqr = P75−P25
    v = np.array([f["ndvi"] for f in usable], float)
    amp = float(np.percentile(v, 95) - np.percentile(v, 5))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    std_all = float(v.std(ddof=1))
    p95, p05 = float(np.percentile(v, 95)), float(np.percentile(v, 5))

    # 次口径：仅取 n=1 附近 ±200 天窗口（同季对照，剔除年际漂移）
    cen = datetime.strptime(n1["acq"], "%Y-%m-%d")
    win = [f for f in usable
           if abs((datetime.strptime(f["acq"], "%Y-%m-%d") - cen).days) <= AMP_HALF_WINDOW]
    amp_win = np.nan
    if len(win) >= 4:
        vw = np.array([f["ndvi"] for f in win], float)
        amp_win = float(np.percentile(vw, 95) - np.percentile(vw, 5))

    return {"cov": cov, "n_frames": len(frames), "n_usable": len(usable),
            "hr_date": n1["hr"],
            "n1_acq": n1["acq"], "ndvi_n1": n1["ndvi"], "clm_n1": n1["clm_frac"],
            "dt_n1": n1["dt_target"],
            "ndvi_amp": amp, "ndvi_iqr": iqr, "ndvi_std_all": std_all,
            "ndvi_p95": p95, "ndvi_p05": p05,
            "ndvi_amp_win": amp_win, "n_win": len(win),
            "frames": [{"n": f["n"], "acq": f["acq"], "hr": f["hr"], "dt": f["dt_target"],
                        "ndvi": f["ndvi"], "clm": f["clm_frac"]} for f in frames]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    d = json.load(open(SRC_JSON, encoding="utf-8"))
    covs = [s["cov"] for s in d["scenes"]]
    if a.probe:
        covs = covs[:a.probe]
    print("[info] coverages = %d" % len(covs), flush=True)

    ok = 0
    t0 = __import__("time").time()
    with open(OUT, "w", encoding="utf-8") as f, ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one_cov, c): c for c in covs}
        for k, fu in enumerate(as_completed(futs)):
            try:
                r = fu.result()
            except Exception as e:
                r = None
                print("[warn] %s: %s" % (futs[fu], type(e).__name__), flush=True)
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                ok += 1
            if (k + 1) % 100 == 0:
                el = __import__("time").time() - t0
                print("  ..%d/%d ok=%d  (%.0fs, %.1f/s)"
                      % (k + 1, len(covs), ok, el, (k + 1) / max(el, 1e-6)), flush=True)
    print("[ok] wrote %s  n=%d / %d" % (OUT, ok, len(covs)))


if __name__ == "__main__":
    main()
