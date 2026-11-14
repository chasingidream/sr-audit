# -*- coding: utf-8 -*-
"""
从 WorldClim 2.1 (10 arc-min) **逐月** 降水/气温栅格按景坐标采样，并合成年度气候学量。

为什么是独立数据：
  WorldClim 2.1 是由全球气象站观测（1970–2000）空间插值得到的气候学面
  (Fick & Hijmans 2017)，既不来自本文影像，也不来自本文所用任何传感器；
  时间不变（multi-decadal climatology），因此不存在「用同一景推断同一景」的循环。

输入（逐变量小包，避免 50 MB bio 整包）：
  data/external/worldclim/_prec.zip  -> wc2.1_10m_prec_01..12.tif  (mm, 月总量)
  data/external/worldclim/_tavg.zip  -> wc2.1_10m_tavg_01..12.tif  (°C × 10, 月均值)
  data/external/worldclim/wc2.1_10m_elev.zip -> wc2.1_10m_elev.tif (m)

合成量（与 WorldClim bio 定义一致）：
  precip_ann          = Σ_m prec_m                     (mm)
  t_mean_ann          = mean_m tavg_m                  (°C)
  t_seasonality       = SD_m tavg_m                    (°C)      [= bio4/100]
  precip_cv           = SD_m prec_m / mean_m prec_m    (fraction)[= bio15/100]
  precip_seasonality  = SD_m prec_m                    (mm)
  elevation           = elev                           (m)

输出：experiments/climate_independent.jsonl（cov join key，供 geoscience_validate.py 直接消费）
"""
import json
import os
import zipfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
WC = os.path.join(ROOT, "data", "external", "worldclim")
EXTRACT = os.path.join(WC, "wc2.1_10m_monthly")

RES = 10.0 / 60.0
LON0, LAT0 = -180.0, 90.0
# prec: int32, mm/month, nodata -32768 ; tavg: float32, °C, nodata -3.4e38 ; elev: int32, m, nodata -32768
NODATA = {-9999, -32768, 65535, -2147483648}


def is_nodata(v):
    return (v in NODATA) or (v < -1e30) or (v > 1e30)


def extract(zp, tag):
    os.makedirs(EXTRACT, exist_ok=True)
    if not os.path.exists(zp):
        raise FileNotFoundError(zp)
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        for n in names:
            tgt = os.path.join(EXTRACT, tag + "_" + os.path.basename(n))
            if os.path.exists(tgt):
                continue
            with zf.open(n) as src, open(tgt, "wb") as dst:
                dst.write(src.read())
    print("[ok] extracted %s (%d files)" % (os.path.basename(zp), len(names)))


def load_month(tag, m):
    p = os.path.join(EXTRACT, "%s_wc2.1_10m_%s_%02d.tif" % (tag, tag, m))
    return np.array(Image.open(p)).astype(np.float64)


def sample_bilinear(arr, lat, lon):
    xc = (lon - LON0) / RES - 0.5
    yc = (LAT0 - lat) / RES - 0.5
    if not np.isfinite(xc) or not np.isfinite(yc):
        return np.nan
    x0, y0 = int(np.floor(xc)), int(np.floor(yc))
    dx, dy = xc - x0, yc - y0
    tot = wsum = 0.0
    for j, wy in ((y0, 1 - dy), (y0 + 1, dy)):
        if j < 0 or j >= arr.shape[0]:
            continue
        for i, wx in ((x0, 1 - dx), (x0 + 1, dx)):
            if i < 0 or i >= arr.shape[1]:
                continue
            v = arr[j, i]
            if not np.isfinite(v) or is_nodata(v):
                continue
            w = wx * wy
            tot += v * w
            wsum += w
    return tot / wsum if wsum > 0 else np.nan


def main():
    extract(os.path.join(WC, "_prec.zip"), "prec")
    extract(os.path.join(WC, "_tavg.zip"), "tavg")
    P = [load_month("prec", m) for m in range(1, 13)]
    T = [load_month("tavg", m) for m in range(1, 13)]
    print("[read] 12 prec + 12 tavg layers, shape=%s" % (P[0].shape,))

    elev = np.array(Image.open(os.path.join(WC, "wc2.1_10m", "wc2.1_10m_elev.tif"))).astype(np.float64)

    d = json.load(open(os.path.join(EXP, "stratified_floor_ws.json"), encoding="utf-8"))
    scenes = d["scenes"]
    out_path = os.path.join(EXP, "climate_independent.jsonl")
    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in scenes:
            lat, lon = s.get("lat"), s.get("lon")
            if lat is None or lon is None:
                continue
            pv, tv = [], []
            for m in range(12):
                a = sample_bilinear(P[m], lat, lon)
                b = sample_bilinear(T[m], lat, lon)
                if not (np.isfinite(a) and np.isfinite(b)):
                    pv = None
                    break
                pv.append(a)
                tv.append(b)
            if not pv:
                continue
            pv = np.array(pv, float)
            tv = np.array(tv, float)               # tavg 已是 °C（float32）
            prec_ann = float(pv.sum())
            t_mean = float(tv.mean())
            t_sd = float(tv.std(ddof=1))
            if prec_ann <= 0:
                continue
            p_cv = float(pv.std(ddof=1) / pv.mean())
            rec = {"cov": s["cov"], "lat": lat, "lon": lon,
                   "year": int(str(s.get("hr_date", "2000"))[:4]),
                   "precip_ann": prec_ann, "t_mean_ann": t_mean,
                   "t_seasonality": t_sd,
                   "precip_seasonality": float(pv.std(ddof=1)),
                   "precip_cv": p_cv,
                   "elevation": float(sample_bilinear(elev, lat, lon))}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
    print("[ok] wrote %s  n=%d / %d" % (out_path, n_ok, len(scenes)))


if __name__ == "__main__":
    main()
