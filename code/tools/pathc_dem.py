# -*- coding: utf-8 -*-
"""C-1：地形静态量提取（独立于 HR 影像）。

动机
  `texture_hr` 与 Δ_floor 都出自同一张 HR 影像，二者存在**定义层耦合**：
  Δ_floor 的基线项是 std(HR)，而 texture_hr 是 |∇HR|。审稿人会问
  "你是不是只是把 HR 的方差换了个写法"。本脚本用**完全独立的地表产品**
  替换它 —— 外部 DEM 的地形量（坡度/粗糙度/局部起伏），
  与本文任何传感器影像无关、时间不变。

数据通道
  AWS Terrain Tiles (Terrarium PNG，免认证、公开)
    https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
  解码： height_m = (R*256 + G + B/256) - 32768
  z=13 时地面分辨率 ≈ 19 m（赤道），与 SRTM 30 m 同量级，足以刻画
  1.6 km 场景内的地形结构。

派生量（每个场景 1.6 km × 1.6 km 窗口内）
  dem_mean       平均高程 (m)
  dem_std        高程标准差 (m)          —— 局部起伏
  slope_mean     平均坡度 (deg)
  slope_std      坡度标准差 (deg)
  tri_mean       地形崎岖度指数 TRI = mean |h_c - mean(h_8nb)|  (m)
  relief         窗口内 95–5 分位差 (m)
  roughness      高程一阶差分的 RMS = sqrt(mean((∇h)^2))  (m/px)

输出
  experiments/dem_terrain.jsonl   每行 {cov, lat, lon, z, n_px, <派生量>}
  data/external/dem_cache/        tile PNG 缓存（可复用，避免重复请求）

用法
  python code/tools/pathc_dem.py --probe 40
  python code/tools/pathc_dem.py --workers 8          # 全量 3,927
"""
import argparse
import csv
import json
import math
import os
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "experiments")
META = os.path.join(ROOT, "data", "raw", "metadata.csv")
CACHE = os.path.join(ROOT, "data", "external", "dem_cache")
OUT = os.path.join(EXP, "dem_terrain.jsonl")

URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 13
TILE = 256
# z13 在赤道的地面分辨率（m/px）：2πR / (256 · 2^13)
EARTH_R = 6378137.0
MPP_EQ = 2.0 * math.pi * EARTH_R / (TILE * (2 ** ZOOM))


# --------------------------------------------------------------- tile 数学
def lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(max(min(lat, 85.05112878), -85.05112878))
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def fetch_tile(z, x, y, retries=3):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{z}_{x}_{y}.png")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    url = URL.format(z=z, x=x, y=y)
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pathc-dem/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                buf = r.read()
            if buf:
                tmp = p + ".part"
                with open(tmp, "wb") as f:
                    f.write(buf)
                os.replace(tmp, p)
                return p
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"tile {z}/{x}/{y} failed: {last}")


def decode_terrarium(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return (a[:, :, 0] * 256.0 + a[:, :, 1] + a[:, :, 2] / 256.0) - 32768.0


# --------------------------------------------------------------- 地形量
def terrain_metrics(h, mpp):
    """h: 高程窗口 (m)，mpp: 该纬度处每像元米数。"""
    gy, gx = np.gradient(h, mpp)                 # 每像元米数 -> 坡度
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # TRI：中心与 8 邻域均值之差
    c = h[1:-1, 1:-1]
    nb = (h[:-2, 1:-1] + h[2:, 1:-1] + h[1:-1, :-2] + h[1:-1, 2:] +
          h[:-2, :-2] + h[:-2, 2:] + h[2:, :-2] + h[2:, 2:]) / 8.0
    tri = np.abs(c - nb)
    p5, p95 = np.percentile(h, [5, 95])
    return {
        "dem_mean": float(h.mean()),
        "dem_std": float(h.std()),
        "slope_mean": float(slope.mean()),
        "slope_std": float(slope.std()),
        "tri_mean": float(tri.mean()),
        "relief": float(p95 - p5),
        "roughness": float(np.sqrt((gx ** 2 + gy ** 2).mean())),
    }


def scene_dem(lat, lon, half_km=0.9):
    """取场景中心附近 half_km 半径的 DEM 窗口，返回地形量。

    注意：不用 metadata 的 bounds（那是 WorldStrat 影像的取景框），
    而用场景中心的等距窗口，避免依赖影像本身。
    """
    xf, yf = lonlat_to_tile(lon, lat, ZOOM)
    tx, ty = int(math.floor(xf)), int(math.floor(yf))
    h = decode_terrarium(fetch_tile(ZOOM, tx, ty))
    # 场景中心在 tile 内的像素位置
    px = (xf - tx) * TILE
    py = (yf - ty) * TILE
    # 该纬度的缩放：Mercator 在纬度 phi 处放大 1/cos(phi)
    mpp = MPP_EQ * math.cos(math.radians(lat))
    half_px = int(round(half_km * 1000.0 / mpp))
    half_px = max(8, min(half_px, TILE // 2 - 2))
    cx, cy = int(round(px)), int(round(py))
    x0, x1 = max(0, cx - half_px), min(TILE, cx + half_px)
    y0, y1 = max(0, cy - half_px), min(TILE, cy + half_px)
    win = h[y0:y1, x0:x1]
    if win.shape[0] < 12 or win.shape[1] < 12:
        return None
    m = terrain_metrics(win, mpp)
    m["n_px"] = int(win.size)
    m["mpp"] = round(mpp, 3)
    return m


# --------------------------------------------------------------- 主流程
def load_meta():
    out = {}
    with open(META, encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if not row or not row[0].strip():
                continue
            sid = row[0].strip()
            if sid in out:
                continue
            try:
                out[sid] = (float(row[2 + 1]) if False else None)
            except Exception:  # noqa: BLE001
                pass
    # 上面按位置取列不可靠；改用 stratify_floor 的既有解析
    return out


def scenes():
    """复用 stratify_floor 的 manifest + metadata（单一真源）。"""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import stratify_floor as SF
    man = SF.build_manifest()
    meta = SF.load_metadata()
    rows = []
    for item in man:
        cov = item["cov"]
        md = meta.get(cov)
        if not md or md.get("lat") is None:
            continue
        rows.append({"cov": cov, "lat": float(md["lat"]), "lon": float(md["lon"])})
    return rows


def one(job):
    r = job
    try:
        m = scene_dem(r["lat"], r["lon"])
        if m is None:
            return {"cov": r["cov"], "error": "window too small"}
        m.update({"cov": r["cov"], "lat": r["lat"], "lon": r["lon"], "z": ZOOM})
        return m
    except Exception as e:  # noqa: BLE001
        return {"cov": r["cov"], "error": str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0, help="只跑前 N 个，用于验证")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="本次最多处理多少个（分段推进用）")
    ap.add_argument("--no-resume", action="store_true", help="忽略已完成的 cov，全部重跑")
    a = ap.parse_args()

    rows = scenes()
    if a.probe:
        # 取前 N 个（覆盖四个战役）
        by = {}
        for r in rows:
            g = r["cov"].split("-")[0]
            by.setdefault(g, []).append(r)
        pick = []
        per = max(1, a.probe // max(1, len(by)))
        for g, lst in by.items():
            pick += lst[:per]
        pick += rows[: a.probe - len(pick)]
        rows = pick[: a.probe]

    # ---- 增量续跑：跳过已成功写入的 cov（沙箱单次 CPU 有上限，须分段推进）----
    done_cov = set()
    if not a.no_resume and os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done_cov.add(r["cov"])
    todo = [r for r in rows if r["cov"] not in done_cov]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[info] scenes={len(rows)}  already={len(done_cov)}  todo={len(todo)}  "
          f"workers={a.workers} zoom={ZOOM} mpp_eq={MPP_EQ:.1f}")

    from concurrent.futures import ThreadPoolExecutor
    mode = "a" if (done_cov and not a.no_resume) else "w"
    ok = err = 0
    with open(a.out, mode, encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(one, todo)):
            if "error" in rec:
                err += 1
            else:
                ok += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                f.flush()
                print(f"  {i + 1}/{len(todo)}  ok={ok} err={err}", flush=True)
    print(f"[ok] wrote {a.out}  (+{ok} ok, {err} err)")


if __name__ == "__main__":
    main()
