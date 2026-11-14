# -*- coding: utf-8 -*-
"""C-7：第二类独立静态量 —— ESA WorldCover 2021 土地覆盖组成。

为什么不是土壤质地
  SoilGrids 瓦片缺地理变换标签、命名规则难逆推；WCS 超时、zip 404。
  改用 **ESA WorldCover 2021**（10 m 全球土地覆盖，AWS S3 公开免认证）：
  它提供与影像强度纹理**不同的信息轴**（覆盖类别组成 vs 连续强度），
  且是独立发布的产品（2021 年）。

关键技术：只下载需要的 tile
  瓦片为 3°×3°、10 m、36000×36000 uint8，**tiled** 组织（TileWidth=TileLength=1024），
  1,296 个 tile/瓦片，Deflate 压缩，**整文件 12 MB**。
  全量下 1,144 个瓦片 = 13.7 GB（不可行）。
  但每景只需中心 ±0.9 km（≈180×180 像素）⇒ 命中 **1~4 个 tile**，
  压缩后每 tile 仅数 KB~数十 KB ⇒ 总计约 90 MB，可行。

  实现：`HttpReader`（HTTP Range）+ **手动解析 TIFF IFD**
  （tifffile 对该文件的惰性 tag 读取拿不到 TileOffsets，故自己解析字节布局）。

派生量（每景窗口内，面积占比 %）
  wc_tree / wc_shrub / wc_grass / wc_crop / wc_built / wc_bare
  wc_water / wc_wetland / wc_snow / wc_mangrove / wc_moss
  wc_nclass      出现的类别数
  wc_entropy     归一化 Shannon 多样性
  wc_dominant    最大占比类别
  wc_veg_frac / wc_water_frac / wc_built_frac   汇总

输出
  experiments/worldcover_2021.jsonl
"""
import argparse
import json
import math
import os
import struct
import sys
import threading
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pathc_models as M  # noqa: E402

URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
       "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
CACHE = os.path.join(M.EXP, "_wc_tiles")
OUT = os.path.join(M.EXP, "worldcover_2021.jsonl")

WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
              60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove",
              100: "moss"}
SUM_ONLY = ("wc_nclass", "wc_entropy", "wc_dominant", "wc_veg_frac",
            "wc_water_frac", "wc_built_frac")
DERIVED_KEYS = ["wc_" + v for v in WC_CLASSES.values()] + list(SUM_ONLY)

_TSIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8}
_TFMT = {1: "B", 3: "H", 4: "I"}


def wc_tile(lat, lon):
    """3° 瓦片名。约定由 tiepoint 实测确定（易错，务必按此）：

    N 瓦片名 = 覆盖区间的**南端**   (N03 -> lat0=6  -> 覆盖 3~6°N)
    S 瓦片名 = 覆盖区间的**南端绝对值** (S12 -> lat0=-9 -> 覆盖 12~9°S)
    E 瓦片名 = 覆盖区间的**西端**   (E006 -> lon0=6  -> 覆盖 6~9°E)
    W 瓦片名 = 覆盖区间的**东端绝对值** (W060 -> lon0=-60 -> 覆盖 60~57°W)

    ⚠ 对负值必须用 ceil 而非 floor：lon=-55.27 属于 57~54°W -> W057，
      用 floor 会得到 W054（覆盖 54~51°W）而完全取不到数据。
    """
    if lat >= 0:
        ns, lband = "N", int(math.floor(lat / 3.0) * 3)
    else:
        ns, lband = "S", int(math.ceil(abs(lat) / 3.0) * 3)
    if lon >= 0:
        ew, oband = "E", int(math.floor(lon / 3.0) * 3)
    else:
        ew, oband = "W", int(math.ceil(abs(lon) / 3.0) * 3)
    if lband > 87 or oband > 177:
        return None
    return f"{ns}{lband:02d}{ew}{oband:03d}"


class HttpReader:
    """HTTP Range 读取器 + 瓦片 TIFF 头解析（手动，不用 tifffile 的惰性 tag）。"""

    def __init__(self, url, timeout=60):
        self.url = url
        self.timeout = timeout
        self._lock = threading.Lock()
        self._n = None

    def size(self):
        if self._n is None:
            req = urllib.request.Request(self.url, method="HEAD")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self._n = int(r.headers.get("Content-Length", 0))
        return self._n

    def rng(self, a, b, retries=4):
        """读 [a, b)；分块补足（Range 偶发返回短读）。"""
        a = max(0, int(a))
        b = min(int(b), self.size())
        if a >= b:
            return b""
        out = bytearray()
        # 大块（tile 数据）分 256 KB 片，小块一次读完
        step = 262144 if (b - a) > 262144 else (b - a)
        p = a
        while p < b and len(out) < (b - a):
            q = min(p + step, b)
            got = None
            for _ in range(retries):
                try:
                    req = urllib.request.Request(
                        self.url, headers={"Range": f"bytes={p}-{q - 1}"})
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        got = r.read()
                    break
                except Exception:  # noqa: BLE001
                    got = None
                    continue
            if not got:
                break
            out += got
            p += len(got)
            step = 262144
        return bytes(out)

    def tags(self):
        """解析 IFD，返回 {tag: int 或 (arr_offset, arr_bytes)}。"""
        head = self.rng(0, 8)
        if len(head) < 8:
            return {}, "<", 0
        bo = "<" if head[:2] == b"II" else ">"
        if struct.unpack(bo + "H", head[2:4])[0] != 42:
            return {}, bo, 0
        ifd = struct.unpack(bo + "I", head[4:8])[0]
        two = self.rng(ifd, ifd + 2)
        n = struct.unpack(bo + "H", two[:2])[0]
        raw = self.rng(ifd + 2, ifd + 2 + n * 12 + 4)
        out = {}
        for i in range(n):
            b = raw[i * 12:i * 12 + 12]
            if len(b) < 12:
                break
            tag, typ, cnt = struct.unpack(bo + "HHI", b[:8])
            val = b[8:12]
            sz = _TSIZE.get(typ, 1) * cnt
            if sz <= 4 and typ in _TFMT:
                out[tag] = struct.unpack(bo + _TFMT[typ] * cnt, val[:sz])
            else:
                out[tag] = (struct.unpack(bo + "I", val)[0], sz)
        return out, bo, n

    def array(self, off, sz, bo, dtype="I"):
        raw = self.rng(off, off + sz)
        if len(raw) < sz:
            raw += b"\x00" * (sz - len(raw))
        if dtype == "I":
            return np.frombuffer(raw, dtype=bo + "u4", count=sz // 4)
        return np.frombuffer(raw, dtype=bo + "f8", count=sz // 8)


# ⚠ 只缓存**成功**解析的瓦片布局。缓存失败结果会造成"一次网络抖动 →
# 该瓦片所有场景永久失败"的假性缺测（实测 213/1195 景如此）。
_META = {}
_META_LOCK = threading.Lock()


def tile_meta(tile, timeout=60):
    """取（并缓存）瓦片的 tile 布局。"""
    with _META_LOCK:
        if tile in _META:
            return _META[tile]
    hr = HttpReader(URL.format(tile=tile), timeout=timeout)
    try:
        tags, bo, _ = hr.tags()
        if not tags:
            return None          # ⚠ 不缓存失败（见 _META 注释）
        W, H = int(tags[256][0]), int(tags[257][0])
        tw, tl = int(tags[322][0]), int(tags[323][0])
        toff, tsz = tags[324]
        tbc, bsz = tags[325]
        offs = hr.array(toff, tsz, bo, "I")
        cnts = hr.array(tbc, bsz, bo, "I")
        # tiepoint（tag 33922, type 12 = double, 6 个）
        tpoff, tpsz = tags[33922]
        tpv = struct.unpack(bo + "6d", hr.rng(tpoff, tpoff + 48))
        meta = {"W": W, "H": H, "tw": tw, "tl": tl,
                "cols": int(np.ceil(W / tw)), "rows": int(np.ceil(H / tl)),
                "lon0": float(tpv[3]), "lat0": float(tpv[4]),
                "comp": int(tags[259][0]), "spp": int(tags[277][0]),
                "bits": int(tags[258][0])}
        with _META_LOCK:
            _META[tile] = (meta, offs, cnts)
    except Exception:  # noqa: BLE001
        return None              # ⚠ 不缓存失败
    return _META.get(tile)


def read_window(tile, lat, lon, half_km=0.9, timeout=60):
    """读瓦片内以 (lat, lon) 为中心、半径 half_km 的窗口（uint8 类别码）。"""
    import imagecodecs
    got = tile_meta(tile, timeout=timeout)
    if got is None:
        return None
    meta, offs, cnts = got
    H, W, tw, tl = meta["H"], meta["W"], meta["tw"], meta["tl"]
    res_lat, res_lon = 3.0 / H, 3.0 / W
    py = int((meta["lat0"] - lat) / res_lat)      # 自上而下
    px = int((lon - meta["lon0"]) / res_lon)
    half_px = max(8, int(round(half_km * 1000.0 / 10.0)))
    r0, r1 = max(0, py - half_px), min(H, py + half_px)
    c0, c1 = max(0, px - half_px), min(W, px + half_px)
    if r1 - r0 < 8 or c1 - c0 < 8:
        return None

    hr = HttpReader(URL.format(tile=tile), timeout=timeout)
    out = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    tr0, tr1 = r0 // tl, (r1 - 1) // tl
    tc0, tc1 = c0 // tw, (c1 - 1) // tw
    for trow in range(tr0, tr1 + 1):
        for tcol in range(tc0, tc1 + 1):
            ti = trow * meta["cols"] + tcol
            if ti >= len(offs):
                continue
            o, n = int(offs[ti]), int(cnts[ti])
            if n <= 0:
                continue
            raw = hr.rng(o, o + n)
            if len(raw) < n:
                continue
            try:
                dec = imagecodecs.zlib_decode(raw) if meta["comp"] in (8, 32946) \
                    else imagecodecs.lzw_decode(raw) if meta["comp"] in (5, 32773) \
                    else raw
            except Exception:  # noqa: BLE001
                continue
            a = np.frombuffer(dec, dtype=np.uint8)
            nh = min(tl, H - trow * tl)
            nw = min(tw, W - tcol * tw)
            if a.size < nh * nw:
                continue
            blk = a[:nh * nw].reshape(nh, nw)
            # 与输出窗口求交
            gy0 = trow * tl
            gx0 = tcol * tw
            ra, rb = max(r0, gy0), min(r1, gy0 + nh)
            ca, cb = max(c0, gx0), min(c1, gx0 + nw)
            if rb > ra and cb > ca:
                out[ra - r0:rb - r0, ca - c0:cb - c0] = blk[ra - gy0:rb - gy0,
                                                          ca - gx0:cb - gx0]
    return out


def metrics(win):
    if win is None or win.size == 0:
        return None
    v, cnt = np.unique(win, return_counts=True)
    tot = float(win.size)
    out, present, ent = {}, 0, 0.0
    for cls, c in zip(v.tolist(), cnt.tolist()):
        name = WC_CLASSES.get(int(cls))
        if name is None:
            continue                            # 0 = nodata（海洋等）
        p = c / tot
        out["wc_" + name] = round(100.0 * p, 3)
        present += 1
        if p > 0:
            ent -= p * math.log(p)
    for k in DERIVED_KEYS:
        out.setdefault(k, 0.0)
    out["wc_nclass"] = present
    out["wc_entropy"] = round(ent / math.log(len(WC_CLASSES)), 4) if present > 1 else 0.0
    comp = max((x for k, x in out.items()
                if k.startswith("wc_") and k not in SUM_ONLY), default=0.0)
    out["wc_dominant"] = round(comp, 3)
    out["wc_veg_frac"] = round(out["wc_tree"] + out["wc_shrub"] + out["wc_grass"] +
                               out["wc_crop"] + out["wc_mangrove"] +
                               out["wc_moss"], 3)
    out["wc_water_frac"] = round(out["wc_water"] + out["wc_wetland"], 3)
    out["wc_built_frac"] = round(out["wc_built"], 3)
    out["n_px"] = int(win.size)
    return out


def one(job):
    """⚠ 必须捕获**所有**异常：任一 worker 抛出未捕获异常会让 ThreadPoolExecutor
    的 map 迭代整体中断，已完成的结果全部丢失（实测 RemoteDisconnected）。"""
    r, half = job
    try:
        t = wc_tile(r["lat"], r["lon"])
        if not t:
            return {"cov": r["cov"], "error": "no tile"}
        key = os.path.join(CACHE, f"{t}_{r['lat']:.4f}_{r['lon']:.4f}_{half}.npy")
        if os.path.exists(key):
            win = np.load(key)
        else:
            win = read_window(t, r["lat"], r["lon"], half_km=half)
            if win is not None:
                os.makedirs(CACHE, exist_ok=True)
                try:
                    np.save(key, win)
                except OSError:
                    pass
        m = metrics(win)
        if m is None:
            return {"cov": r["cov"], "error": "empty window"}
        m.update({"cov": r["cov"], "lat": r["lat"], "lon": r["lon"], "tile": t})
        return m
    except Exception as e:  # noqa: BLE001
        return {"cov": r["cov"], "error": f"{type(e).__name__}: {str(e)[:80]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--half-km", type=float, default=0.9)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rows = M.load()
    if a.probe:
        by = {}
        for r in rows:
            by.setdefault(r["cov"].split("-")[0], []).append(r)
        pick = []
        per = max(1, a.probe // max(1, len(by)))
        for g, lst in by.items():
            pick += lst[:per]
        rows = pick[: a.probe]

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                done.add(r["cov"])
    todo = [r for r in rows if r["cov"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[info] scenes={len(rows)} done={len(done)} todo={len(todo)} "
          f"workers={a.workers} half_km={a.half_km}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    ok = err = 0
    mode = "a" if (done and len(todo) < len(rows)) else "w"
    with open(a.out, mode, encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(one, [(r, a.half_km) for r in todo])):
            if "error" in rec:
                err += 1
            else:
                ok += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                f.flush()
                print(f"  {i + 1}/{len(todo)}  ok={ok} err={err}", flush=True)
    print(f"[ok] wrote {a.out}  (+{ok} ok, {err} err)")


if __name__ == "__main__":
    main()
