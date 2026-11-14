# -*- coding: utf-8 -*-
"""SEN2VENµS 站点直读提取器（不经 tacoreader）。

背景：`tacoreader` 2.4.21 已把 `.taco` 判为 legacy 并直接抛错，因此老的
`s2v_hf_extract.py`（走 tacoreader.load）在当前环境不可用。本脚本改用
**字节区间直读**：

  外层分片 (.part.taco, magic 'WX')
    [0:2]  magic | [2:10] footer_off(u64 LE) | [10:18] footer_len
    footer = Parquet 表，每行一个样本，列含 region / tortilla:id /
             tortilla:offset / tortilla:length  （= 嵌套容器绝对起点与长度）
  嵌套容器 (magic '#y')
    同一 18 字节头布局；其 footer 为 Parquet，列出资产 lr / hr，
    资产 tortilla:offset / tortilla:length **相对嵌套容器起点**
    （已核对：hr 相对区间终点 == 嵌套 footer 起点）。

输出布局与 datasets_external.Sen2VenusDataset(dir 模式) 兼容：
  {root}/{SITE}/{tid}_b2b3b4b8_10m.zip  -> {SITE}/{tid}_b2b3b4b8/10m/{tid}_b2b3b4b8_10m.tif
  {root}/{SITE}/{tid}_b2b3b4b8_05m.zip  -> {SITE}/{tid}_b2b3b4b8/05m/{tid}_b2b3b4b8_05m.tif

用法：
  python code/s2v_extract_direct.py --site NARYN --limit 300
"""
import argparse
import io
import json
import os
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
import rasterio
import requests
from affine import Affine
from rasterio.io import MemoryFile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://hf-mirror.com/datasets/tacofoundation/sen2venus/resolve/main/"
SHARDS = [f"sen2venus.{i:04d}.part.taco" for i in range(6)]
PROX = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
CACHE = os.path.join(REPO, "experiments", "s2v_cache")
BAND_IDX = [0, 1, 2, 6]          # b2, b3, b4, b8

_S = None
_LOCK = threading.Lock()


def sess():
    global _S
    if _S is None:
        _S = requests.Session()
        try:
            from requests.adapters import HTTPAdapter
            _S.mount("https://", HTTPAdapter(pool_maxsize=32, pool_connections=32))
        except Exception:  # noqa: BLE001
            pass
    return _S


def rng(url, start, end, tries=10):
    """HTTP Range 取 [start, end]（含端点），带重试与退避。"""
    last = None
    for k in range(tries):
        try:
            r = sess().get(url, headers={"Range": f"bytes={start}-{end}"},
                           proxies=PROX, timeout=180, allow_redirects=True)
            if r.status_code in (200, 206) and len(r.content) >= (end - start + 1):
                return r.content[: end - start + 1]
            last = f"HTTP={r.status_code} len={len(r.content)}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(min(1.5 * (k + 1), 8))
    raise RuntimeError(f"Range 失败 ({last}) {url[-50:]} [{start},{end}]")


def footer_of(url, cache_key=None):
    """读容器 footer -> pyarrow Table（带磁盘缓存）。"""
    if cache_key:
        cp = os.path.join(CACHE, cache_key)
        if os.path.exists(cp):
            return pq.read_table(cp)
    h = rng(url, 0, 17)
    magic = h[:2]
    if magic not in (b"WX", b"#y"):
        raise ValueError(f"bad magic {magic!r} for {url[-50:]}")
    fo = int.from_bytes(h[2:10], "little")
    fl = int.from_bytes(h[10:18], "little")
    blob = rng(url, fo, fo + fl - 1)
    if len(blob) < fl:
        raise RuntimeError(f"footer 短读 {len(blob)}<{fl}")
    t = pq.read_table(io.BytesIO(blob))
    if cache_key:
        os.makedirs(CACHE, exist_ok=True)
        pq.write_table(t, os.path.join(CACHE, cache_key))
    return t


def read_tif_bands(data, band_idx):
    with rasterio.open(io.BytesIO(data)) as src:
        arr = src.read()
    return arr[band_idx, ...].astype("uint16")


def tif_bytes(arr, transform, crs):
    C, H, W = arr.shape
    profile = dict(driver="GTiff", height=H, width=W, count=C, dtype="uint16",
                   crs=crs, transform=transform, compress="zstd",
                   predictor=2, zstd_level=13, interleave="band")
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(arr)
        return mem.read()


def index_site(site):
    """扫描 6 个分片 footer，返回 [(shard_url, row, tid)]。"""
    out = []
    for sh in SHARDS:
        url = BASE + sh
        t = footer_of(url, cache_key=f"footer_{sh}.parquet")
        if "region" not in t.column_names:
            continue
        reg = [str(v) for v in t.column("region").to_pylist()]
        ids = [str(v) for v in t.column("tortilla:id").to_pylist()]
        for r, v in enumerate(reg):
            if v == site:
                out.append((url, r, ids[r]))
    return out


def sample_indices(n, limit):
    """分层均匀抽样（可复现），与 sen2naipv2_audit 同一约定。"""
    if not limit or limit >= n:
        return list(range(n))
    return sorted({int(round((k + 0.5) * n / limit - 0.5)) for k in range(limit)})


def extract_one(url, row, tid, out_dir, meta):
    z_lr = os.path.join(out_dir, f"{tid}_b2b3b4b8_10m.zip")
    z_hr = os.path.join(out_dir, f"{tid}_b2b3b4b8_05m.zip")
    if (os.path.exists(z_lr) and os.path.exists(z_hr)
            and os.path.getsize(z_lr) > 1000 and os.path.getsize(z_hr) > 1000):
        return tid, True, "skip"
    # 外层行 -> 嵌套容器绝对起点
    t = footer_of(url, cache_key=f"footer_{os.path.basename(url)}.parquet")
    c_start = int(t.column("tortilla:offset")[row].as_py())
    # 嵌套容器 footer
    nh = rng(url, c_start, c_start + 17)
    if nh[:2] != b"#y":
        return tid, False, f"nested magic {nh[:2]!r}"
    nfo = int.from_bytes(nh[2:10], "little")
    nfl = int.from_bytes(nh[10:18], "little")
    nt = pq.read_table(io.BytesIO(rng(url, c_start + nfo, c_start + nfo + nfl - 1)))
    nd = nt.to_pydict()
    lr_rec = hr_rec = None
    for j in range(nt.num_rows):
        aid = str(nd["tortilla:id"][j])
        rel = int(nd["tortilla:offset"][j])
        ln = int(nd["tortilla:length"][j])
        shape = tuple(nd["stac:raster_shape"][j])
        rec = (c_start + rel, ln, tuple(nd["stac:geotransform"][j]), nd["stac:crs"][j], shape)
        if aid == "lr" or shape[0] == 128:
            lr_rec = rec
        elif aid == "hr" or shape[0] == 256:
            hr_rec = rec
    if lr_rec is None or hr_rec is None:
        return tid, False, "lr/hr 未识别"
    try:
        lr_raw = rng(url, lr_rec[0], lr_rec[0] + lr_rec[1] - 1)
        hr_raw = rng(url, hr_rec[0], hr_rec[0] + hr_rec[1] - 1)
        lr = read_tif_bands(lr_raw, BAND_IDX)
        hr = read_tif_bands(hr_raw, BAND_IDX)
        lr_t = tif_bytes(lr, Affine.from_gdal(*lr_rec[2]), lr_rec[3])
        hr_t = tif_bytes(hr, Affine.from_gdal(*hr_rec[2]), hr_rec[3])
        with zipfile.ZipFile(z_lr, "w", zipfile.ZIP_STORED) as z:
            z.writestr(f"{os.path.basename(out_dir)}/{tid}_b2b3b4b8/10m/{tid}_b2b3b4b8_10m.tif", lr_t)
        with zipfile.ZipFile(z_hr, "w", zipfile.ZIP_STORED) as z:
            z.writestr(f"{os.path.basename(out_dir)}/{tid}_b2b3b4b8/05m/{tid}_b2b3b4b8_05m.tif", hr_t)
        with _LOCK:
            meta[tid] = dict(crs=lr_rec[3], shape_lr=list(lr_rec[4]), shape_hr=list(hr_rec[4]),
                             gt=lr_rec[2])
        return tid, True, None
    except Exception as e:  # noqa: BLE001
        return tid, False, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--root", default=os.path.join(REPO, "data", "external", "sen2venus_hf"))
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    idx = index_site(args.site)
    print(f"[1/3] {args.site}: 归档 {len(idx)} 个样本", flush=True)
    if not idx:
        sys.exit("站点无样本")
    pos = sample_indices(len(idx), args.limit)
    rows = [idx[p] for p in pos]
    print(f"      抽样 {len(rows)} 个（分层均匀）", flush=True)

    out_dir = os.path.join(args.root, args.site)
    os.makedirs(out_dir, exist_ok=True)
    meta = {}
    ok, fail = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, u, r, t, out_dir, meta): t for u, r, t in rows}
        for k, f in enumerate(as_completed(futs), 1):
            tid, good, info = f.result()
            (ok if good else fail).append((tid, info))
            if k % 25 == 0 or k == len(rows):
                print(f"      {k}/{len(rows)} ok={len(ok)} fail={len(fail)} {time.time()-t0:.0f}s",
                      flush=True)

    rep = dict(site=args.site, archive_n=len(idx), sampled=len(rows),
               ok=len(ok), fail=len(fail),
               failed=[(t, str(i)) for t, i in fail][:40],
               crs=meta[ok[0][0]]["crs"] if ok and ok[0][0] in meta else None,
               out_dir=out_dir)
    rp = os.path.join(REPO, "experiments", f"s2v_extract_{args.site}.json")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    json.dump(rep, open(rp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[3/3] ok={len(ok)} fail={len(fail)} 报告 {rp}", flush=True)
    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
