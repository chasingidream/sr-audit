#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sen2naipv2_probe.py —— 直读 SEN2NAIPv2 的 .taco 容器（不走 tacoreader）。

背景
----
`tacoreader` 2.4.21 已把 `.taco` 判为 legacy 并直接抛错（v2 只认 `.tacozip`/`.tacocat`），
但 formats 本身是公开且简单的（见 `tacoreader/v1/TortillaDataFrame.partial_load_file`）：

    [0:2]    magic  b"WX"（TACO 🌮）或 b"#y"（Tortilla 🫓）
    [2:10]   footer_offset  (uint64 LE，相对文件起点)
    [10:18]  footer_length  (uint64 LE)
    ...      内嵌数据块
    [FO:FO+FL]  footer = Parquet 表（每行一个样本）

每行的资产以 `/vsisubfile/{offset}_{length},{path}` 形式内嵌在同一文件里，
所以 **只需按 Range 取 footer + 目标样本的字节**，不必下载整个 9.7 GB。

关键偏移语义（实测确认，见 --diag）
----------------------------------
* 外层 parquet 行的 `tortilla:offset` 是**嵌套容器在文件中的绝对起点**。
* 嵌套容器自己的 footer_offset 是**相对容器起点**的 ⇒ 绝对位置 = container_start + nfo。
* 嵌套 footer 里各资产（lr/hr）的 `tortilla:offset` 是**相对嵌套容器起点**的
  ⇒ 绝对位置 = container_start + asset_off。

本脚本做三件事：
  1. 探针：读 header + footer，打印 schema / 行数 / 前几行的关键字段。
  2. `--diag idx`：打印某样本内外层偏移逐一验证（TIFF magic 断言）。
  3. `--decode idx`：拉取该样本的 lr/hr 两个 GTiff，用 rasterio 解到数组并打印统计。
  4. `--fetch idx[,idx...]`：把指定样本的原始字节落盘到 experiments/sen2sr_audit/sen2naip/。

用法（Windows，需显式指定可用代理 7890；env 里的 9840 是死端口）：
  python code/sen2naipv2_probe.py
  python code/sen2naipv2_probe.py --diag 0
  python code/sen2naipv2_probe.py --decode 0
  python code/sen2naipv2_probe.py --fetch 0,1,2
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "experiments", "sen2sr_audit", "sen2naip")

REPO = "tacofoundation/SEN2NAIPv2"
DEFAULT_FILE = "sen2naipv2-crosssensor.taco"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{DEFAULT_FILE}"

# env 里的 http_proxy 指向一个未监听的端口；本机可用代理在 7890（clash 内核）。
PROXIES = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}


def _range_get(url, start, end, timeout=120, tries=10):
    """取 [start, end] 闭区间的字节。跟随 302 到 CDN，保留 Range。

    代理（clash 内核）会间歇性 SSL EOF / TLS 握手失败，故重试次数取大并加退避。
    """
    import time
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                             proxies=PROXIES, timeout=timeout, allow_redirects=True)
            if r.status_code in (200, 206) and len(r.content) == (end - start + 1):
                return r.content
            last = f"http={r.status_code} got={len(r.content)} want={end-start+1}"
        except Exception as e:                      # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        print(f"  [retry {k+1}/{tries}] {last}", file=sys.stderr)
        time.sleep(min(1.5 * (k + 1), 8))
    raise RuntimeError(f"range get failed: {last}")


def _parse_container_header(blob, label=""):
    """从容器字节里解出 (magic, footer_offset, footer_length)。"""
    mb = blob[:2]
    if mb not in (b"WX", b"#y"):
        raise ValueError(f"{label} not a TACO/Tortilla container: magic={mb!r}")
    fo = int.from_bytes(blob[2:10], "little")
    fl = int.from_bytes(blob[10:18], "little")
    return mb, fo, fl


def read_outer_footer(verbose=True):
    import pyarrow.parquet as pq
    head = _range_get(URL, 0, 17)
    mb, fo, fl = _parse_container_header(head, "outer")
    if verbose:
        print(f"[outer ] magic={mb!r} footer_offset={fo:,} footer_length={fl:,}")
    raw = _range_get(URL, fo, fo + fl - 1)
    tbl = pq.read_table(io.BytesIO(raw))
    if verbose:
        print(f"[outer ] parquet {len(raw):,} B -> {tbl.num_rows:,} rows x {tbl.num_columns} cols")
    return fo, tbl


def sample_assets(df, i, outer_off):
    """返回 {name: (abs_offset, length, path)}，已把资产偏移换算成文件绝对位置。

    实测语义：outer 行的 `tortilla:offset` = 嵌套容器绝对起点；
    嵌套 footer 里资产的 `tortilla:offset` 相对嵌套容器起点。
    """
    import pyarrow.parquet as pq
    row = df.iloc[i]
    c_start = int(row["tortilla:offset"])          # 嵌套容器绝对起点
    c_len = int(row["tortilla:length"])
    blob = _range_get(URL, c_start, c_start + c_len - 1)
    mb, nfo, nfl = _parse_container_header(blob, f"nested[{i}]")
    nraw = blob[nfo:nfo + nfl]
    ntbl = pq.read_table(io.BytesIO(nraw)).to_pandas()
    assets = {}
    for _, a in ntbl.iterrows():
        name = str(a.get("tortilla:id", a.get("tortilla:name", "?")))   # "lr" / "hr"
        a_off = int(a["tortilla:offset"])          # 相对 nested 容器
        a_len = int(a["tortilla:length"])
        assets[name] = (c_start + a_off, a_len, str(a.get("tortilla:path", "")))
    return {"container_start": c_start, "container_len": c_len,
            "nested_magic": mb, "nested_footer": (nfo, nfl),
            "nested_cols": list(ntbl.columns), "assets": assets}


def load_gtiff(abs_off, length, div=None):
    """Range 拉取单个 GTiff 并用 rasterio 解到 numpy（rasterio 自带 zstd/压缩后端）。"""
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    raw = _range_get(URL, abs_off, abs_off + length - 1)
    with MemoryFile(raw) as mf:
        with mf.open() as ds:
            arr = ds.read().astype("float64")       # (bands, H, W)
            prof = dict(ds.profile)
            tr = ds.transform
    if div:
        arr = arr / div
    return arr, prof, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", default=None, help="逗号分隔的样本索引，落盘其内嵌字节")
    ap.add_argument("--diag", default=None, help="逗号分隔的样本索引，打印偏移逐一校验")
    ap.add_argument("--decode", default=None, help="逗号分隔的样本索引，rasterio 解到数组")
    args = ap.parse_args()

    fo, tbl = read_outer_footer()
    df = tbl.to_pandas()
    print("\n[columns]")
    for c in df.columns:
        print("   ", c)
    print("\n[first rows]")
    keep = [c for c in ["tortilla:file_format", "tortilla:offset", "tortilla:length",
                        "stac:raster_shape", "scale_factor"] if c in df.columns]
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"  row {i}: " + json.dumps({c: str(row[c])[:120] for c in keep},
                                          ensure_ascii=False))
    if "tortilla:file_format" in df.columns:
        print("\n[file_format counts]", df["tortilla:file_format"].value_counts().to_dict())

    if args.diag:
        for s in args.diag.split(","):
            i = int(s)
            info = sample_assets(df, i, fo)
            print(f"\n[diag] sample {i}")
            print(f"   nested container start={info['container_start']:,} "
                  f"len={info['container_len']:,} magic={info['nested_magic']!r}")
            print(f"   nested footer rel=({info['nested_footer'][0]:,},"
                  f"{info['nested_footer'][1]:,})  cols={info['nested_cols']}")
            for name, (ao, al, apth) in info["assets"].items():
                head = _range_get(URL, ao, ao + 8)
                ok = head[:4] in (b"II*\x00", b"MM\x00*")
                print(f"   asset {name:4s} abs_off={ao:,} len={al:,} "
                      f"head={head[:8]!r} tiff_ok={ok}  path={apth[-40:]}")

    if args.decode:
        for s in args.decode.split(","):
            i = int(s)
            info = sample_assets(df, i, fo)
            print(f"\n[decode] sample {i}")
            for name, (ao, al, _p) in info["assets"].items():
                arr, prof, tr = load_gtiff(ao, al)
                print(f"   {name:4s} shape={arr.shape} dtype->f64 "
                      f"min={arr.min():.2f} max={arr.max():.2f} mean={arr.mean():.3f} "
                      f"compress={prof.get('compress')} crs={prof.get('crs')}")
                print(f"        transform={tr}")

    if args.fetch:
        os.makedirs(OUT, exist_ok=True)
        for s in args.fetch.split(","):
            i = int(s)
            info = sample_assets(df, i, fo)
            print(f"\n[fetch] sample {i}  container_start={info['container_start']:,}")
            for name, (ao, al, _p) in info["assets"].items():
                blob = _range_get(URL, ao, ao + al - 1)
                ext = ".tif"
                p = os.path.join(OUT, f"sample_{i:05d}_{name}{ext}")
                with open(p, "wb") as f:
                    f.write(blob)
                print(f"   [saved] {p}  {len(blob):,} B  head={blob[:8]!r}")


if __name__ == "__main__":
    main()
