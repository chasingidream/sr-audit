# -*- coding: utf-8 -*-
"""从 HuggingFace TACO 分片提取 SEN2VENµS 指定站点 -> 现有 loader 兼容布局。

背景：Zenodo 403 限流（SO2.zip 790MB 拿不到），改走 HF 镜像
tacofoundation/sen2venus（TACO v1 格式，6 分片 104GB）。TACO 分片内嵌
索引，可按字节偏移 Range 精准拉取单样本，无需下载全量。

管线：
  1. tacoreader.v1.load(hf-mirror 分片 URL x6) -> 合并索引（13 万行，仅索引块流量）
  2. 过滤 region == 目标站点
  3. 每行 read 子样本（LR=S2 10 波段 128x128 / HR=VENµS 8 波段 256x256）
     解析 internal:subfile (/vsisubfile/{off}_{len},<url>) 自控 requests Range 拉字节
  4. 切 b2b3b4b8 波段（idx 0,1,2,6），uint16 写 GeoTIFF（内存）
  5. 落盘二级 zip：{root}/{SITE}/{id}_b2b3b4b8.zip
     内含 SO2/{id}_b2b3b4b8/{05m,10m}/{id}_b2b3b4b8_{res}.tif
     -> datasets_external.Sen2VenusDataset(dir 模式) 直接可用

用法（docx venv，torch-free）：
  python code/s2v_hf_extract.py --site SO2 --workers 8
输出：data/external/sen2venus_hf/{SITE}/ 与提取报告 json
"""
import argparse
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHARDS = [f"https://hf-mirror.com/datasets/tacofoundation/sen2venus/resolve/main/"
          f"sen2venus.{i:04d}.part.taco" for i in range(6)]

BAND_IDX = [0, 1, 2, 6]          # b2,b3,b4,b8
SUBFILE_RE = re.compile(r"/vsisubfile/(\d+)_(\d+),(.+)", re.S)

# 连接复用 + 签名 URL 缓存：hf-mirror 每次 302 → cas-bridge 的往返是主要瓶颈。
# 预取一次签名 URL（1h 有效）后直接打 CDN；过期/失效时自动刷新。
_SESSION = None
_SIGNED = {}          # shard_url -> signed_url
_SIGN_LOCK = None


def _sess():
    global _SESSION, _SIGN_LOCK
    if _SESSION is None:
        import requests
        import threading
        _SESSION = requests.Session()
        try:
            from requests.adapters import HTTPAdapter
            _SESSION.mount("https://", HTTPAdapter(pool_maxsize=24, pool_connections=24))
        except Exception:  # noqa: BLE001
            pass
        _SIGN_LOCK = threading.Lock()
    return _SESSION


def _resolve(url: str) -> str:
    """hf-mirror resolve URL -> cas-bridge 签名 URL（缓存，失败重取）。"""
    import threading
    lock = _SIGN_LOCK if _SIGN_LOCK is not None else threading.Lock()
    with lock:
        signed = _SIGNED.get(url)
    if signed:
        return signed
    r = _sess().head(url, allow_redirects=True, timeout=60)
    final = r.url
    if "hf-mirror.com" in final and "xethub" not in final:
        raise RuntimeError(f"签名 URL 解析异常: {final[:100]}")
    with lock:
        _SIGNED[url] = final
    return final


def fetch_subfile(subfile: str, retries: int = 5):
    """解析 /vsisubfile/{off}_{len},{url}，Range 拉取原始 GeoTIFF 字节。"""
    m = SUBFILE_RE.match(subfile)
    if not m:
        raise ValueError(f"无法解析 subfile: {subfile[:120]}")
    off, length, url = int(m.group(1)), int(m.group(2)), m.group(3)
    url = re.sub(r"^/vsicurl/|^/vsicurl%", "", url)   # 剥离嵌套 GDAL 前缀
    if not url.startswith("http"):
        raise ValueError(f"subfile URL 非 http: {url[:120]}")
    last = None
    for a in range(retries):
        try:
            target = _resolve(url)
            r = _sess().get(target, headers={"Range": f"bytes={off}-{off + length - 1}"},
                            timeout=120, allow_redirects=False)
            if r.status_code in (200, 206) and len(r.content) >= length:
                return r.content[:length]
            last = f"HTTP={r.status_code} len={len(r.content)}"
            if r.status_code in (403, 404):     # 签名过期 -> 强制刷新
                with (_SIGN_LOCK if _SIGN_LOCK else __import__("threading").Lock()):
                    _SIGNED.pop(url, None)
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            with (_SIGN_LOCK if _SIGN_LOCK else __import__("threading").Lock()):
                _SIGNED.pop(url, None)
        time.sleep(min(2 * (a + 1), 8))
    raise RuntimeError(f"Range 拉取失败 ({last}): {url[-60:]} offset={off}")


def read_tif_bands(data: bytes, band_idx):
    """字节 -> rasterio -> (C,H,W) uint16，切指定波段。"""
    import rasterio
    with rasterio.open(io.BytesIO(data)) as src:
        arr = src.read()  # (bands,H,W)
    return arr[band_idx, ...].astype("uint16")


def tif_bytes(arr, transform, crs):
    """(C,H,W) uint16 -> zstd GeoTIFF 字节（与 v2 一致：zstd/predictor 2）。"""
    import rasterio
    from rasterio.io import MemoryFile
    C, H, W = arr.shape
    profile = dict(driver="GTiff", height=H, width=W, count=C, dtype="uint16",
                   crs=crs, transform=transform, compress="zstd",
                   predictor=2, zstd_level=13, interleave="band")
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(arr)
        return mem.read()


def extract_site(site: str, root: str, workers: int, limit: int):
    import tacoreader.v1 as tacoreader
    from affine import Affine

    out_dir = os.path.join(root, site)
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(REPO, "experiments", f"s2v_hf_extract_{site}.json")

    print(f"[1/4] load TACO 索引 x{len(SHARDS)}（逐分片+重试） ...", flush=True)
    t0 = time.time()
    import pandas as pd
    shards = []          # [(dataset_obj, local_df)]
    for si, url in enumerate(SHARDS):
        for attempt in range(4):
            try:
                d = tacoreader.load(url)
                ddf = d.data if hasattr(d, "data") else d
                shards.append((d, ddf))
                print(f"      shard {si}: {ddf.shape[0]} 行", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"      shard {si} attempt {attempt} 失败: {e!r}", flush=True)
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError(f"分片 {si} 索引加载重试耗尽: {url}")
    df = pd.concat([d for _, d in shards], ignore_index=True)
    print(f"      索引共 {df.shape[0]} 行, {time.time()-t0:.0f}s", flush=True)

    col_region = "region" if "region" in df.columns else None
    if col_region is None:
        raise RuntimeError("索引无 region 列")
    # 按分片局部过滤，保证 read() 的局部行号正确
    # 注意：ddf.index 是全局 label（非 0-based 位置），read() 走 iloc 必须用位置
    import numpy as np
    targets = []         # [(dataset_obj, local_row_position, tid)]
    for d, ddf in shards:
        if col_region not in ddf.columns:
            continue
        mask = (ddf[col_region] == site).to_numpy()
        for pos in np.flatnonzero(mask):
            targets.append((d, int(pos), str(ddf.iloc[pos]["tortilla:id"])))
    ids = [t[2] for t in targets]
    print(f"[2/4] {site}: {len(ids)} patches", flush=True)
    if limit:
        targets = targets[:limit]

    # 逐 patch：read 子样本 -> 拉取 -> 写二级 zip（幂等：已落盘的跳过）
    def work(item):
        d, i, tid = item
        z_lr = os.path.join(out_dir, f"{tid}_b2b3b4b8_10m.zip")
        z_hr = os.path.join(out_dir, f"{tid}_b2b3b4b8_05m.zip")
        if os.path.exists(z_lr) and os.path.exists(z_hr) \
                and os.path.getsize(z_lr) > 1000 and os.path.getsize(z_hr) > 1000:
            return tid, True, "skip"
        subs = d.read(i)
        lr_rec = hr_rec = None
        for _, s in subs.iterrows():
            arr_meta = (s["internal:subfile"], tuple(s["stac:geotransform"]), s["stac:crs"])
            shape = tuple(s["stac:raster_shape"])
            if shape[0] == 128:            # S2 10m
                lr_rec = arr_meta
            elif shape[0] == 256:          # VENµS 5m
                hr_rec = arr_meta
        if lr_rec is None or hr_rec is None:
            return tid, False, "子样本缺失 (LR/HR 未识别)"

        try:
            lr_raw = fetch_subfile(lr_rec[0])
            hr_raw = fetch_subfile(hr_rec[0])
            lr = read_tif_bands(lr_raw, BAND_IDX)
            hr = read_tif_bands(hr_raw, BAND_IDX)
            import numpy as np
            # 质量哨兵：全 0 / nodata(65525) 占比过高则记录
            def bad_ratio(a):
                return float(((a == 0) | (a >= 65525)).mean())
            b_lr, b_hr = bad_ratio(lr), bad_ratio(hr)
            lr_t = tif_bytes(lr, Affine.from_gdal(*lr_rec[1]), lr_rec[2])
            hr_t = tif_bytes(hr, Affine.from_gdal(*hr_rec[1]), hr_rec[2])
            # 布局对齐 Zenodo v2：每个 res 一个二级 zip（loader dir 模式按 zip 名过滤 res）
            import zipfile
            with zipfile.ZipFile(os.path.join(out_dir, f"{tid}_b2b3b4b8_10m.zip"),
                                 "w", zipfile.ZIP_STORED) as z:
                z.writestr(f"{site}/{tid}_b2b3b4b8/10m/{tid}_b2b3b4b8_10m.tif", lr_t)
            with zipfile.ZipFile(os.path.join(out_dir, f"{tid}_b2b3b4b8_05m.zip"),
                                 "w", zipfile.ZIP_STORED) as z:
                z.writestr(f"{site}/{tid}_b2b3b4b8/05m/{tid}_b2b3b4b8_05m.tif", hr_t)
            return tid, True, dict(bad_lr=b_lr, bad_hr=b_hr,
                                   lr_shape=list(lr.shape), hr_shape=list(hr.shape))
        except Exception as e:  # noqa: BLE001
            return tid, False, repr(e)

    print(f"[3/4] 并行拉取 {len(targets)} patches x2 (workers={workers}) ...", flush=True)
    ok, fail = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, item): item[2] for item in targets}
        for k, fut in enumerate(as_completed(futs), 1):
            tid, good, info = fut.result()
            (ok if good else fail).append((tid, info))
            if k % 25 == 0 or k == len(targets):
                print(f"      {k}/{len(targets)}  ok={len(ok)} fail={len(fail)}  "
                      f"{time.time()-t0:.0f}s", flush=True)

    report = dict(
        site=site, source="hf-mirror tacofoundation/sen2venus TACO v1",
        total=len(targets), ok=len(ok), fail=len(fail),
        failed=[(t, str(i)) for t, i in fail][:50],
        bad_lr=[round(i["bad_lr"], 4) for _, i in ok if isinstance(i, dict)][:10],
        out_dir=out_dir,
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[4/4] 完成: ok={len(ok)} fail={len(fail)}  报告 {report_path}", flush=True)
    return len(fail) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="SO2")
    ap.add_argument("--root", default=os.path.join(REPO, "data", "external", "sen2venus_hf"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="调试用，0=全量")
    args = ap.parse_args()
    good = extract_site(args.site, args.root, args.workers, args.limit)
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
