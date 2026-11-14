#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""外部真实超分数据集加载器（P1 跨数据集审计用）。

与训练管线解耦，只提供"取配对 (LR, HR)"的最小接口，供 code/audit_datasets.py 使用。

1) ProbaVDataset  — ESA PROBA-V (Kelvins) MISR
   * 同传感器、多时相、3x（300m LR 128x128 -> 100m HR 384x384）
   * 16-bit PNG（14-bit 有效），自带 quality/status mask
   * 作用：隔离"多时相融合"在【无跨传感器辐射差】条件下的真实增益

2) Sen2VenusDataset — SEN2VENµS (Zenodo 14603764, v2 GeoTIFF)
   * 跨传感器、同日、2x（S2 10m 128x128 -> VENµS 5m 256x256）
   * int16 GeoTIFF，/10000 还原反射率
   * 作用：隔离"跨传感器辐射/几何失配"在【无多时相内容差】条件下的影响

两个数据集 + WorldStrat（跨传感器 + 多时相 + 6.63x）构成三轴解耦设计。
"""
from __future__ import annotations

import io
import glob
import os
import re
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

# ---------------------------------------------------------------- 通用工具

def _imread_any(path_or_bytes, is_bytes=False):
    """读 16-bit PNG / GeoTIFF，返回 float32 数组（保持原始 DN 尺度）。"""
    if is_bytes:
        data = path_or_bytes
    else:
        with open(path_or_bytes, "rb") as f:
            data = f.read()
    ext = b"" if is_bytes else os.path.splitext(str(path_or_bytes))[1].lower().encode()
    if ext == b".png":
        import cv2
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError("PNG 解码失败")
        return arr.astype(np.float32)
    import tifffile
    return tifffile.imread(io.BytesIO(data)).astype(np.float32)


def _first_channel(arr: np.ndarray) -> np.ndarray:
    """(H,W,C) -> 首通道；(H,W) 原样返回。"""
    return arr[:, :, 0] if arr.ndim == 3 else arr


# ---------------------------------------------------------------- PROBA-V

class ProbaVDataset:
    """PROBA-V Super-Resolution（Kelvins / Zenodo 6327426）。

    目录结构：
        probav_data/
          train/{RED,NIR}/imgsetXXXX/{LR*.png, QM*.png, HR.png, SM.png}
          test/{RED,NIR}/imgsetXXXX/{LR*.png, QM*.png}
          norm.csv
    """

    def __init__(self, root: str, band: str = "RED", split: str = "train",
                 div: float = 65535.0):
        self.root = root
        self.band = band
        self.split = split
        self.div = div
        base = os.path.join(root, split, band)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"PROBA-V 目录不存在: {base}")
        self.scenes = sorted(
            os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and d.startswith("imgset")
        )

    def __len__(self) -> int:
        return len(self.scenes)

    @property
    def scale(self) -> int:
        return 3

    def load(self, idx: int) -> Dict:
        """返回 dict: lrs(T,h,w) / qms(T,h,w) / hr(H,W) / sm(H,W) / name"""
        d = self.scenes[idx]
        lrs, qms = [], []
        for fn in sorted(os.listdir(d)):
            m = re.match(r"LR(\d+)\.png", fn)
            if not m:
                continue
            lrs.append(_first_channel(_imread_any(os.path.join(d, fn))))
            qm_p = os.path.join(d, f"QM{m.group(1)}.png")
            qms.append(_first_channel(_imread_any(qm_p)) if os.path.exists(qm_p)
                       else np.ones_like(lrs[-1]))
        hr = _first_channel(_imread_any(os.path.join(d, "HR.png")))
        sm_p = os.path.join(d, "SM.png")
        sm = _first_channel(_imread_any(sm_p)) if os.path.exists(sm_p) else np.ones_like(hr)
        # 统一形状约定：lrs (T,C,h,w) / hr (C,H,W) / sm (C,H,W)
        return {
            "lrs": np.stack(lrs).astype(np.float32)[:, None] / self.div,
            "qms": (np.stack(qms) > 0),
            "hr": (hr / self.div)[None],
            "sm": (sm > 0)[None],
            "name": os.path.basename(d),
        }


# ---------------------------------------------------------------- SEN2VENµS

class Sen2VenusDataset:
    """SEN2VENµS v2（GeoTIFF 版，Zenodo 14603764）。

    每个站点一个 zip：{SITE}.zip
      内含 {SITE}/{id}_{res}_{bands}.zip  （二级 zip）
      二级 zip 内：{bands}/{res}/{SITE} {idx} {date} {tile} {bands}_{res}.tif

    因嵌套 zip，这里直接在内存里解压读取，不落盘。
    """

    BAND_GROUPS = {
        "b2b3b4b8": ("05m", "10m", 2),   # HR 5m / LR 10m / scale
        "b5b6b7b8a": ("05m", "20m", 4),
    }

    def __init__(self, root: str, site: Optional[str] = None,
                 bands: str = "b2b3b4b8", div: float = 10000.0):
        self.root = root
        self.bands = bands
        self.div = div
        self.hr_res, self.lr_res, self.scale = self.BAND_GROUPS[bands]
        if site is None:
            zs = sorted(f for f in os.listdir(root) if f.endswith(".zip"))
            if not zs:
                raise FileNotFoundError(f"未找到站点 zip: {root}")
            site = zs[0].rsplit(".", 1)[0]
        self.site = site
        self.zpath = os.path.join(root, f"{site}.zip")
        self.sdir = os.path.join(root, site)
        if os.path.isdir(self.sdir):
            self.mode = "dir"                     # 已解压：直接读二级 zip 文件（更快）
        elif os.path.exists(self.zpath):
            self.mode = "zip"                     # 未解压：从外层 zip 里读二级 zip
        else:
            raise FileNotFoundError(f"{self.zpath} 或 {self.sdir} 都不存在")
        self.pairs = self._index()

    def _read_inner(self, inner: str) -> bytes:
        """取二级 zip 的原始字节（外层 zip 内嵌 / 已解压目录）。"""
        if self.mode == "dir":
            with open(os.path.join(self.sdir, os.path.basename(inner)), "rb") as f:
                return f.read()
        with zipfile.ZipFile(self.zpath) as z:
            return z.read(inner)

    def _index(self) -> List[Tuple[str, str, str]]:
        """扫描嵌套 zip，按文件名前缀配对 (outer_inner_path, hr_member, lr_member)。"""
        out: List[Tuple[str, str, str]] = []
        if self.mode == "dir":
            inners = [os.path.join(self.sdir, f) for f in os.listdir(self.sdir)
                      if f.lower().endswith(".zip")]
        else:
            with zipfile.ZipFile(self.zpath) as z:
                inners = [n for n in z.namelist() if n.lower().endswith(".zip")]
            inners = [n for n in inners
                      if (self.hr_res in n or self.lr_res in n) and self.bands in n]
        # 关键：HR(05m) 与 LR(10m/20m) 存放在【不同】的二级 zip 里，
        # 必须在全部 zip 上建索引后再按文件名前缀跨 zip 配对
        # （同一 zip 内配对会得到 0 对——已踩过）。
        def key(n):
            return os.path.basename(n).rsplit("_", 1)[0]   # 去掉 _{res}.tif

        hr_idx: Dict[str, Tuple[str, str]] = {}
        lr_idx: Dict[str, Tuple[str, str]] = {}
        for inner in inners:
            if self.bands not in inner:
                continue
            if self.hr_res not in inner and self.lr_res not in inner:
                continue
            data = self._read_inner(inner)
            with zipfile.ZipFile(io.BytesIO(data)) as z2:
                for n in z2.namelist():
                    if not n.lower().endswith(".tif"):
                        continue
                    k = key(n)
                    if n.endswith(f"_{self.hr_res}.tif"):
                        hr_idx[k] = (inner, n)
                    elif n.endswith(f"_{self.lr_res}.tif"):
                        lr_idx[k] = (inner, n)
            if len(hr_idx) >= 200000:
                break
        for k, (hi, hm) in hr_idx.items():
            if k in lr_idx:
                li, lm = lr_idx[k]
                out.append((hi, hm, li, lm))
        return out

    def __len__(self) -> int:
        return len(self.pairs)

    def load(self, idx: int) -> Dict:
        h_inner, h_mem, l_inner, l_mem = self.pairs[idx]
        with zipfile.ZipFile(io.BytesIO(self._read_inner(h_inner))) as z2:
            hr = _imread_any(z2.read(h_mem), is_bytes=True)
        with zipfile.ZipFile(io.BytesIO(self._read_inner(l_inner))) as z2:
            lr = _imread_any(z2.read(l_mem), is_bytes=True)
        hr = np.moveaxis(hr, 0, -1) if hr.ndim == 3 and hr.shape[0] <= 12 else hr
        lr = np.moveaxis(lr, 0, -1) if lr.ndim == 3 and lr.shape[0] <= 12 else lr
        # 统一形状约定：lrs (T,C,h,w)（SEN2VENµS 为同日单帧 -> T=1）
        return {
            "lrs": (np.moveaxis(lr, -1, 0).astype(np.float32) / self.div)[None],
            "hr": np.moveaxis(hr, -1, 0).astype(np.float32) / self.div,    # (C,H,W)
            "name": os.path.basename(h_mem),
        }


# ---------------------------------------------------------------- S2<->L8 (cross-sensor 3x)

class S2L8Dataset:
    """Sentinel-2 (10 m) <-> Landsat-8 (30 m) 原生分辨率跨传感器 3x 配对。

    数据源：namelesscoder/L8-S2-NAIP（本臂仅用其 S2/L8 导出对，NAIP 参考不用）。
    每个 ROI 为同一地面足迹的配对：
        <root>/sentinel2_export_roi_<id>.tif   (H, W, 4) float32  S2 10m 波段 B2/B3/B4/B8
        <root>/landsat8_export_roi_<id>.tif    (H, W, 6) float64  L8 30m 波段
    S2 在双空间维度恰为 L8 的 3 倍（原生 3x）。取 L8 RGB = 波段 [3,2,1]（红/绿/蓝），
    S2 RGB = 波段 [2,1,0]（B4/B3/B2 = 红/绿/蓝），使二者为物理对齐的红/绿/蓝通道，
    构成干净的跨传感器 3x 超分（LR=L8 RGB, HR=S2 RGB，均 3 通道）。

    load(idx) -> {"lrs":(1,C,Hl,Wl), "hr":(C,Hs,Ws), "name":id}，T=1 单帧。
    """

    def __init__(self, root: str, lr_bands=(3, 2, 1), hr_bands=(2, 1, 0), div: float = 1.0):
        self.root = root
        self.lr_bands = list(lr_bands)
        self.hr_bands = list(hr_bands)
        self.div = div
        s2 = glob.glob(os.path.join(root, "sentinel2_export_roi_*.tif"))
        ids = []
        for s in s2:
            rid = int(os.path.basename(s)[len("sentinel2_export_roi_"):-4])
            l8 = os.path.join(root, "landsat8_export_roi_%d.tif" % rid)
            if (os.path.exists(l8) and os.path.getsize(l8) > 50000
                    and os.path.getsize(s) > 50000):
                ids.append(rid)
        self.ids = sorted(ids)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def scale(self) -> int:
        return 3

    @staticmethod
    def _to_chw(arr, bands):
        """(H,W,C) 或 (C,H,W) -> 选波段的 (C,H,W) float32。"""
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1] // 2:
                pass  # 已是 (C,H,W)
            else:
                arr = np.transpose(arr, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        return arr[list(bands)]

    def load(self, idx: int) -> Dict:
        rid = self.ids[idx]
        sp = os.path.join(self.root, "sentinel2_export_roi_%d.tif" % rid)
        lp = os.path.join(self.root, "landsat8_export_roi_%d.tif" % rid)
        s_raw = tifffile.imread(sp)                               # (Hs,Ws,4)
        l_raw = tifffile.imread(lp)                               # (Hl,Wl,6)
        l = self._to_chw(l_raw, self.lr_bands)                   # (3,Hl,Wl)
        s = self._to_chw(s_raw, self.hr_bands)                   # (3,Hs,Ws)
        # NaN 有效掩膜（填充前记录）：L8/S2 边界 ~4% 无数据。评测只在
        # L8-有效 ∩ S2-有效 重叠区计算（与 e1 脚本 df_core 一致）；否则错位零边界会把
        # 相关/PSNR 打没（导致 G1≈0 的伪失败）。全通道都有效的像素才计入。
        # 注意：l/s 已是 (C,H,W)，axis=0 折叠通道维 -> (H,W)。
        l_valid = (~np.isnan(l)).all(axis=0).astype(np.float32)[None]   # (1,Hl,Wl)
        s_valid = (~np.isnan(s)).all(axis=0).astype(np.float32)[None]   # (1,Hs,Ws)
        l = l.astype(np.float32) / self.div
        s = s.astype(np.float32) / self.div
        # 裁剪到 [0,1] 前先把 NaN 归零，避免 NaN 经模型前向传播污染 batch 指标。
        l = np.clip(np.nan_to_num(l, nan=0.0), 0.0, 1.0)
        s = np.clip(np.nan_to_num(s, nan=0.0), 0.0, 1.0)
        return {"lrs": l[None].astype(np.float32), "hr": s.astype(np.float32),
                "lr_valid": l_valid, "hr_valid": s_valid,
                "name": "roi_%d" % rid}
