#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为 WorldStrat 计算「逐场景」辐射标定参数（LR -> HR 尺度）。

背景（#23 结论）：全局单一仿射 lr_calib=[0.68586,0.001835] 在 6.63× 多时相
WorldStrat 上相对双三次地板是负贡献（地板 16.67 -> 逐场景 20.61 dB, +4.96 dB,
p=2e-14）。应改为逐场景拟合 slope_s / intercept_s。

做法（零训练、泄漏安全，且与训练管线【完全一致】的地理配准）：
  复用 WorldStratDataset 自身的 _pair_box 配准数学，对每场景取若干确定性窗口，
  提取【已地理配准】的 LR 多帧裁剪 + 对应 HR 裁剪，把 HR 裁剪下采样到 LR 网格后
  逐通道用稳健 2-pass 最小二乘拟合  HR ≈ slope_s * LR + intercept_s。
  标定是「该场景自身 LR/HR 对的确定性预处理变换」，各场景只用自己数据，无跨集泄漏
  （与 #23 判定一致）。仅依赖 numpy + cv2（不 import torch，避开 vcomp140 冲突）。

输出：data/processed/worldstrat_scene_calib.json
  list[scene_idx] -> [[s_ch0,b_ch0],[s_ch1,b_ch1],[s_ch2,b_ch2]]
  长度 = subset（与 get_dataset 的 manifest[:subset] 对齐，取前 600）。
"""
from __future__ import annotations

import json
import os
import sys
import random

import cv2
import numpy as np
import tifffile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
from utils.paths import rebase_manifest  # noqa: E402

# 与 external_worldstrat.py 严格一致的数据参数
HR_BANDS = (0, 1, 2)     # HR(_ps) RGBA -> RGB
LR_BANDS = (3, 2, 1)     # S2 12 波段：B4/B3/B2 = RGB
HR_DIV = 10000.0         # HR DN = 反射率*10000
LR_SIZE = 64
HR_PER_LR = 6.63
SUBSET = 600
MANIFEST = os.path.join(BASE, "data", "processed", "worldstrat_manifest.json")
OUT = os.path.join(BASE, "data", "processed", "worldstrat_scene_calib.json")
N_WINDOWS = 12            # 每场景聚合的确定性注册窗口数（提升标定稳健性）
# 退化场景（常量patch/拟合发散）回退到 #22 用的全局标定，避免灾难性斜率
GLOBAL_FALLBACK = [0.68586, 0.001835]


def read_tif_cached(path, bands, div=None):
    arr = tifffile.imread(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1] // 2:
            pass
        else:
            arr = np.transpose(arr, (2, 0, 1))
    if bands is not None:
        arr = arr[list(bands)]
    if div is not None:
        arr = arr / float(div)
    elif arr.max() > 1.0:
        arr = arr / 65535.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def resize_bc(arr, size):
    """arr: (C,H,W) -> (C,size,size) bicubic via cv2 (per channel)."""
    C = arr.shape[0]
    out = np.empty((C, size, size), dtype=np.float32)
    for c in range(C):
        out[c] = cv2.resize(arr[c], (size, size), interpolation=cv2.INTER_CUBIC)
    return out


def robust_affine(x, y):
    """稳健 2-pass 最小二乘拟合 y ≈ s*x + b；用 MAD 剔除离群（云/错配）。

    返回 [slope, intercept]；若该场景退化（LR/HR 近常量、拟合发散、非有限或斜率越界）
    返回 None，由调用方回退到全局标定。
    """
    x = np.asarray(x, np.float64).ravel()
    y = np.asarray(y, np.float64).ravel()
    if x.std() < 1e-4 or y.std() < 1e-4:
        return None  # 近常量patch，仿射无意义
    A = np.stack([x, np.ones_like(x)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    med = np.median(res)
    mad = np.median(np.abs(res - med)) + 1e-12
    mask = np.abs(res - med) < 3.0 * 1.4826 * mad
    if mask.sum() < 10:
        mask = np.ones_like(mask)
    coef, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
    s, b = float(coef[0]), float(coef[1])
    if not (np.isfinite(s) and np.isfinite(b)):
        return None
    if not (0.01 <= abs(s) <= 20.0) or not (-1.0 <= b <= 1.0):
        return None  # 越界（如常量patch致发散），回退全局
    return [s, b]


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    manifest = rebase_manifest(manifest)   # 发布版清单为相对路径，按数据根补全
    manifest = manifest[:SUBSET]
    print(f"[info] manifest subset={len(manifest)} (前 600 场景)")

    out = []
    for idx in range(len(manifest)):
        item = manifest[idx]
        hr_full = read_tif_cached(item["hr"], HR_BANDS, div=HR_DIV)
        lr0_full = read_tif_cached(item["lr"][0], LR_BANDS)
        # 复用 _pair_box 的配准数学（与 dataset.py 完全一致）
        ratio = float(hr_full.shape[-1]) / float(lr0_full.shape[-1])
        hs = min(int(round(LR_SIZE * HR_PER_LR)), hr_full.shape[-1], hr_full.shape[-2])
        lw = max(2, min(int(round(hs / ratio)), lr0_full.shape[-1], lr0_full.shape[-2]))
        Hl, Wl = lr0_full.shape[-2], lr0_full.shape[-1]
        Hr, Wr = hr_full.shape[-2], hr_full.shape[-1]
        i_max, j_max = max(0, Hl - lw), max(0, Wl - lw)
        xs_all, ys_all = [[] for _ in range(3)], [[] for _ in range(3)]
        for w in range(N_WINDOWS):
            rng = random.Random(idx * 7919 + 13 + w * 104729)
            i, j = rng.randint(0, i_max), rng.randint(0, j_max)
            hi = min(max(0, int(round(i * ratio))), max(0, Hr - hs))
            hj = min(max(0, int(round(j * ratio))), max(0, Wr - hs))
            # LR 裁剪 + 网格对齐到 LR_SIZE
            lr_crop = lr0_full[:, i:i + lw, j:j + lw]
            if lw != LR_SIZE:
                lr_crop = resize_bc(lr_crop, LR_SIZE)
            hr_crop = hr_full[:, hi:hi + hs, hj:hj + hs]
            if hr_crop.shape[-2] < hs or hr_crop.shape[-1] < hs:
                hr_crop = resize_bc(hr_crop, hs)
            # HR 裁剪下采样到 LR_SIZE 网格
            hr_ds = resize_bc(hr_crop, LR_SIZE)
            for c in range(3):
                xs_all[c].append(lr_crop[c].ravel())
                ys_all[c].append(hr_ds[c].ravel())

        calib = []
        for c in range(3):
            x = np.concatenate(xs_all[c])
            y = np.concatenate(ys_all[c])
            fit = robust_affine(x, y)
            calib.append(fit if fit is not None else list(GLOBAL_FALLBACK))
        out.append(calib)
        n_degen = sum(1 for c in range(3) if calib[c] == list(GLOBAL_FALLBACK))
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  fitted {idx + 1}/{len(manifest)}  scene{idx} slope="
                  f"[{calib[0][0]:.3f},{calib[1][0]:.3f},{calib[2][0]:.3f}]"
                  f"{'  [回退全局x'+str(n_degen)+']' if n_degen else ''}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[saved] {OUT}  ({len(out)} 场景, 每场景 3 通道 [slope,intercept])")

    arr = np.array(out)
    for c, name in enumerate(("R", "G", "B")):
        s = arr[:, c, 0]; b = arr[:, c, 1]
        print(f"  ch{name}: slope μ={s.mean():.4f} σ={s.std():.4f}  "
              f"intercept μ={b.mean():.5f} σ={b.std():.5f}")
    print(f"  全局对照 lr_calib=[0.68586,0.001835]")


if __name__ == "__main__":
    main()
