#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为 WorldStrat 计算「逐场景归一化」统计量（与 SEN2VENµS/PROBA-V 的
norm="per_scene" 完全同构）：对每个场景，用该场景所有 LR 帧的全部像素估计
(mu, sd)，之后 WorldStratDataset 把 LR 与 HR 都按 (x - mu)/sd 标准化。

为何要这步（#23 落地诊断结论）：
  逐场景【仿射】标定拟合出 slope≈0.4（S2↔SPOT 相关弱），把 LR 量程压到 HR 的
  约一半 -> 网络学到同量程小输出、L1 致回归均值 -> model 落后双三次 0.83 dB。
  逐场景【归一化】把 LR 变成 unit-variance 输入（梯度条件极好），网络只需学
  "空间 SR + 平滑增益"，正是兄弟数据集击败双三次的机制。

仅依赖 numpy + tifffile（不 import torch，避开 vcomp140）。
输出：data/processed/worldstrat_scene_norm.json  list[scene] -> [mu, sd]
（pooled 标量，与 ExternalSRDataset._pack 的 norm='per_scene' 语义一致）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import tifffile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
from utils.paths import rebase_manifest  # noqa: E402

HR_BANDS = (0, 1, 2)
LR_BANDS = (3, 2, 1)
HR_DIV = 10000.0
SUBSET = 600
MANIFEST = os.path.join(BASE, "data", "processed", "worldstrat_manifest.json")
OUT = os.path.join(BASE, "data", "processed", "worldstrat_scene_norm.json")


def read_tif(path, bands, div=None):
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


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    manifest = rebase_manifest(manifest)   # 发布版清单为相对路径，按数据根补全
    manifest = manifest[:SUBSET]
    print(f"[info] manifest subset={len(manifest)} (前 600 场景)")

    out = []
    for idx in range(len(manifest)):
        item = manifest[idx]
        # pooled 统计量：所有 LR 帧、所有通道的全部像素
        pix = []
        for p in item["lr"]:
            lr = read_tif(p, LR_BANDS)          # (3,Hl,Wl)
            pix.append(lr.reshape(-1))
        x = np.concatenate(pix)
        mu = float(x.mean())
        sd = float(x.std()) + 1e-6
        out.append([mu, sd])
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  fitted {idx + 1}/{len(manifest)}  mu={mu:.4f} sd={sd:.4f}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    arr = np.array(out)
    print(f"[saved] {OUT}")
    print(f"  mu: μ={arr[:,0].mean():.4f} σ={arr[:,0].std():.4f}")
    print(f"  sd: μ={arr[:,1].mean():.4f} σ={arr[:,1].std():.4f}")


if __name__ == "__main__":
    main()
