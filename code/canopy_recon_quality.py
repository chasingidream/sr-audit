# -*- coding: utf-8 -*-
"""Path A 诊断 B —— 重算 PSNR 分解，判定「|Δt|×texture」是真机制还是 dB 尺度伪影。

动机
  Δ_floor := PSNR(bicubic(LR), HR_n) − PSNR(mean(HR_n), HR_n)。
  第二项 = −10·log10(var(HR_n)) —— **与采集间隔无关**，却与 texture 强相关。
  因此 Δ_floor 上的「|Δt|×texture」交互可能是尺度耦合造成的伪影。
  干净的做法：直接对 PSNR(bic)（真正的重建质量）与 PSNR(mean)（基线质量）分别建模。

输出 experiments/recon_quality.jsonl（每覆盖一行）
  psnr_bic / psnr_mean / delta_floor（校验用）/ var_hr_n / contrast_lr

用法：python code/canopy_recon_quality.py [--workers 8] [--probe N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stratify_floor import (WS_ROOT, HR_OUT, DIV_HR, read_tif, bicubic_np, psnr)  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "experiments", "recon_quality.jsonl")
SRC_JSON = os.path.join(BASE, "experiments", "stratified_floor_ws.json")


def one(cov):
    d = os.path.join(WS_ROOT, cov)
    try:
        fs = os.listdir(d)
    except OSError:
        return None
    hrp = [f for f in fs if f.endswith("_ps.tiff")] or \
          [f for f in fs if f.endswith("_rgbn.tiff")]
    if not hrp:
        return None
    lrd = os.path.join(d, "L2A")
    try:
        lrs = sorted(f for f in os.listdir(lrd) if f.endswith("L2A_data.tiff"))
    except OSError:
        return None
    if not lrs:
        return None

    # —— 与 stratify_floor.extract 逐行同口径 ——
    hr_full = read_tif(os.path.join(d, hrp[0]), bands=(0, 1, 2), div=DIV_HR)
    lr0 = read_tif(os.path.join(lrd, lrs[0]), bands=(3, 2, 1))
    Hr, Wr = hr_full.shape[-2], hr_full.shape[-1]
    Hl, Wl = lr0.shape[-2], lr0.shape[-1]
    ratio = float(Hr) / float(Hl)
    hr_size = min(HR_OUT, Hr, Wr)
    lr_win = max(2, min(int(round(hr_size / ratio)), Hl, Wl))
    i = max(0, (Hl - lr_win) // 2)
    j = max(0, (Wl - lr_win) // 2)
    hi = min(max(0, int(round(i * ratio))), max(0, Hr - hr_size))
    hj = min(max(0, int(round(j * ratio))), max(0, Wr - hr_size))
    hr = hr_full[:, hi:hi + hr_size, hj:hj + hr_size]
    lr = lr0[:, i:i + lr_win, j:j + lr_win]

    mu = float(lr.mean())
    sd = float(lr.std()) + 1e-6
    hr_n = (hr - mu) / sd

    bic = bicubic_np(lr, (hr_size, hr_size))
    p_bic = psnr(bic, hr_n)
    p_mean = psnr(np.full_like(hr_n, hr_n.mean()), hr_n)

    # 额外的、尺度无关的重建质量指标：bic 与 HR_n 的相关系数
    a = bic.ravel().astype(np.float64)
    b = hr_n.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    corr = float((a * b).sum() / den) if den > 1e-12 else np.nan

    return {"cov": cov,
            "psnr_bic": round(p_bic, 4),
            "psnr_mean": round(p_mean, 4),
            "delta_floor": round(p_bic - p_mean, 4),
            "var_hr_n": float(hr_n.var()),
            "corr_bic_hr": round(corr, 6),
            "contrast_lr": round(sd, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    covs = [s["cov"] for s in
            json.load(open(SRC_JSON, encoding="utf-8"))["scenes"]]
    if a.probe:
        covs = covs[:a.probe]
    print("[info] coverages = %d" % len(covs), flush=True)

    ok = 0
    import time
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, c): c for c in covs}
        for k, fu in enumerate(as_completed(futs)):
            try:
                r = fu.result()
            except Exception as e:
                r = None
                print("[warn] %s: %s: %s" % (futs[fu], type(e).__name__, e), flush=True)
            if r:
                f.write(json.dumps(r) + "\n")
                ok += 1
            if (k + 1) % 400 == 0:
                el = time.time() - t0
                print("  ..%d/%d ok=%d (%.0fs, %.1f/s)"
                      % (k + 1, len(covs), ok, el, (k + 1) / max(el, 1e-6)), flush=True)
    print("[ok] wrote %s n=%d / %d" % (OUT, ok, len(covs)))


if __name__ == "__main__":
    main()
