#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e1_scale_isolation_matched.py —— E1 纯尺度控制的【场景匹配】重跑（评审④修复）。

背景：模拟评审④指出 `e1_scale_isolation.py` 的纯尺度控制用全 manifest 末 60 景
（`m[-60:]`，中心 640 裁剪），而 WorldStrat real arm 的 val-60 是 600 子集的尾 60
（`manifest[540:600]`，确定性 424 窗口）——两者场景不同，≈10 dB 去相关分解存在
场景混杂。本脚本把纯尺度控制重跑在【与 real arm 完全相同的 60 景 + 相同 424
确定性窗口】上，给出场景匹配的分解（零训练，纯数据属性测量）。

双口径（与主文两层数字对应）：
  * method-aligned（主口径，对应 paired_worldstrat.json 的 −7.36）：
    自采样 LR 与 HR 均按该 LR 的 (mu,sd) 逐场景归一化（与真实管线同构：
    mu/sd = LR 全部像素混池统计，见 compute_worldstrat_scene_norm.py），
    data_range = 60 景归一化 HR 的全局 min/max（与 eval_paired_floor 同式）；
    bicubic 上采样 = F.interpolate bicubic align_corners=False（与审计装载同约定）。
  * raw（数据层口径，对应 −0.09）：原始 [0,1] 反射率，data_range=1.0。

预期：场景匹配后 6.63× 纯尺度 Δ_floor 仍显著为正（量级 ≈ +4~5 dB），
⇒ WS 轴去相关贡献 ≈ −12 dB 的结论对场景选择稳健。

用法：python code/e1_scale_isolation_matched.py
输出：experiments/e1_scale_isolation_matched.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import _dll_guard  # noqa: E402
_dll_guard.hide()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from data.dataset import get_dataset  # noqa: E402
from eval_metrics import calc_psnr  # noqa: E402
from eval_paired_floor import load_cfg  # noqa: E402

CFG_PATH = os.path.join(BASE, "code", "configs", "external_worldstrat.py")
OUT_JSON = os.path.join(BASE, "experiments", "e1_scale_isolation_matched.json")
SCALES = [2.0, 3.0, 4.0, 5.0, 6.63]

# real-arm 方法层 Δ_floor（与主文/旧 e1 同源：paired json 逐样本口径）
PAIRED_SOURCES = {6.63: "paired_worldstrat.json"}
# real-arm 数据层 Δ_floor（docs/跨数据集审计_汇总.md，同旧 e1 脚本）
REAL_DATALAYER = {2.0: 33.86 - 11.90, 3.0: 21.92 - 19.83, 6.63: 20.08 - 20.17}
# 旧（场景不匹配）纯尺度结果，仅作对照打印
OLD_PURE = {2.0: 18.75, 3.0: 11.22, 4.0: None, 5.0: None, 6.63: 4.60}


def bicubic_up(x, size):
    return F.interpolate(x, size=size, mode="bicubic", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="matched", choices=["matched", "tail60"],
                    help="matched=与 real arm 同 60 景+确定性 424 窗口; "
                         "tail60=旧脚本的全 manifest 末 60 景+中心 640 裁剪")
    ap.add_argument("--antialias", action="store_true", default=True,
                    help="下采样是否抗锯齿（默认开，物理上正确的降质）")
    ap.add_argument("--no-antialias", dest="antialias", action="store_false")
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(CFG_PATH)
    full = get_dataset(cfg, project_root=BASE, deterministic=True)
    val_frac = float(cfg.get("val_frac", 0.1))
    n_train = max(1, min(len(full) - 1, int((1.0 - val_frac) * len(full))))
    subset = list(range(n_train, len(full)))          # 与 real arm val 完全同一批场景
    loader = DataLoader(Subset(full, subset), batch_size=1, shuffle=False,
                        num_workers=0)
    print(f"[info] scenes={a.scenes} antialias={a.antialias} "
          f"val-{len(subset)}，device={device}")

    # ---- 第一遍：收集 raw HR 裁剪 + 自采样 LR ----
    # 两种场景来源：matched=real arm 同景同窗（424）；tail60=旧脚本末 60 景中心 640
    CROP = 424 if a.scenes == "matched" else 640
    hr_raws, lrs_raw = [], []                          # 每景 [3,CROP,CROP]
    if a.scenes == "matched":
        with torch.no_grad():
            for idx_rel, batch in enumerate(loader):
                idx = n_train + idx_rel
                gt_n = batch["target"].float().to(device)      # 归一化 GT [1,3,424,424]
                mu, sd = full.scene_norm[idx]                  # 与管线同一 (mu,sd)
                hr_raw = (gt_n * (float(sd) + 1e-6) + float(mu)).clamp(0.0, 1.0)
                lrs_s = {}
                for s in SCALES:
                    lo = int(round(CROP / s))
                    lr = F.interpolate(hr_raw, size=(lo, lo), mode="bicubic",
                                       align_corners=False, antialias=a.antialias)
                    lrs_s[s] = lr.clamp(0.0, 1.0)
                hr_raws.append(hr_raw[0])
                lrs_raw.append(lrs_s)
    else:
        import tifffile
        mani = json.load(open(os.path.join(BASE, "data", "processed",
                                           "worldstrat_manifest.json")))
        for e in mani[-60:]:
            arr = tifffile.imread(e["hr"]).astype(np.float32)
            rgb = arr[..., :3] / 10000.0
            rgb = np.clip(rgb, 0.0, 1.0)
            H, W = rgb.shape[:2]
            i, j = max(0, (H - CROP) // 2), max(0, (W - CROP) // 2)
            hr = torch.from_numpy(rgb[i:i + CROP, j:j + CROP]
                                  .transpose(2, 0, 1)).float().to(device)
            lrs_s = {}
            for s in SCALES:
                lo = int(round(CROP / s))
                lr = F.interpolate(hr[None], size=(lo, lo), mode="bicubic",
                                   align_corners=False, antialias=a.antialias)
                lrs_s[s] = lr.clamp(0.0, 1.0)
            hr_raws.append(hr)
            lrs_raw.append(lrs_s)
    print(f"[info] raw 裁剪收集完成 n={len(hr_raws)} crop={CROP}")

    # ---- 第二遍：双口径逐样本 Δ_floor ----
    rows = {s: {"raw": [], "matched": []} for s in SCALES}
    # method-aligned 的 val_dr：先扫一遍自 LR 统计归一化后的 HR 全局 min/max
    hns = []
    dr_mn, dr_mx = float("inf"), float("-inf")
    for i in range(len(hr_raws)):
        hr = hr_raws[i][None]
        # 与真实管线一致：scene_norm 用该场景【全部 LR 帧】的混池统计；纯尺度无多帧，
        # 用三个尺度自 LR 的合并像素池近似“全部 LR 帧”的混池（对 (mu,sd) 的影响 <1%）。
        pix = torch.cat([lrs_raw[i][s].reshape(-1) for s in SCALES])
        mu, sd = float(pix.mean()), float(pix.std()) + 1e-6
        hn = ((hr - mu) / sd)[0]
        hns.append(hn)
        dr_mn2, dr_mx2 = float(hn.min()), float(hn.max())
        dr_mn, dr_mx = min(dr_mn, dr_mn2), max(dr_mx, dr_mx2)
    val_dr_m = (dr_mx - dr_mn) if dr_mx - dr_mn > 1e-8 else 1.0
    print(f"[info] method-aligned val_dr={val_dr_m:.4f}")

    with torch.no_grad():
        for i in range(len(hr_raws)):
            hr = hr_raws[i][None]
            hn = hns[i][None]
            mean_r = hr.mean(dim=(2, 3), keepdim=True).expand_as(hr)
            mean_m = hn.mean(dim=(2, 3), keepdim=True).expand_as(hn)
            p_mean_raw = float(calc_psnr(mean_r[0], hr[0], data_range=1.0))
            p_mean_m = float(calc_psnr(mean_m[0], hn[0], data_range=val_dr_m))
            for s in SCALES:
                lr = lrs_raw[i][s]
                up = bicubic_up(lr, (CROP, CROP))
                # raw 口径
                p_bic_r = float(calc_psnr(up[0], hr[0], data_range=1.0))
                rows[s]["raw"].append(p_bic_r - p_mean_raw)
                # method-aligned 口径：自 LR 的 (mu,sd) 归一 LR 与 HR
                pix = lr.reshape(-1)
                mu, sd = float(pix.mean()), float(pix.std()) + 1e-6
                up_n = (up - mu) / sd
                hr_n = (hr - mu) / sd
                p_bic_m = float(calc_psnr(up_n[0], hr_n[0], data_range=val_dr_m))
                rows[s]["matched"].append(p_bic_m - p_mean_m)

    # ---- 聚合 + 分解对照 ----
    real_ml = {}
    for s, f in PAIRED_SOURCES.items():
        d = json.load(open(os.path.join(BASE, "experiments", f)))
        g = d["group_mean_per_sample"]
        real_ml[s] = float(g["bicubic"]["psnr"]) - float(g["mean"]["psnr"])

    out = {"n_scenes": len(hr_raws), "crop": CROP, "scenes": a.scenes,
           "antialias": bool(a.antialias),
           "design": ("pure-scale control: LR = HR self-downsampled (bicubic, "
                      f"antialias={a.antialias}), bicubic(ac=False) upsampled; "
                      "scenes=" + a.scenes),
           "scales": {}, "decomposition": {}}
    print("\n== 场景匹配纯尺度 Δ_floor（n=60，与 real arm 同景同窗）==")
    print(f"{'scale':>6} {'raw':>8} {'matched':>9} {'旧(不匹配)':>10}")
    for s in SCALES:
        r = {k: float(np.mean(v)) for k, v in rows[s].items()}
        sd_ = {k: float(np.std(v, ddof=1)) for k, v in rows[s].items()}
        out["scales"][s] = {"delta_floor_raw_mean": round(r["raw"], 3),
                            "delta_floor_raw_std": round(sd_["raw"], 3),
                            "delta_floor_matched_mean": round(r["matched"], 3),
                            "delta_floor_matched_std": round(sd_["matched"], 3),
                            "old_unmatched_pure": OLD_PURE[s],
                            "per_sample": {k: [round(x, 4) for x in v]
                                           for k, v in rows[s].items()}}
        _old = OLD_PURE[s]; _olds = f"{_old:.2f}" if _old is not None else "  --  "; print(f"{s:>6} {r['raw']:>8.3f} {r['matched']:>9.3f} {_olds:>10}")
        if s in real_ml:
            dec_m = real_ml[s] - r["matched"]
            dec_r = REAL_DATALAYER[s] - r["raw"]
            out["decomposition"][s] = {
                "real_method_layer": round(real_ml[s], 3),
                "pure_matched": round(r["matched"], 3),
                "decorrelation_method_layer": round(dec_m, 3),
                "real_data_layer": round(REAL_DATALAYER[s], 3),
                "pure_matched_raw": round(r["raw"], 3),
                "decorrelation_data_layer": round(dec_r, 3)}
            print(f"  → WS 轴场景匹配分解：去相关贡献(方法层) = "
                  f"{real_ml[s]:.2f} − ({r['matched']:.2f}) = {dec_m:+.2f} dB  "
                  f"(数据层 {dec_r:+.2f} dB)")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {a.out}")


if __name__ == "__main__":
    main()
