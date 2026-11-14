#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sen2sr_ws_audit.py —— 把 SEN2SR 从「定性友军」升级为「可审计臂」。

背景与动机（2026-10-05，用户指令「补 SOTA」）
------------------------------------------------
`experiments/sota_review/sen2sr.json` 迄今是 **定性** 记录，理由写得很明确：
  "其评测仍依赖合成 SEN2NAIP + 未做跨数据集技能地板/逐样本配对检验
   → 定性引用, 不作增益缩水量化。"
本脚本补上这一缺口：把 SEN2SR（Aybar et al., RSE 2026, ESA Φ-lab；RSE 334:115222）
的 **已发布权重** 直接跑在本项目 WorldStrat 真实跨传感器配对数据上，用与
Table 2 其余 9 臂 **完全同一套 SR-Audit 口径**（逐样本 PSNR + 配对 t + G1–G5 +
Δ_floor 地板）量化它是否过地板。

为何是「新的数据点」而非重复
------------------------------------------------
Table 2 的 9 臂都跑在项目自定的 **6.63×** 协议上（LR 64 → HR 424）。
SEN2SR 是原生 **×4** 模型且其 Fourier 硬约束掩码是 512×512（不可替换：见下），
故本臂在 **原生 ×4 体制** 上评测：LR 128×128 (S2 10 m, RGBN) → SR 512×512 (2.5 m)，
GT = SPOT 参考在同一地面覆盖上重采样到 512（2.5 m 网格）。
这恰好落在主文 E1 尺度轴（2/3/4/5/6.63×）的 **×4 点** 上——即「真实跨传感器 ×4」
这一此前空缺的格子。

技术要点（全部零近似）
------------------------------------------------
1. **权重与掩码原封不动**：model.safetensor / hard_constraint.safetensor 为官方发布物。
   掩码 (512,512) 经查为可分（separable）、中心 (255.5,255.5) 的平滑低通；因其直接与
   SR 频谱逐元素相乘，SR 尺寸必须 = 512 ⇒ 输入必须 = 128。本脚本因此不做任何
   掩码外推/重采样，保证与 SEN2SR 发布行为逐位一致。
2. **输入波段 = SEN2SR 原生顺序 (B04,B03,B02,B08)**：WorldStrat L2A 12 波段顺序为
   [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12]（`data/dataset.py` L227 有记载），
   故 RGBN = 索引 [3,2,1,7]。输入为 **原始反射率 [0,1]**（read_tif 直读，未做任何归一化），
   与 SEN2SR 训练域（example_data: min 0.011 / max 0.998）一致。
3. **辐射桥接不接触 GT**：SEN2SR 输出与 GT 皆用数据管线自身的逐场景 (mu,sd)
   （`worldstrat_scene_norm.json`，由 LR RGB 反射率统计而来）映射到 per_scene 归一化
   空间——与其余 4 臂同空间。**不使用** hrnet 臂那种 GT 知情 bias_adjust。
4. **配对的 bicubic 基准与本臂同口径**：bicubic_up(lr_norm, (512,512), align_corners=False)。

用法
------------------------------------------------
  python code/sen2sr_ws_audit.py --limit 4     # 冒烟
  python code/sen2sr_ws_audit.py               # 全量（val 尾 10%，默认 60 场景）
输出：experiments/sen2sr_audit/ws_arm.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, os.path.join(BASE, "code", "sota"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from data.dataset import get_dataset, read_tif  # noqa: E402
from eval_metrics import calc_psnr, calc_cc  # noqa: E402
from sr_audit import bicubic_up, periodicity_scan  # noqa: E402
from sr_audit_report import derive  # noqa: E402
from eval_paired_floor import load_cfg  # noqa: E402
from sen2sr.models.opensr_baseline.cnn import CNNSR  # noqa: E402
from sen2sr.models.tricks import HardConstraint  # noqa: E402
from sen2sr.nonreference import srmodel  # noqa: E402

MODEL_DIR = os.path.join(BASE, "experiments", "sen2sr_audit", "model")
OUT_DIR = os.path.join(BASE, "experiments", "sen2sr_audit")

LR_BANDS_SEN2SR = [3, 2, 1, 7]   # B04,B03,B02,B08 = RGBN（SEN2SR 原生顺序）
# ⚠ read_tif 用 arr[bands] 做 numpy 高级索引：必须是 list，tuple 会被当成多轴索引报错。
LR_CROP = 128                    # SEN2SR 硬约束掩码 512 = LR 128 × 4
SR_SIZE = 512


def load_sen2sr(device="cpu"):
    import safetensors.torch as st
    sr_model = CNNSR(4, 4, 24, 4, True, False, 6)
    miss, unexp = sr_model.load_state_dict(
        st.load_file(os.path.join(MODEL_DIR, "model.safetensor")), strict=False)
    if miss or unexp:
        print(f"[warn] state_dict mismatch missing={len(miss)} unexpected={len(unexp)}")
    sr_model.eval()
    hc = HardConstraint(low_pass_mask=st.load_file(
        os.path.join(MODEL_DIR, "hard_constraint.safetensor"))["weights"], device=device)
    model = srmodel(sr_model, hc, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def paired_t(a, b):
    import scipy.stats as _st
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    t, p = _st.ttest_rel(a, b)
    sd = d.std(ddof=1)
    ci = _st.t.interval(0.95, n - 1, loc=d.mean(), scale=sd / math.sqrt(n))
    return {"n": n, "mean": round(float(d.mean()), 4), "sd": round(float(sd), 4),
            "t": round(float(t), 3), "p": float(f"{p:.3g}"),
            "d": round(float(d.mean() / sd), 3) if sd > 0 else 0.0,
            "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)]}


def pick_pair_box(hr_shape, lr_shape, idx, lr_win):
    """与 WorldStratDataset._pair_box 同约定，但目标窗由 lr_win 反推。"""
    ratio = float(hr_shape[-1]) / float(lr_shape[-1])
    Hl, Wl = lr_shape[-2], lr_shape[-1]
    Hr, Wr = hr_shape[-2], hr_shape[-1]
    hs = int(round(lr_win * ratio))
    hs = min(hs, Hr, Wr)
    i_max, j_max = max(0, Hl - lr_win), max(0, Wl - lr_win)
    rng = random.Random(idx * 7919 + 13)
    i, j = rng.randint(0, i_max), rng.randint(0, j_max)
    hi = min(max(0, int(round(i * ratio))), max(0, Hr - hs))
    hj = min(max(0, int(round(j * ratio))), max(0, Wr - hs))
    return (i, j, lr_win), (hi, hj, hs), ratio


def main():
    ap = argparse.ArgumentParser(description="SEN2SR × WorldStrat 真实跨传感器审计")
    ap.add_argument("--cfg", default=os.path.join(BASE, "code", "configs",
                                                  "external_worldstrat.py"))
    ap.add_argument("--stage", default="mtxa_aligned")
    ap.add_argument("--limit", type=int, default=0, help=">0 只处理前 N 个 val 场景")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = "cpu"

    cfg = load_cfg(args.cfg)
    cfg["stage"] = args.stage
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = get_dataset(cfg, project_root=BASE, deterministic=True)
    val_frac = float(cfg.get("val_frac", 0.1))
    n = len(ds)
    n_train = max(1, min(n - 1, int((1.0 - val_frac) * n)))
    n_val = n - n_train
    val_idx = list(range(n_train, n))

    # val_dr：与 p2_ws_sr_dump / audit_axis 完全同口径（归一化空间）
    loader = DataLoader(Subset(ds, val_idx), batch_size=1, shuffle=False, num_workers=0)
    mn, mx = float("inf"), float("-inf")
    for vb in loader:
        t = vb["target"].float()
        mn, mx = min(mn, float(t.min())), max(mx, float(t.max()))
    val_dr = (mx - mn) if mx - mn > 1e-8 else 1.0
    print(f"[info] n_total={n} n_val={n_val} val_dr={val_dr:.4f} device={device}")

    model = load_sen2sr(device)
    # 消融臂：去掉 Fourier 硬约束的裸网络（clamp(min=0)），用于定位周期伪影来源与量化
    # 硬约束对该站点的贡献。官方 pipeline 的 forward = clamp(net(x),0) -> hard_constraint。
    net_only = CNNSR(4, 4, 24, 4, True, False, 6)
    import safetensors.torch as _st
    net_only.load_state_dict(_st.load_file(os.path.join(MODEL_DIR, "model.safetensor")))
    net_only.eval()
    for p in net_only.parameters():
        p.requires_grad = False
    print(f"[info] SEN2SR NonReference_RGBN_x4 loaded (LR {LR_CROP} -> SR {SR_SIZE})")

    hr_bands = list(ds.hr_bands)   # list，理由同 LR_BANDS_SEN2SR
    hr_div = float(ds.hr_div)
    T = int(ds.T)

    rows = {"bicubic": [], "sen2sr": [], "sen2sr_nohc": []}
    rows_floor, cc_rows, stdr_rows, grid_rows = [], {"bicubic": [], "sen2sr": []}, \
        {"bicubic": [], "sen2sr": []}, {"bicubic": [], "sen2sr": [], "sen2sr_nohc": []}
    d_model, d_floor = [], []
    period_rows = {"bicubic": [], "sen2sr": [], "sen2sr_nohc": []}
    scene_ids, skipped = [], []
    n_used = 0

    for rel, idx in enumerate(val_idx):
        if args.limit and rel >= args.limit:
            break
        item = ds.manifest[idx]
        # --- HR 参考（SPOT，反射率）---
        hr_full = ds._read_cache(item["hr"], hr_bands, div=hr_div)
        lr0 = ds._read_cache(item["lr"][0], ds.lr_bands)     # 仅用于定 ratio/窗口
        lr_box, hr_box, ratio = pick_pair_box(hr_full.shape, lr0.shape, idx, LR_CROP)
        li, lj, lw = lr_box
        hi, hj, hs = hr_box
        # --- LR 原始反射率 RGBN（不经 ds 缓存：band 组合不同，key 会撞）---
        lr_refl = read_tif(item["lr"][0], LR_BANDS_SEN2SR)   # (4,H,W) refl [0,1]
        if lr_refl.shape[-2] < LR_CROP or lr_refl.shape[-1] < LR_CROP:
            skipped.append({"idx": idx, "reason": f"LR too small {lr_refl.shape}"})
            continue
        lr4 = torch.from_numpy(lr_refl[:, li:li + lw, lj:lj + lw].copy())  # (4,128,128)

        mu, sd = ds.scene_norm[idx]
        mu, sd = float(mu), float(sd) + 1e-6

        gt_refl = hr_full[:, hi:hi + hs, hj:hj + hs]          # (3,hs,hs) refl
        gt_norm = (gt_refl - mu) / sd
        gt = F.interpolate(gt_norm[None], size=(SR_SIZE, SR_SIZE), mode="bicubic",
                           align_corners=False)[0]            # (3,512,512) 归一化空间

        # --- bicubic 基准（同口径：归一化 LR -> 512）---
        lr_norm = (lr4[:3] - mu) / sd
        bic = bicubic_up(lr_norm[None], (SR_SIZE, SR_SIZE))[0]

        # --- SEN2SR（输入=原始反射率 RGBN，输出=反射率 RGBN）---
        with torch.no_grad():
            sr4 = model(lr4[None].float())[0]                 # (4,512,512) refl
            sr4_nohc = torch.clamp(net_only(lr4[None].float())[0], min=0.0)
        sr = (sr4[:3] - mu) / sd                              # 归一化空间
        sr_nohc = (sr4_nohc[:3] - mu) / sd

        rows["bicubic"].append(float(calc_psnr(bic, gt, data_range=val_dr)))
        rows["sen2sr"].append(float(calc_psnr(sr, gt, data_range=val_dr)))
        rows["sen2sr_nohc"].append(float(calc_psnr(sr_nohc, gt, data_range=val_dr)))
        mean_img = torch.full_like(gt, float(gt.mean()))
        rows_floor.append(float(calc_psnr(mean_img, gt, data_range=val_dr)))
        for a, o in (("bicubic", bic), ("sen2sr", sr)):
            cc_rows[a].append(float(calc_cc(o, gt)))
            stdr_rows[a].append(float(o.std() / (gt.std() + 1e-8)))
        for a, o in (("bicubic", bic), ("sen2sr", sr), ("sen2sr_nohc", sr_nohc)):
            g, pg = periodicity_scan(o)
            grid_rows[a].append(float(g) if g == g else float("nan"))
            period_rows[a].append(int(pg) if pg else -1)
        scene_ids.append(os.path.basename(os.path.dirname(item["hr"])))
        n_used += 1
        if (rel + 1) % 10 == 0:
            print(f"  ...{rel+1}/{args.limit or n_val}  "
                  f"bic={np.mean(rows['bicubic']):.2f} sr={np.mean(rows['sen2sr']):.2f}")

    if n_used == 0:
        raise SystemExit("no scenes processed")

    # --- 审计量：Δ_model 与 Δ_floor 逐样本配对 ---
    d_model = [a - b for a, b in zip(rows["sen2sr"], rows["bicubic"])]
    d_floor = [a - b for a, b in zip(rows["bicubic"], rows_floor)]

    pt_model = paired_t(rows["sen2sr"], rows["bicubic"])
    pt_nohc = paired_t(rows["sen2sr_nohc"], rows["bicubic"])
    pt_floor = paired_t(rows["bicubic"], rows_floor)

    gain = float(np.mean(rows["sen2sr"]) - np.mean(rows["bicubic"]))
    r_cc = float(np.mean(cc_rows["sen2sr"]) / max(np.mean(cc_rows["bicubic"]), 1e-6))
    card = {
        "method": "SEN2SR (Aybar et al. RSE 2026, ESA Φ-lab)",
        "n_test": n_used,
        "audit_gain_db": round(gain, 3),
        "audit_p": pt_model["p"],
        "passes_floor": bool(np.mean(rows["sen2sr"]) > np.mean(rows_floor)),
        "g1": "PASS" if np.mean(cc_rows["bicubic"]) >= 0.30 else "FAIL",
        "g2": ("PASS" if r_cc >= 0.9 else ("WARN" if r_cc >= 0.7 else "FAIL")),
        "g3": round(float(np.mean(stdr_rows["sen2sr"])), 3),
        "g4": round(float(np.nanmean(grid_rows["sen2sr"])), 3),
        "g5": "PASS" if np.mean(rows["sen2sr"]) <= 45.0 else "FAIL",
        "delta_floor_db": round(float(np.mean(rows["bicubic"]) - np.mean(rows_floor)), 3),
    }
    d = derive(card)

    out = {
        "method": card["method"],
        "protocol": (f"WorldStrat real cross-sensor, SEN2SR native x4: LR {LR_CROP}x{LR_CROP} "
                     f"S2 10m RGBN (B04,B03,B02,B08) -> SR {SR_SIZE} (2.5m); "
                     f"GT = SPOT reference resampled to {SR_SIZE} over the same footprint; "
                     f"per-scene normalised space; val tail {100*val_frac:.0f}% of {n}"),
        "n_test": n_used, "val_dr": round(val_dr, 6),
        "scene_ids": scene_ids, "skipped": skipped,
        "psnr_mean": {a: round(float(np.mean(rows[a])), 3) for a in rows},
        "psnr_floor_mean": round(float(np.mean(rows_floor)), 3),
        "delta_model_db": round(gain, 3),
        "delta_model_paired": pt_model,
        "delta_model_nohc_db": round(float(np.mean(rows["sen2sr_nohc"])
                                           - np.mean(rows["bicubic"])), 3),
        "delta_model_nohc_paired": pt_nohc,
        "dominant_period": {a: (max(set(period_rows[a]), key=period_rows[a].count)
                                if period_rows[a] else -1) for a in period_rows},
        "delta_floor_db_mean": round(float(np.mean(d_floor)), 3),
        "delta_floor_paired": pt_floor,
        "cc_mean": {a: round(float(np.mean(cc_rows[a])), 4) for a in cc_rows},
        "cc_ratio_model_over_bicubic": round(r_cc, 4),
        "std_ratio_mean": {a: round(float(np.mean(stdr_rows[a])), 4) for a in stdr_rows},
        "periodicity_mean": {a: round(float(np.nanmean(grid_rows[a])), 3) for a in grid_rows},
        "per_sample": {"psnr_bicubic": [round(x, 4) for x in rows["bicubic"]],
                       "psnr_sen2sr": [round(x, 4) for x in rows["sen2sr"]],
                       "psnr_floor": [round(x, 4) for x in rows_floor],
                       "delta_model": [round(x, 4) for x in d_model]},
        "card": card,
        "verdict": {"credibility": d["credibility"], "regime": d["regime"],
                    "gate_verdict": d["gate_verdict"]},
        "notes": [
            "权重/掩码为 SEN2SR 官方发布物，未做任何掩码外推或重采样（掩码 512 强制 SR=512）。",
            "输入为原始反射率 [0,1]，波段顺序 (B04,B03,B02,B08) 与 SEN2SR 训练域一致。",
            "SEN2SR 输出与 GT 均用数据管线逐场景 (mu,sd) 映射到 per_scene 归一化空间；未用 GT 知情校正。",
            f"该臂在原生 ×4 体制评测（主文 E1 尺度轴 ×4 点）；不与 6.63× 臂的绝对 PSNR 直接横比，只比 Δ_model/Δ_floor。",
            "sen2sr_nohc = 同一网络去掉 Fourier 硬约束（只 clamp(min=0)），用于定位周期伪影来源并量化硬约束贡献。",
        ],
    }
    path = os.path.join(OUT_DIR, "ws_arm.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[saved] {path}")
    print(f"  n={n_used}  val_dr={val_dr:.4f}")
    print(f"  PSNR  sen2sr={out['psnr_mean']['sen2sr']:.3f}  bicubic={out['psnr_mean']['bicubic']:.3f}"
          f"  floor={out['psnr_floor_mean']:.3f}")
    print(f"  Δ_model  = {gain:+.3f} dB  (p={pt_model['p']:.3g}, CI={pt_model['ci95']}, d={pt_model['d']})")
    print(f"  Δ_model(去 Fourier 硬约束的裸网络) = {out['delta_model_nohc_db']:+.3f} dB "
          f"(p={pt_nohc['p']:.3g})")
    print(f"  主导周期 = bicubic {out['dominant_period']['bicubic']} | "
          f"sen2sr {out['dominant_period']['sen2sr']}")
    print(f"  Δ_floor  = {out['delta_floor_db_mean']:+.3f} dB")
    print(f"  gates    = G1 {card['g1']} | G2 {card['g2']} (CC ratio {r_cc:.3f}) | "
          f"G3 {card['g3']} | G4 {card['g4']} | G5 {card['g5']}")
    print(f"  verdict  = {d['credibility']} | {d['regime']} | {d['gate_verdict']}")


if __name__ == "__main__":
    main()
