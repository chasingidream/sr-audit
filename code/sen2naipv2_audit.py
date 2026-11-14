#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sen2naipv2_audit.py —— 在 SEN2SR 的**原生发布分布**（SEN2NAIPv2 crosssensor）上复核。

动机
------------------------------------------------
`sen2sr_ws_audit.py` 把 SEN2SR 官方权重跑在**本项目 WorldStrat** 真实跨传感器对上，
得 Δ_model = −0.051 dB、G4 FAIL（主导周期 4）。审稿人会问：
  「这是不是你自己的桥接/裁剪造成的？」
本脚本回答该问题：在 SEN2SR **自己的训练/评测分布**（SEN2NAIPv2 crosssensor，
8000 对真实 S2 ↔ NAIP）上，用**同一套 SR-Audit 口径**复核，看结论是否复现。

数据入口（唯一可达路径）
------------------------------------------------
`SEN2NAIPv2/sen2naipv2-crosssensor.taco` = 9.72 GB，`tacoreader` 2.4.21 已把它判为
legacy 并直接抛错。故本脚本走 `code/sen2naipv2_probe.py` 的**字节区间直读**：
每个样本只需 ~1.23 MB（嵌套容器 blob 内含 lr 77 KB + hr 1.15 MB），
60 个样本 ≈ 74 MB，无需下载整包。GTiff 用 rasterio 解码（自带 zstd 后端）。

几何与口径（零近似）
------------------------------------------------
* 原生几何：lr (4,130,130) @10 m RGBN；hr (4,520,520) @2.5 m RGBN；同一 geotransform
  原点 ⇒ 像素 (0,0) 共配准，原生 ×4，无需重采样。
* SEN2SR 发布掩码硬绑 (512,512)（`HardConstraint` 逐元素相乘）⇒ 必须裁到 LR=128：
  取 lr[:, :128, :128] 与 hr[:, :512, :512]（同地面覆盖，1280 m）。**不重采样掩码**。
* 输入/输出均为原始反射率 [0,1]（taco 里为 uint16 ×10^4，除以 10000）。
* 主口径：反射率空间，4 波段 RGBN，data_range=1.0；另报 RGB 3 波段与逐场景归一化空间
  作为稳健性（Δ_model 应在各空间一致）。

用法
------------------------------------------------
  python code/sen2naipv2_audit.py --n 8 --smoke          # 冒烟（打印逐样本）
  python code/sen2naipv2_audit.py --n 60                 # 全量
输出：experiments/sen2sr_audit/sen2naip_arm.json
缓存：experiments/sen2sr_audit/sen2naip/s{idx:05d}_{lr,hr}.npy（断点续跑）
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
sys.path.insert(0, os.path.join(BASE, "code", "sota"))

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

import sen2naipv2_probe as P                                    # noqa: E402
from sen2sr_ws_audit import load_sen2sr, paired_t, MODEL_DIR    # noqa: E402
from eval_metrics import calc_psnr, calc_cc                     # noqa: E402
from sr_audit import bicubic_up, periodicity_scan               # noqa: E402
from sr_audit_report import derive                             # noqa: E402
from sen2sr.models.opensr_baseline.cnn import CNNSR            # noqa: E402

OUT_DIR = os.path.join(BASE, "experiments", "sen2sr_audit")
CACHE = os.path.join(OUT_DIR, "sen2naip")

LR_CROP = 128          # 掩码 512 = LR 128 × 4
SR_SIZE = 512
NATIVE_LR = 130
NATIVE_HR = 520


def select_indices(n_total, n, phase=0.5):
    """在 [0, n_total) 上均匀取样（确定性）。phase=0.5 取格心；改 phase 得到独立子集。"""
    if n >= n_total:
        return list(range(n_total))
    return [int(round((k + phase) * n_total / n - 0.5)) for k in range(n)]


def fetch_sample(df, i, outer_off):
    """取样本 i 的 lr/hr 反射率数组（(4,130,130)/(4,520,520)，值域 [0,1]），带磁盘缓存。"""
    lr_p = os.path.join(CACHE, f"s{i:05d}_lr.npy")
    hr_p = os.path.join(CACHE, f"s{i:05d}_hr.npy")
    if os.path.exists(lr_p) and os.path.exists(hr_p):
        return np.load(lr_p), np.load(hr_p)
    import rasterio
    from rasterio.io import MemoryFile

    info = P.sample_assets(df, i, outer_off)
    out = {}
    for name in ("lr", "hr"):
        ao, al, _ = info["assets"][name]
        raw = P._range_get(P.URL, ao, ao + al - 1)
        with MemoryFile(raw) as mf:
            with mf.open() as ds:
                out[name] = (ds.read().astype("float32") / 10000.0)
    os.makedirs(CACHE, exist_ok=True)
    np.save(lr_p, out["lr"])
    np.save(hr_p, out["hr"])
    return out["lr"], out["hr"]


def main():
    ap = argparse.ArgumentParser(description="SEN2SR × SEN2NAIPv2 原生分布复核")
    ap.add_argument("--n", type=int, default=60, help="样本数（在 8000 上均匀取样）")
    ap.add_argument("--phase", type=float, default=0.5,
                    help="采样相位 [0,1)；=0.5 为格心，改相位得独立子集（稳健性）")
    ap.add_argument("--smoke", action="store_true", help="打印逐样本指标")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "sen2naip_arm.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = "cpu"
    os.makedirs(CACHE, exist_ok=True)

    fo, tbl = P.read_outer_footer(verbose=False)
    df = tbl.to_pandas()
    n_total = len(df)
    idxs = select_indices(n_total, args.n, args.phase)
    print(f"[info] n_total={n_total}  sampled={len(idxs)}  first={idxs[:5]} last={idxs[-3:]}")

    model = load_sen2sr(device)
    import safetensors.torch as _st
    net_only = CNNSR(4, 4, 24, 4, True, False, 6)
    net_only.load_state_dict(_st.load_file(os.path.join(MODEL_DIR, "model.safetensor")))
    net_only.eval()
    for p in net_only.parameters():
        p.requires_grad = False
    print("[info] SEN2SR NonReference_RGBN_x4 loaded")

    # 各口径的容器
    rec = {sp: {k: [] for k in ("sen2sr", "bicubic", "sen2sr_nohc", "floor")}
           for sp in ("refl_rgbn", "refl_rgb", "norm_rgbn")}
    cc = {sp: {k: [] for k in ("sen2sr", "bicubic")} for sp in ("refl_rgbn", "norm_rgbn")}
    stdr = {sp: [] for sp in ("refl_rgbn", "norm_rgbn")}
    per = {sp: [] for sp in ("sen2sr", "bicubic", "sen2sr_nohc")}
    did = {sp: [] for sp in ("sen2sr", "bicubic")}
    corr_lr_hr, used, skipped = [], [], []

    for k, i in enumerate(idxs):
        try:
            lr, hr = fetch_sample(df, i, fo)
        except Exception as e:                                  # noqa: BLE001
            skipped.append({"idx": i, "reason": f"{type(e).__name__}: {e}"})
            print(f"  [skip {i}] {type(e).__name__}: {e}")
            continue
        if lr.shape[-2] < LR_CROP or hr.shape[-1] < SR_SIZE:
            skipped.append({"idx": i, "reason": f"shape {lr.shape}/{hr.shape}"})
            continue

        lr4 = torch.from_numpy(lr[:, :LR_CROP, :LR_CROP].copy())[None].float()   # (1,4,128,128)
        gt4 = torch.from_numpy(hr[:, :SR_SIZE, :SR_SIZE].copy())                 # (4,512,512)

        with torch.no_grad():
            sr4 = model(lr4)[0]                                  # (4,512,512) refl
            sr4_nohc = torch.clamp(net_only(lr4)[0], min=0.0)
        bic4 = bicubic_up(lr4, (SR_SIZE, SR_SIZE))[0]            # (4,512,512) refl

        # 逐场景归一化 (mu,sd) —— 与 WorldStrat 臂一致：由 LR RGB 反射率统计而来
        mu = float(lr4[0, :3].mean())
        sd = float(lr4[0, :3].std()) + 1e-6

        spaces = {
            "refl_rgbn": (sr4, bic4, gt4, 1.0),
            "refl_rgb": (sr4[:3], bic4[:3], gt4[:3], 1.0),
            "norm_rgbn": ((sr4 - mu) / sd, (bic4 - mu) / sd, (gt4 - mu) / sd,
                          float((gt4 - mu).div(sd).max() - (gt4 - mu).div(sd).min())),
        }
        for sp, (sr, bic, gt, dr) in spaces.items():
            rec[sp]["sen2sr"].append(float(calc_psnr(sr, gt, data_range=dr)))
            rec[sp]["bicubic"].append(float(calc_psnr(bic, gt, data_range=dr)))
            mimg = torch.full_like(gt, float(gt.mean()))
            rec[sp]["floor"].append(float(calc_psnr(mimg, gt, data_range=dr)))
            if sp != "refl_rgb":
                srn = (sr4_nohc - mu) / sd if sp == "norm_rgbn" else sr4_nohc
                gtn = gt if sp == "refl_rgbn" else gt
                rec[sp]["sen2sr_nohc"].append(float(calc_psnr(srn, gtn, data_range=dr)))

        for a, o in (("sen2sr", sr4), ("bicubic", bic4)):
            cc["refl_rgbn"][a].append(float(calc_cc(o, gt4)))
            stdr["refl_rgbn"].append(float(o.std() / (gt4.std() + 1e-8))) if a == "sen2sr" else None
            g, pg = periodicity_scan(o)
            per[a].append(float(g) if g == g else float("nan"))
            did[a].append(int(pg) if pg else -1)
        g, pg = periodicity_scan(sr4_nohc)
        per["sen2sr_nohc"].append(float(g) if g == g else float("nan"))

        # lr 与 box4x(hr) 的相关（确认是真跨传感器对而非合成下采样）——用原生 130/520 网格
        hd = hr.reshape(4, NATIVE_LR, 4, NATIVE_LR, 4).mean(axis=(2, 4))
        cc_lr = float(np.mean([np.corrcoef(lr[b].ravel(), hd[b].ravel())[0, 1] for b in range(4)]))
        corr_lr_hr.append(cc_lr)

        used.append(i)
        if args.smoke or (k + 1) % 10 == 0:
            dm = rec["refl_rgbn"]["sen2sr"][-1] - rec["refl_rgbn"]["bicubic"][-1]
            print(f"  [{k+1}/{len(idxs)}] idx={i}  Δ_model(refl)={dm:+.3f}  "
                  f"bic={rec['refl_rgbn']['bicubic'][-1]:.2f} sr={rec['refl_rgbn']['sen2sr'][-1]:.2f}  "
                  f"corr(lr,box4hr)={cc_lr:.3f}")

    if not used:
        raise SystemExit("no samples processed")

    # ---------------- 汇总 ----------------
    def mean(x):
        return float(np.nanmean(x))

    summary_sp = {}
    for sp in rec:
        b, s = rec[sp]["bicubic"], rec[sp]["sen2sr"]
        f = rec[sp]["floor"]
        summary_sp[sp] = {
            "psnr_sen2sr": round(mean(s), 3), "psnr_bicubic": round(mean(b), 3),
            "psnr_floor": round(mean(f), 3),
            "delta_model": round(mean(s) - mean(b), 3),
            "delta_floor": round(mean(b) - mean(f), 3),
            "paired": paired_t(s, b), "paired_floor": paired_t(b, f),
        }
    prim = "refl_rgbn"
    pt = summary_sp[prim]["paired"]
    pt_nohc = paired_t(rec[prim]["sen2sr_nohc"], rec[prim]["bicubic"])
    r_cc = mean(cc["refl_rgbn"]["sen2sr"]) / max(mean(cc["refl_rgbn"]["bicubic"]), 1e-6)

    card = {
        "method": "SEN2SR (native dist., SEN2NAIPv2 crosssensor)",
        "n_test": len(used),
        "audit_gain_db": summary_sp[prim]["delta_model"],
        "audit_p": pt["p"],
        "passes_floor": bool(summary_sp[prim]["psnr_sen2sr"] > summary_sp[prim]["psnr_floor"]),
        "g1": "PASS" if mean(cc["refl_rgbn"]["bicubic"]) >= 0.30 else "FAIL",
        "g2": ("PASS" if r_cc >= 0.9 else ("WARN" if r_cc >= 0.7 else "FAIL")),
        "g3": round(mean(stdr["refl_rgbn"]), 3),
        "g4": round(mean(per["sen2sr"]), 3),
        "g5": "PASS" if summary_sp[prim]["psnr_sen2sr"] <= 45.0 else "FAIL",
        "delta_floor_db": summary_sp[prim]["delta_floor"],
    }
    d = derive(card)

    out = {
        "method": card["method"],
        "protocol": (f"SEN2NAIPv2 crosssensor (SEN2SR native distribution): real S2 10m RGBN "
                     f"(B04,B03,B02,B08) LR {NATIVE_LR}->crop{LR_CROP} -> SR {SR_SIZE} (2.5m); "
                     f"GT = paired NAIP 2.5m RGBN {NATIVE_HR}->crop{SR_SIZE}, same geotransform "
                     f"origin; {len(used)} of {n_total} samples, evenly spaced; reflectance space"),
        "n_test": len(used), "used_idx": used, "skipped": skipped,
        "spaces": summary_sp,
        "primary_space": prim,
        "delta_model_nohc_db": round(mean(rec[prim]["sen2sr_nohc"]) - mean(rec[prim]["bicubic"]), 3),
        "delta_model_nohc_paired": pt_nohc,
        "cc_mean": {a: round(mean(cc["refl_rgbn"][a]), 4) for a in cc["refl_rgbn"]},
        "cc_ratio_model_over_bicubic": round(r_cc, 4),
        "std_ratio_mean": round(mean(stdr["refl_rgbn"]), 4),
        "periodicity_mean": {a: round(mean(per[a]), 3) for a in per},
        "dominant_period": {a: (max(set(did[a]), key=did[a].count) if did[a] else -1)
                            for a in did},
        "corr_lr_box4hr": {"mean": round(mean(corr_lr_hr), 4),
                           "min": round(float(np.nanmin(corr_lr_hr)), 4),
                           "max": round(float(np.nanmax(corr_lr_hr)), 4)},
        "per_sample_delta_model_refl": [round(a - b, 4) for a, b in
                                        zip(rec[prim]["sen2sr"], rec[prim]["bicubic"])],
        "card": card,
        "verdict": {"credibility": d["credibility"], "regime": d["regime"],
                    "gate_verdict": d["gate_verdict"]},
        "notes": [
            "数据入口 = 字节区间直读 .taco（tacoreader 2.4.21 拒收 .taco）；每样本 ~1.23MB。",
            "GTiff 由 rasterio 解码（自带 zstd 后端）；tifffile 因缺 compression.zstd 不可用。",
            "SEN2SR 权重/掩码为官方发布物；掩码 (512,512) 硬绑 SR=512 ⇒ LR 裁到 128（不重采样掩码）。",
            "lr/hr 均为真实采集（S2 ↔ NAIP），非合成下采样：corr(lr, box4x(hr)) 见 corr_lr_box4hr。",
            "主口径=反射率空间 RGBN data_range=1.0；另报 RGB 与逐场景归一化空间作稳健性。",
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[saved] {args.out}")
    print(f"  n={len(used)}/{n_total}  corr(lr,box4hr)={out['corr_lr_box4hr']['mean']:.3f}")
    for sp in summary_sp:
        v = summary_sp[sp]
        print(f"  [{sp:11s}] Δ_model={v['delta_model']:+.3f} dB (p={v['paired']['p']:.3g}) "
              f"Δ_floor={v['delta_floor']:+.3f}  PSNR sr/bic/floor="
              f"{v['psnr_sen2sr']:.2f}/{v['psnr_bicubic']:.2f}/{v['psnr_floor']:.2f}")
    print(f"  Δ_model(去硬约束裸网络) = {out['delta_model_nohc_db']:+.3f} dB "
          f"(p={pt_nohc['p']:.3g})")
    print(f"  周期: sen2sr {out['periodicity_mean']['sen2sr']:.2f} @ "
          f"{out['dominant_period']['sen2sr']}px | bicubic "
          f"{out['periodicity_mean']['bicubic']:.2f} @ {out['dominant_period']['bicubic']}px")
    print(f"  gates = G1 {card['g1']} | G2 {card['g2']} ({r_cc:.3f}) | G3 {card['g3']} | "
          f"G4 {card['g4']} | G5 {card['g5']}")
    print(f"  verdict = {d['credibility']} | {d['regime']} | {d['gate_verdict']}")


if __name__ == "__main__":
    main()
