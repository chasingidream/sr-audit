#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""audit_axis.py —— 三轴通用 SR 审计（sr_audit.py 的泛化版，支持 --cfg）。

背景：sr_audit.py 的数据装载硬编码 WorldStrat（经 eval_paired_mtxa.load_val），
只能审计 WorldStrat 各档。本脚本把同一套 G1–G5 门控证据计算泛化到任意
external cfg（PROBA-V / SEN2VENµS / WorldStrat），口径三重对齐：

  * 数据装载 / val 划分 / val_dr —— 与 eval_paired_floor.py 完全同口径
    （get_dataset + cfg.val_frac + val 目标全局 min/max 满量程）；
  * 门控证据（四预测器 + pair_cc + bias/std_ratio + FFT 网格比 + G1–G5 判定）
    —— 与 sr_audit.audit() 完全同构（直接复用其 bicubic_up / periodicity_scan /
    base_stage）；
  * ckpt = save_dir/<stage>/ 末轮（pick_ckpt 复用 eval_paired_floor，主口径=末轮不择优）。

额外输出（sr_audit 没有、审计服务 /audit 需要的量）：
  * gain_db        = model PSNR − bicubic PSNR（逐样本配对均值差）
  * delta_floor_db = bicubic PSNR − mean PSNR（Δ_floor，数据属性）
  * paired_psnr    = 逐样本配对 t 检验（CI95 / p / Cohen's d，scipy）

用法：
  python code/audit_axis.py --cfg code/configs/external_probav_n300_s42.py \
      --stage mtxa_aligned --save experiments/audit_axis_probav_n300_s42.json
  python code/audit_axis.py --cfg code/configs/external_sen2venus.py \
      --stage sisr --save experiments/audit_axis_s2v_sisr.json
  python code/audit_axis.py --cfg code/configs/external_worldstrat.py \
      --stage mtxa_aligned --save experiments/audit_axis_ws_mtxa_aligned.json
"""
from __future__ import annotations

import argparse
import json
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
from eval_metrics import calc_psnr, calc_ssim, calc_cc, calc_sam  # noqa: E402
from models.sr_model import build_model  # noqa: E402
from sr_audit import bicubic_up, periodicity_scan, base_stage  # noqa: E402
from eval_paired_floor import load_cfg, pick_ckpt, paired_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="三轴通用 SR 审计（G1–G5 证据卡）")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--max_samples", type=int, default=0, help=">0 时只审前 N 个样本（冒烟测试）")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.stage:
        cfg["stage"] = args.stage
    stage = cfg["stage"]
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    # ---------- 数据：与 eval_paired_floor.py 完全同口径 ----------
    bsz = int(cfg.get("batch_size", 4))
    val_ds = get_dataset(cfg, project_root=BASE, deterministic=True)
    val_frac = float(cfg.get("val_frac", 0.1))          # 须与 train.py 一致
    n_train = max(1, min(len(val_ds) - 1, int((1.0 - val_frac) * len(val_ds))))
    val_ds = Subset(val_ds, list(range(n_train, len(val_ds))))
    loader = DataLoader(val_ds, batch_size=bsz, shuffle=False, num_workers=0)

    mn, mx = float("inf"), float("-inf")
    for vb in loader:
        t = vb["target"].float()
        mn, mx = min(mn, float(t.min())), max(mx, float(t.max()))
    val_dr = (mx - mn) if mx - mn > 1e-8 else 1.0
    loader = DataLoader(val_ds, batch_size=bsz, shuffle=False, num_workers=0)

    # ---------- 模型：save_dir/<stage>/ 末轮 ----------
    ckpt_path = pick_ckpt(cfg["save_dir"], stage, args.epoch)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    print(f"[info] cfg={args.cfg}  stage={stage}  device={device}  bsz={bsz}")
    print(f"[info] ckpt={ckpt_path} (ep{ck.get('epoch')})  n_val={len(val_ds)}  val_dr={val_dr:.6f}")

    # ---------- 门控与 train.py/eval_paired_floor 一致 ----------
    _base = base_stage(stage)
    if _base in ("sisr", "baseline"):
        use_rsa_eval = False
    else:
        use_rsa_eval = not _base.endswith("noxa")
    print(f"[info] stage gating: base={_base}  use_rsa={use_rsa_eval}  use_ref=False")

    # sr_audit 铁律：仅当数据本身在 [0,1]（val_dr≈1）才允许 clamp，
    # per_scene/global 归一化数据（val_dr≠1）clamp 会造成 G1/G2/G3 假 FAIL。
    _do_clamp = abs(val_dr - 1.0) < 1e-6

    def _c(x):
        return x.clamp(0, 1) if _do_clamp else x

    acc = {k: [] for k in ("mean", "bicubic", "model")}
    rows_model, rows_bic, rows_mean = [], [], []          # 逐样本（配对检验用）
    pair_cc, bias_list, stdr_list, grid_m, grid_b = [], [], [], [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            lrs = batch["lrs"].float().to(device)
            tgt = batch["target"].float().to(device)
            lrs_in = lrs[:, :1] if _base.startswith("sisr") else lrs
            pred = model(lrs_in, batch["hr_ref"].float().to(device),
                         return_aux=False, use_rsa=use_rsa_eval, use_ref=False)
            if not torch.is_tensor(pred):
                pred = pred[0]
            if tuple(pred.shape[-2:]) != tuple(tgt.shape[-2:]):
                pred = F.interpolate(pred, size=tuple(tgt.shape[-2:]),
                                     mode="bilinear", align_corners=True)
            bic = bicubic_up(lrs[:, 0], size=tuple(tgt.shape[-2:])).to(device)

            for i in range(pred.shape[0]):
                g = tgt[i]
                mnc = torch.full_like(g, float(g.mean()))
                outs = {"mean": mnc, "bicubic": bic[i], "model": pred[i]}
                row = {}
                for name, out in outs.items():
                    o = _c(out)
                    m = {"psnr": float(calc_psnr(o, g, data_range=val_dr)),
                         "ssim": float(calc_ssim(o, g, data_range=val_dr)),
                         "cc": float(calc_cc(o, g)),
                         "sam_deg": float(calc_sam(o, g))}
                    acc[name].append(m)
                    row[name] = m
                rows_model.append(row["model"])
                rows_bic.append(row["bicubic"])
                rows_mean.append(row["mean"])
                pair_cc.append(float(calc_cc(_c(bic[i]), g)))
                p = _c(pred[i])
                bias_list.append(abs(float(p.mean() - g.mean())))
                stdr_list.append(float(p.std() / (g.std() + 1e-8)))
                grid_m.append(periodicity_scan(p))
                grid_b.append(periodicity_scan(bic[i].clamp(0, 1)))
                n += 1
                if args.max_samples and n >= args.max_samples:
                    break
            if args.max_samples and n >= args.max_samples:
                break

    def agg(d):
        return {k: float(np.mean([x[k] for x in d])) for k in d[0]}

    res = {
        "stage": stage,
        "axis_cfg": os.path.basename(args.cfg),
        "ckpt_path": os.path.relpath(ckpt_path, BASE),
        "n_samples": n,
        "val_dr": float(val_dr),
        "predictors": {k: agg(v) for k, v in acc.items()},
        "pair_cc": float(np.mean(pair_cc)),
        "bias": float(np.mean(bias_list)),
        "std_ratio": float(np.mean(stdr_list)),
        "grid_ratio_model": float(np.nanmean([a for a, _ in grid_m])),
        "grid_ratio_bicubic": float(np.nanmean([a for a, _ in grid_b])),
        "grid_period_model": int(round(float(np.median([b for _, b in grid_m])))),
        "grid_period_bicubic": int(round(float(np.median([b for _, b in grid_b])))),
    }
    res["gain_db"] = res["predictors"]["model"]["psnr"] - res["predictors"]["bicubic"]["psnr"]
    res["delta_floor_db"] = res["predictors"]["bicubic"]["psnr"] - res["predictors"]["mean"]["psnr"]
    res["paired_psnr"] = paired_test(rows_model, rows_bic).get("psnr", {})

    # ---------------- 门控：与 sr_audit.py 完全同阈值 ----------------
    g = {}
    g["G1_配对完整性"] = ("PASS" if res["pair_cc"] >= 0.30 else "FAIL",
                          f"CC(bic,GT)={res['pair_cc']:.3f} (阈值≥0.30)")
    r_cc = res["predictors"]["model"]["cc"] / max(res["predictors"]["bicubic"]["cc"], 1e-6)
    g["G2_结构保真"] = ("PASS" if r_cc >= 0.9 else ("WARN" if r_cc >= 0.7 else "FAIL"),
                        f"model CC / bicubic CC = {r_cc:.2f}")
    g["G3_输出统计"] = ("PASS" if 0.5 <= res["std_ratio"] <= 1.8 else "FAIL",
                        f"std_ratio={res['std_ratio']:.2f}, |bias|={res['bias']:.4f}")
    g["G4_周期伪影"] = ("PASS" if res["grid_ratio_model"] < 2.5 else "WARN",
                        f"ratio={res['grid_ratio_model']:.2f} @ period="
                        f"{res['grid_period_model']}px (bicubic "
                        f"{res['grid_ratio_bicubic']:.2f} @ "
                        f"{res['grid_period_bicubic']}px)")
    LEAK_STAGES = ("mt", "mt+rsa", "end2end", "full",
                   "xs_aligned", "xs_misaligned_noxa", "xs_misaligned_xa")
    leak = stage in LEAK_STAGES or res["predictors"]["model"]["psnr"] > 45.0
    g["G5_泄漏风险"] = ("FAIL" if leak else "PASS",
                        "HR 参与重建或 PSNR>45dB" if leak else "未见泄漏")
    res["gates"] = {k: {"verdict": v[0], "detail": v[1]} for k, v in g.items()}
    res["overall"] = "FAIL" if any(v[0] == "FAIL" for v in g.values()) else (
        "WARN" if any(v[0] == "WARN" for v in g.values()) else "PASS")

    # ------- derive-schema 顶层字段（sr_audit_report.derive / 审计服务 /assess 直接可判，
    #         与 sr_audit.py 原始证据字段并存，一张卡两套 schema）-------
    res["method"] = stage
    res["n_test"] = n
    res["audit_gain_db"] = res["gain_db"]
    res["audit_p"] = res["paired_psnr"].get("p")
    res["passes_floor"] = bool(res["predictors"]["model"]["psnr"]
                               > res["predictors"]["mean"]["psnr"])
    res["g1"] = res["gates"]["G1_配对完整性"]["verdict"]
    res["g2"] = res["gates"]["G2_结构保真"]["verdict"]
    res["g3"] = res["std_ratio"]            # 数值型门：derive 按 [0.5,1.8] 判
    res["g4"] = res["grid_ratio_model"]     # 数值型门：derive 按 <2.5 判
    res["g5"] = res["gates"]["G5_泄漏风险"]["verdict"]

    # ---------------- 控制台摘要 ----------------
    p = res["predictors"]
    print(f"\n[audit] stage={stage}  n={n}  overall={res['overall']}")
    for k, v in res["gates"].items():
        print(f"  {v['verdict']:<5} {k:<14} {v['detail']}")
    print(f"  PSNR: mean {p['mean']['psnr']:.2f} | bicubic {p['bicubic']['psnr']:.2f}"
          f" | model {p['model']['psnr']:.2f}")
    print(f"  gain(model−bicubic)={res['gain_db']:+.2f} dB   "
          f"Δ_floor(bicubic−mean)={res['delta_floor_db']:+.2f} dB")
    pp = res["paired_psnr"]
    if pp:
        print(f"  paired: Δ={pp['mean_diff']:+.3f}  95%CI=[{pp['ci95'][0]:+.3f},{pp['ci95'][1]:+.3f}]"
              f"  p={pp['p']:.3g}  d={pp['cohen_d']:+.2f}  n={pp['n']}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.save}")
    if args.report:
        L = [f"# SR 审计报告 — `{stage}`（{os.path.basename(args.cfg)}）", "",
             f"- ckpt：`{res['ckpt_path']}`（末轮，Tier-B）",
             f"- 样本数：{n}　val_dr：{val_dr:.4f}",
             f"- **总判定：{res['overall']}**", "",
             "## 1. 四预测器技能地板", "| 预测器 | PSNR | SSIM | CC | SAM° |", "|---|---|---|---|---|"]
        for k in ("mean", "bicubic", "model"):
            d = p[k]
            L.append(f"| {k} | {d['psnr']:.2f} | {d['ssim']:.4f} | {d['cc']:+.4f} | {d['sam_deg']:.3f} |")
        L += ["", "## 2. 审计门控", "| 门 | 判定 | 依据 |", "|---|---|---|"]
        for k, v in res["gates"].items():
            L.append(f"| {k} | **{v['verdict']}** | {v['detail']} |")
        L += ["", "## 3. 关键量",
              f"- gain(model−bicubic) = {res['gain_db']:+.2f} dB",
              f"- Δ_floor(bicubic−mean) = {res['delta_floor_db']:+.2f} dB"]
        if pp:
            L.append(f"- 逐样本配对：Δ={pp['mean_diff']:+.3f} dB，"
                     f"95%CI=[{pp['ci95'][0]:+.3f},{pp['ci95'][1]:+.3f}]，"
                     f"p={pp['p']:.3g}，Cohen's d={pp['cohen_d']:+.2f}（n={pp['n']}）")
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        print(f"[saved] {args.report}")


if __name__ == "__main__":
    main()
