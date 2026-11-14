#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S2<->L8 跨传感器 3x：模型增益 Δ_model 评测（全 22 ROI，n=22，与 e1 Δ_floor 臂同口径）。

关键修正（2026-09-30）：L8/S2 GeoTIFF 含 ~4% 无数据 NaN 边界，且 L8 与 S2 的 NaN 边界
位置不同（30m vs 10m）。若在全图（NaN 填 0）上算 PSNR/CC，错位零区会把相关性彻底打没
（G1≈0 伪失败）。此处严格复刻 e1 脚本 df_core 口径——只在 L8-有效 ∩ S2-有效 重叠区评测，
且用灰度（各通道均值）以与正文 S2↔L8 Δ_floor=+2.24 dB 的"conservative grayscale convention"
一致；同时附 RGB 逐通道均值作为补充。

输出：
  - 组均值（灰度 PSNR / CC）：mean / bicubic / model
  - 配对 t 检验：model−bicubic（Δ_model）、bicubic−mean（Δ_floor）
  - G1 = 每个 ROI 的 CC(bicubic,GT)（仅重叠区），及通过比例
  - 保存 json 到 --save

用法:
  python code/eval_s2l8.py --cfg code/configs/external_s2l8.py --epoch 79 \
      --save experiments/paired_s2l8.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as _st
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import _dll_guard  # noqa: E402

from data.dataset import get_dataset  # noqa: E402
from models.sr_model import build_model  # noqa: E402

METRICS = ["psnr", "cc"]


def load_cfg(path):
    spec = importlib.util.spec_from_file_location("user_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.CONFIG)


def pick_ckpt(save_dir: str, stage: str, epoch: int | None):
    d = os.path.join(save_dir, stage)
    if not os.path.isdir(d):
        raise FileNotFoundError(d)
    if epoch is not None:
        p = os.path.join(d, f"ckpt_ep{epoch}.pt")
        if os.path.exists(p):
            return p
    eps = []
    for f in os.listdir(d):
        if f.startswith("ckpt_ep") and f.endswith(".pt"):
            try:
                eps.append((int(f[len("ckpt_ep"):-3]), f))
            except ValueError:
                pass
    if not eps:
        raise FileNotFoundError(f"no ckpt in {d}")
    return os.path.join(d, max(eps)[1])


def psnr_cc_on_common(pred_g, gt_g, common):
    """pred_g, gt_g: (H,W) 灰度；common: bool (H,W)。返回 (psnr, cc) 仅在 common 像素。"""
    p = pred_g[common]
    g = gt_g[common]
    if p.size < 100:
        return float("nan"), float("nan")
    dr = float(g.max() - g.min())
    if dr <= 0:
        dr = 1.0
    mse = float(((p - g) ** 2).mean())
    if mse <= 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10(dr * dr / mse)
    if p.std() > 1e-8 and g.std() > 1e-8:
        cc = float(np.corrcoef(p.flatten(), g.flatten())[0, 1])
    else:
        cc = float("nan")
    return psnr, cc


def gray(arr3):
    return arr3.mean(axis=0)  # (H,W)


def paired_test(a, b):
    out = {}
    for k in ("psnr", "cc"):
        xs = np.asarray([r[k] for r in a], float)
        ys = np.asarray([r[k] for r in b], float)
        n = min(len(xs), len(ys))
        xs, ys = xs[:n], ys[:n]
        diffs = xs - ys
        mean = float(diffs.mean())
        sd = float(diffs.std(ddof=1))
        se = sd / (n ** 0.5)
        if sd > 0 and n >= 2:
            t_stat, p_val = _st.ttest_rel(xs, ys)
            t_stat, p_val = float(t_stat), float(p_val)
            lo, hi = _st.t.interval(0.95, n - 1, loc=mean, scale=se)
            ci = (float(lo), float(hi))
            d = mean / sd
        else:
            t_stat, p_val, ci, d = float("nan"), float("nan"), (mean, mean), float("nan")
        out[k] = dict(mean_diff=mean, sd=sd, ci95=ci, t=t_stat, p=p_val,
                      cohen_d=float(d), n=n)
    return out


def fmt(r):
    if r["p"] != r["p"]:
        star = ""
    elif r["p"] < 0.001:
        star = "***"
    elif r["p"] < 0.01:
        star = "**"
    elif r["p"] < 0.05:
        star = "*"
    else:
        star = ""
    sig = "显著" if (r["p"] == r["p"] and r["p"] < 0.05 and r["ci95"][0] > 0) else "不显著"
    return (f"    Δ={r['mean_diff']:+.4f}  sd={r['sd']:.4f}  "
            f"95%CI=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
            f"t={r['t']:+.2f}  p={r['p']:.4f}{star}  d={r['cohen_d']:+.2f}  → {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--ids", default=None,
                    help="逗号分隔的 ROI id 列表（4-fold CV 评测 held-out fold 用）；"
                         "省略则用 cfg 的 use_ids / 全量")
    args = ap.parse_args()

    _dll_guard.hide()
    try:
        cfg = load_cfg(args.cfg)
        if args.stage:
            cfg["stage"] = args.stage
        if args.ids:
            cfg.setdefault("dataset", {})["use_ids"] = [int(x) for x in args.ids.split(",")]
        stage = cfg["stage"]
        device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

        val_ds = get_dataset(cfg, project_root=BASE, deterministic=True)
        n_total = len(val_ds)
        bsz = int(cfg.get("batch_size", 4))
        loader = DataLoader(val_ds, batch_size=bsz, shuffle=False, num_workers=0)

        ckpt_path = pick_ckpt(cfg["save_dir"], stage, args.epoch)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(cfg).to(device)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()
        print(f"[info] cfg={args.cfg}  stage={stage}  device={device}  n={n_total}  "
              f"ckpt=ep{ck.get('epoch')}")

        base = stage
        ch = True
        while ch:
            ch = False
            for suf in ("_l1", "_ct", "_rbc", "_zi"):
                if base.endswith(suf):
                    base = base[:-len(suf)]; ch = True
            if __import__("re").search(r"_s\d+$", base):
                base = __import__("re").sub(r"_s\d+$", "", base); ch = True
        use_rsa = (base not in ("sisr", "baseline")) and not base.endswith("noxa")
        print(f"[info] stage gating base={base} use_rsa={use_rsa} use_ref=False")

        rows_model, rows_bic, rows_mean = [], [], []
        g1_list = []
        names = []
        with torch.no_grad():
            for batch in loader:
                lrs = batch["lrs"].to(device)
                hr_ref = batch["hr_ref"].to(device)
                gt = batch["target"].to(device)
                pred, _ = model(lrs, hr_ref, return_aux=True,
                                use_rsa=use_rsa, use_ref=False)
                mean_map = gt.mean(dim=(2, 3), keepdim=True).expand_as(gt)
                bic = F.interpolate(lrs[:, 0], size=gt.shape[-2:], mode="bicubic",
                                    align_corners=False)
                lv = batch["lr_valid"].numpy()       # (B,1,64,64)
                hv = batch["hr_valid"].numpy()       # (B,1,192,192)
                sc = int(cfg.get("scale", 3))
                for i in range(pred.shape[0]):
                    p = pred[i].cpu().numpy().astype(np.float64)   # (3,192,192)
                    b = bic[i].cpu().numpy().astype(np.float64)
                    m = mean_map[i].cpu().numpy().astype(np.float64)
                    g = gt[i].cpu().numpy().astype(np.float64)
                    pg, bg, mg, gg = gray(p), gray(b), gray(m), gray(g)
                    # 重叠区：S2-有效 ∩ L8-有效(最近邻上采样到 192)
                    lvu = zoom(lv[i, 0], (sc, sc), order=0) > 0.5
                    common = (hv[i, 0] > 0.5) & lvu
                    pm, cm_ = psnr_cc_on_common(pg, gg, common)
                    pb, cb_ = psnr_cc_on_common(bg, gg, common)
                    pmean, _ = psnr_cc_on_common(mg, gg, common)
                    # 逐通道 RGB PSNR（同样重叠区）
                    rgb_psnr = []
                    for c in range(p.shape[0]):
                        pp, _ = psnr_cc_on_common(p[c], g[c], common)
                        rgb_psnr.append(pp)
                    rows_model.append({"psnr": pm, "cc": cm_,
                                       "psnr_rgb_mean": float(np.nanmean(rgb_psnr))})
                    rows_bic.append({"psnr": pb, "cc": cb_})
                    rows_mean.append({"psnr": pmean, "cc": float("nan")})
                    g1_list.append(cb_)
                    names.append(str(batch.get("name", ["?"] * pred.shape[0])[i]) if "name" in batch else "?")

        def gmean(rows, key):
            vals = [r[key] for r in rows if r[key] == r[key]]
            return float(np.mean(vals)) if vals else float("nan")

        print(f"\n[info] 组均值（灰度 PSNR / CC，n={len(rows_model)}，仅重叠区）")
        print(f"{'预测器':10s} {'PSNR':>8s} {'CC':>8s}")
        for nm, r in (("mean", rows_mean), ("bicubic", rows_bic), ("model", rows_model)):
            print(f"{nm:10s} {gmean(r,'psnr'):8.2f} {gmean(r,'cc'):+8.4f}")
        print(f"[info] 组均值 RGB 逐通道平均 PSNR: model={gmean(rows_model,'psnr_rgb_mean'):.2f}  "
              f"bicubic={gmean(rows_bic,'psnr'):.2f}")

        print(f"\n===== model − bicubic（Δ_model，灰度）=====")
        for k, v in paired_test(rows_model, rows_bic).items():
            print(f"  {k}:")
            print(fmt(v))
        print(f"\n===== bicubic − mean（Δ_floor，数据层地板，灰度）=====")
        for k, v in paired_test(rows_bic, rows_mean).items():
            print(f"  {k}:")
            print(fmt(v))

        g1_arr = np.array([x for x in g1_list if x == x])
        print(f"\n[info] G1 (CC bicubic,GT, 仅重叠区): mean={g1_arr.mean():.3f}  "
              f"frac>=0.30={float((g1_arr>=0.30).mean()):.3f}  min={g1_arr.min():.3f}")

        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump({
                    "cfg": args.cfg, "stage": stage, "n": len(rows_model),
                    "note": "仅在 L8-有效 ∩ S2-有效 重叠区评测；灰度与 e1 Δ_floor=+2.24dB 同口径",
                    "group_mean_gray": {
                        "mean": {k: gmean(rows_mean, k) for k in METRICS},
                        "bicubic": {k: gmean(rows_bic, k) for k in METRICS},
                        "model": {k: gmean(rows_model, k) for k in METRICS},
                        "model_psnr_rgb_mean": gmean(rows_model, "psnr_rgb_mean"),
                    },
                    "paired_model_vs_bicubic_gray": paired_test(rows_model, rows_bic),
                    "paired_bicubic_vs_mean_gray": paired_test(rows_bic, rows_mean),
                    "g1_bicubic_gt": {"mean": float(g1_arr.mean()),
                                      "frac_pass": float((g1_arr >= 0.30).mean()),
                                      "min": float(g1_arr.min()),
                                      "per_sample": g1_list},
                    # 逐 ROI 行（4-fold CV 汇总用）：name/三预测器灰度 PSNR/bicubic CC
                    "per_roi": [
                        {"name": names[i],
                         "psnr_model": rows_model[i]["psnr"],
                         "psnr_bicubic": rows_bic[i]["psnr"],
                         "psnr_mean": rows_mean[i]["psnr"],
                         "cc_bicubic": rows_bic[i]["cc"],
                         "psnr_model_rgb_mean": rows_model[i].get("psnr_rgb_mean"),
                         } for i in range(len(rows_model))],
                }, f, ensure_ascii=False, indent=2)
            print(f"\n[saved] {args.save}")
    finally:
        _dll_guard.restore()
        print("[info] vcomp140.dll restored")


if __name__ == "__main__":
    main()
