#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐样本配对显著性检验：model vs bicubic（兼报 vs mean 技能地板）。

为何需要它（#22 铁律：小验证集必须配对 + 报 CI/p/效应量）：
  跨数据集技能地板评估 eval_skill_floor_cfg.py 只给【组均值】，且它的"组均值"是
  **整批 4 张池化 MSE 之后再平均**（非标准）。本脚本发现：在 SR 标准的【逐样本 PSNR
  平均】口径下，WorldStrat 的 model 并不比 bicubic 高（+0.73 是批池化聚合产物）。
  故本脚本同时输出两种口径，并对每种口径做配对 t 检验：

  口径 A（逐样本，SR 标准）：每个验证样本单独算 PSNR，再对 60 个样本平均；配对 = 同一样本 model−bicubic。
  口径 B（整批池化，eval_skill_floor_cfg 原口径）：每 4 张一批算池化 PSNR，再对 15 批平均；
        配对 = 同一批 model−bicubic（n=15，功效较低但口径与 #22 表完全一致，可对账）。

判据：p < 0.05 且 95%CI 不跨 0 → 差异显著。

用法:
  python code/eval_paired_floor.py --cfg code/configs/external_worldstrat.py \
      --stage mtxa_aligned --save experiments/paired_worldstrat.json
  python code/eval_paired_floor.py --cfg code/configs/external_probav.py \
      --stage mtxa_aligned --save experiments/paired_probav.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import _dll_guard  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import stats as _st  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from data.dataset import get_dataset  # noqa: E402
from eval_metrics import evaluate_all  # noqa: E402
from models.sr_model import build_model  # noqa: E402

METRICS = ["psnr", "ssim", "cc", "sam_deg"]


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
    for f in glob.glob(os.path.join(d, "ckpt_ep*.pt")):
        m = re.search(r"ckpt_ep(\d+)\.pt", f)
        if m:
            eps.append((int(m.group(1)), f))
    if not eps:
        raise FileNotFoundError(f"no ckpt in {d}")
    return max(eps)[1]


def paired_test(a, b):
    """逐指标配对 t 检验（a − b），返回可读 dict。"""
    out = {}
    for k in METRICS:
        xs = np.asarray([r[k] for r in a], dtype=float)
        ys = np.asarray([r[k] for r in b], dtype=float)
        n = min(len(xs), len(ys))
        if n < 2:
            continue
        xs, ys = xs[:n], ys[:n]
        diffs = xs - ys
        mean = float(diffs.mean())
        sd = float(diffs.std(ddof=1))
        se = sd / (n ** 0.5)
        if sd > 0:
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
    ap.add_argument("--save", default=None, help="保存逐样本 + 配对结果到 json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    _dll_guard.hide()
    try:
        cfg = load_cfg(args.cfg)
        if args.stage:
            cfg["stage"] = args.stage
        stage = cfg["stage"]
        device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                              else "cpu")

        bsz = int(cfg.get("batch_size", 4))
        val_ds = get_dataset(cfg, project_root=BASE, deterministic=True)
        val_frac = float(cfg.get("val_frac", 0.1))   # 须与 train.py 完全一致
        n_train = max(1, min(len(val_ds) - 1, int((1.0 - val_frac) * len(val_ds))))
        val_ds = Subset(val_ds, list(range(n_train, len(val_ds))))
        loader = DataLoader(val_ds, batch_size=bsz, shuffle=False, num_workers=0)

        mn, mx = float("inf"), float("-inf")
        for vb in loader:
            t = vb["target"].float()
            mn, mx = min(mn, float(t.min())), max(mx, float(t.max()))
        val_dr = (mx - mn) if mx - mn > 1e-8 else 1.0
        loader = DataLoader(val_ds, batch_size=bsz, shuffle=False, num_workers=0)

        ckpt_path = pick_ckpt(cfg["save_dir"], stage, args.epoch)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(cfg).to(device)
        try:
            model.load_state_dict(ck["model"], strict=True)
            print("[info] ckpt loaded: strict=True, all keys matched")
        except Exception as e:
            print(f"[WARN] strict load failed -> {type(e).__name__}: {e}")
            model.load_state_dict(ck["model"], strict=False)
        model.eval()
        print(f"[info] cfg={args.cfg}  stage={stage}  device={device}  bsz={bsz}")
        print(f"[info] ckpt={ckpt_path} (ep{ck.get('epoch')})  n_val={len(val_ds)}  val_dr={val_dr:.6f}")

        # 门控必须与 train.py 完全一致：先剥离变体后缀（_s<N> 多种子 / _l1/_ct/_rbc/_zi），
        # 用基础名判 use_rsa；否则 mtxa_aligned_s42 / sisr_s42 会走错分支。
        _base = stage
        _ch = True
        while _ch:
            _ch = False
            for _suf in ("_l1", "_ct", "_rbc", "_zi"):
                if _base.endswith(_suf):
                    _base = _base[:-len(_suf)]
                    _ch = True
            if re.search(r"_s\d+$", _base):
                _base = re.sub(r"_s\d+$", "", _base)
                _ch = True
        if _base in ("sisr", "baseline"):
            use_rsa_eval = False
        else:
            use_rsa_eval = not _base.endswith("noxa")
        print(f"[info] stage gating: base={_base}  use_rsa={use_rsa_eval}  use_ref=False")

        # 口径 A：逐样本  rows_*_ps（每样本一个 dict）
        # 口径 B：整批池化 rows_*_batch（每批一个 dict，与 eval_skill_floor_cfg 同口径）
        rows_model_ps, rows_bic_ps, rows_mean_ps = [], [], []
        rows_model_b, rows_bic_b, rows_mean_b = [], [], []
        with torch.no_grad():
            for batch in loader:
                lrs = batch["lrs"].to(device)
                hr_ref = batch["hr_ref"].to(device)
                gt = batch["target"].to(device)
                pred, _ = model(lrs, hr_ref, return_aux=True,
                                use_rsa=use_rsa_eval, use_ref=False)
                mean_map = gt.mean(dim=(2, 3), keepdim=True).expand_as(gt)
                bic = F.interpolate(lrs[:, 0], size=gt.shape[-2:], mode="bicubic",
                                    align_corners=False)
                # 口径 B：整批池化（eval_skill_floor_cfg 约定）
                rows_model_b.append({k: float(evaluate_all(pred, gt, data_range=val_dr)[k])
                                     for k in METRICS})
                rows_bic_b.append({k: float(evaluate_all(bic, gt, data_range=val_dr)[k])
                                   for k in METRICS})
                rows_mean_b.append({k: float(evaluate_all(mean_map, gt, data_range=val_dr)[k])
                                    for k in METRICS})
                # 口径 A：逐样本
                for i in range(pred.shape[0]):
                    p = pred[i].cpu(); g = gt[i].cpu(); b = bic[i].cpu(); m = mean_map[i].cpu()
                    rows_model_ps.append({k: float(evaluate_all(p, g, data_range=val_dr)[k])
                                          for k in METRICS})
                    rows_bic_ps.append({k: float(evaluate_all(b, g, data_range=val_dr)[k])
                                        for k in METRICS})
                    rows_mean_ps.append({k: float(evaluate_all(m, g, data_range=val_dr)[k])
                                         for k in METRICS})

        def gmean(rows, key):
            return float(np.mean([r[key] for r in rows]))

        print(f"\n[info] 组均值对比（口径A=逐样本平均 / 口径B=整批池化平均）")
        print(f"{'预测器':10s} {'A:PSNR':>8s} {'B:PSNR':>8s} {'A:SSIM':>8s} {'A:CC':>8s}")
        for name, rps, rb in (("mean", rows_mean_ps, rows_mean_b),
                              ("bicubic", rows_bic_ps, rows_bic_b),
                              ("model", rows_model_ps, rows_model_b)):
            print(f"{name:10s} {gmean(rps,'psnr'):8.2f} {gmean(rb,'psnr'):8.2f} "
                  f"{gmean(rps,'ssim'):8.4f} {gmean(rps,'cc'):+8.4f}")

        def report(title, r_mb, r_mm, r_bm):
            print(f"\n===== {title} =====")
            print("-- model − bicubic --")
            for k, v in r_mb.items():
                print(f"  {k}:")
                print(fmt(v))
            print("-- model − mean (负且CI不跨0 → 未过地板) --")
            for k, v in r_mm.items():
                print(f"  {k}:")
                print(fmt(v))
            print("-- bicubic − mean --")
            for k, v in r_bm.items():
                print(f"  {k}:")
                print(fmt(v))

        print(f"\n########## 口径 A：逐样本 (n={len(rows_model_ps)}) ##########")
        report("逐样本",
               paired_test(rows_model_ps, rows_bic_ps),
               paired_test(rows_model_ps, rows_mean_ps),
               paired_test(rows_bic_ps, rows_mean_ps))

        print(f"\n########## 口径 B：整批池化 (n_batch={len(rows_model_b)}) ##########")
        report("整批池化",
               paired_test(rows_model_b, rows_bic_b),
               paired_test(rows_model_b, rows_mean_b),
               paired_test(rows_bic_b, rows_mean_b))

        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump({
                    "cfg": args.cfg, "stage": stage, "val_dr": val_dr,
                    "n_sample": len(rows_model_ps), "n_batch": len(rows_model_b),
                    "group_mean_per_sample": {
                        "mean": {k: gmean(rows_mean_ps, k) for k in METRICS},
                        "bicubic": {k: gmean(rows_bic_ps, k) for k in METRICS},
                        "model": {k: gmean(rows_model_ps, k) for k in METRICS},
                    },
                    "group_mean_batchpooled": {
                        "mean": {k: gmean(rows_mean_b, k) for k in METRICS},
                        "bicubic": {k: gmean(rows_bic_b, k) for k in METRICS},
                        "model": {k: gmean(rows_model_b, k) for k in METRICS},
                    },
                    "paired_per_sample": {
                        "model_vs_bicubic": paired_test(rows_model_ps, rows_bic_ps),
                        "model_vs_mean": paired_test(rows_model_ps, rows_mean_ps),
                        "bicubic_vs_mean": paired_test(rows_bic_ps, rows_mean_ps),
                    },
                    "paired_batchpooled": {
                        "model_vs_bicubic": paired_test(rows_model_b, rows_bic_b),
                        "model_vs_mean": paired_test(rows_model_b, rows_mean_b),
                        "bicubic_vs_mean": paired_test(rows_bic_b, rows_mean_b),
                    },
                }, f, ensure_ascii=False, indent=2)
            print(f"\n[saved] {args.save}")
    finally:
        _dll_guard.restore()
        print("[info] vcomp140.dll restored")


if __name__ == "__main__":
    main()
