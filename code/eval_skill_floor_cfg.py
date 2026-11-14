#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨数据集技能地板评估（支持任意 --cfg）。

与 code/diag_skill_floor.py 的区别：后者硬编码 WorldStrat，本脚本通过 --cfg
驱动，可用于 PROBA-V / SEN2VENµS / WorldStrat 三者的同口径对照。

在同一批验证样本、同一个 data_range 下比较：
  1. mean    ：GT 均值常数图（无技能地板——任何方法必须显著超过它）
  2. bicubic ：LR frame0 双三次上采样（不做 SR 的下界）
  3. model   ：训练得到的模型
  4. gt      ：GT 自身（指标自检，应满分）

核心判据（P1 三轴对照）：
  model − bicubic  = 方法层增益（跨数据集可比）
  bicubic − mean   = 数据层信息增益（见 audit_datasets.py，无需训练）

用法:
  python code/eval_skill_floor_cfg.py --cfg code/configs/external_probav.py \
      --stage mtxa_aligned --epoch 29
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

import _dll_guard  # noqa: E402
_dll_guard.hide()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.dataset import get_dataset  # noqa: E402
from eval_metrics import evaluate_all  # noqa: E402
from models.sr_model import build_model  # noqa: E402


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


def stats(p, g, dr):
    m = evaluate_all(p, g, data_range=dr, scale=4, with_lpips=False)
    out = {k: float(m[k]) for k in ("psnr", "ssim", "cc", "sam_deg")
           if m.get(k) is not None}
    out["pred_std"] = float(p.std().item())
    out["gt_std"] = float(g.std().item())
    return out


def agg(rows):
    keys = [k for k in rows[0] if k != "n"]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--stage", default=None)
    ap.add_argument("--epoch", type=int, default=None, help="默认取最后一个 ckpt")
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.stage:
        cfg["stage"] = args.stage
    stage = cfg["stage"]
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")

    # --- 与 train.py 完全一致的 90/10 切分 ---
    val_ds = get_dataset(cfg, project_root=BASE, deterministic=True)
    n_train = max(1, int(0.9 * len(val_ds)))
    val_ds = torch.utils.data.Subset(val_ds, list(range(n_train, len(val_ds))))
    loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4),
                        shuffle=False, num_workers=0)

    # --- val_dr：与 train.py 同口径（val 集 GT 全局 min/max）---
    mn, mx = float("inf"), float("-inf")
    for vb in loader:
        t = vb["target"].float()
        mn, mx = min(mn, float(t.min())), max(mx, float(t.max()))
    val_dr = (mx - mn) if mx - mn > 1e-8 else 1.0

    # --- model ---
    ckpt_path = pick_ckpt(cfg["save_dir"], stage, args.epoch)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    # strict=True 优先：若架构与 ckpt 不完全一致会立刻暴露（strict=False 会静默
    # 跳过不匹配键 -> 部分权重随机初始化 -> 预测被污染却看不出来）。
    try:
        model.load_state_dict(ck["model"], strict=True)
        print("[info] ckpt loaded: strict=True, all keys matched")
    except Exception as e:
        print(f"[WARN] strict load failed -> {type(e).__name__}: {e}")
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"[WARN] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[WARN]   missing e.g. {missing[:3]}")
        if unexpected:
            print(f"[WARN]   unexpected e.g. {unexpected[:3]}")
    model.eval()
    print(f"[info] cfg={args.cfg}  stage={stage}")
    print(f"[info] ckpt={ckpt_path} (ep{ck.get('epoch')})  device={device}")
    print(f"[info] val_samples={len(val_ds)}  val_dr={val_dr:.6f}")

    # --- stage gating：必须与 train.py 严格一致 ---
    # 否则会启用训练时从未使用的模块 -> 预测被污染。真实事故：SEN2VENµS(sisr)
    # 训练时 use_rsa=False，而旧版此处 `not stage.endswith("noxa")` 对 sisr 算出
    # True，导致评测误开 RSA，训练 val 50.39 dB 掉到评测 35.56 dB（-15 dB）。
    if stage in ("sisr", "baseline"):
        use_rsa_eval = False
    else:
        use_rsa_eval = not stage.endswith("noxa")
    print(f"[info] stage gating: use_rsa={use_rsa_eval}  use_ref=False")

    rows = {"mean": [], "bicubic": [], "model": [], "gt": []}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            lrs = batch["lrs"].to(device)
            hr_ref = batch["hr_ref"].to(device)
            gt = batch["target"].to(device)

            pred, _ = model(lrs, hr_ref, return_aux=True,
                            use_rsa=use_rsa_eval, use_ref=False)
            mean_map = gt.mean(dim=(2, 3), keepdim=True).expand_as(gt)
            bic = F.interpolate(lrs[:, 0], size=gt.shape[-2:], mode="bicubic",
                                align_corners=False)
            rows["mean"].append(stats(mean_map, gt, val_dr))
            rows["bicubic"].append(stats(bic, gt, val_dr))
            rows["model"].append(stats(pred, gt, val_dr))
            rows["gt"].append(stats(gt, gt, val_dr))

    n = len(rows["model"])
    print(f"\n[info] n={n} 验证 batch\n")
    print(f"{'预测器':10s} {'PSNR':>8s} {'SSIM':>8s} {'CC':>8s} "
          f"{'SAM°':>8s} {'std比':>8s}")
    for k in ("mean", "bicubic", "model", "gt"):
        a = agg(rows[k])
        print(f"{k:10s} {a['psnr']:8.2f} {a['ssim']:8.4f} {a['cc']:+8.4f} "
              f"{a['sam_deg']:8.3f} {a['pred_std']/max(a['gt_std'],1e-12):8.3f}")

    m, b, mo = agg(rows["mean"]), agg(rows["bicubic"]), agg(rows["model"])
    print("\n===== 判定 =====")
    print(f"bicubic − mean : PSNR {b['psnr']-m['psnr']:+.2f} dB  "
          f"（数据层信息增益，应≈audit_datasets 结果）")
    print(f"model − bicubic: PSNR {mo['psnr']-b['psnr']:+.2f} dB  "
          f"SSIM {mo['ssim']-b['ssim']:+.4f}  CC {mo['cc']-b['cc']:+.4f}  "
          f"← 方法层增益")
    print(f"model − mean   : PSNR {mo['psnr']-m['psnr']:+.2f} dB")
    print("\n[RESULT] " + ",".join(
        f"{k}={v:.4f}" for k, v in (
            ("bic_minus_mean", b["psnr"] - m["psnr"]),
            ("model_minus_bic", mo["psnr"] - b["psnr"]),
            ("model_minus_mean", mo["psnr"] - m["psnr"]),
            ("model_psnr", mo["psnr"]), ("bic_psnr", b["psnr"]),
            ("mean_psnr", m["psnr"]))))


if __name__ == "__main__":
    main()
