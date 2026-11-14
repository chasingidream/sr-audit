"""技能地板检验：我们的模型是否真的优于"预测均值"？

触发原因：n=300 检验里 PSNR +0.60dB 显著，但 CC≈0.02（几乎与 GT 无关）。
若 val_dr(0.345) 远大于单图方差(gt_std≈0.0068，差 50 倍)，PSNR/SSIM 的
C2 常数会淹没结构项 —— 指标可能主要在奖励"预测局部均值"。

本脚本在同一批验证样本上比较四个预测器：
  1. mean   ：逐波段 GT 均值常数图（无技能上界）
  2. bicubic：LR frame0 双三次上采样（不做 SR 的下界）
  3. model  ：mtxa_mis_xa（我们的模型，用 frame0 锚点，无泄漏）
  4. gt_shuffle：GT 自身（仅作指标自检，应得满分）

判定：若 model 的 PSNR 仅比 mean 高 <1dB 且 CC≈0，则此前所有 ~28-29dB
结论都是在噪声上比较，必须先修评测协议（改为逐图归一化 / 高分位截断 /
结构指标）再谈贡献。
"""
import argparse
import math
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, os.path.join(BASE, "code", "configs"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from eval_paired_mtxa import load_val  # noqa: E402
from eval_metrics import evaluate_all  # noqa: E402
from models.sr_model import build_model  # noqa: E402


def stats(p, g, dr):
    """统一口径指标 + 输出/GT 标准差比（模糊诊断）。"""
    m = evaluate_all(p.clamp(0, 1), g, data_range=dr, scale=4,
                     with_lpips=False)
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
    ap.add_argument("--stage", default="mtxa_mis_xa")
    ap.add_argument("--subset", type=int, default=600)
    ap.add_argument("--val_frac", type=float, default=0.9)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")
    ck, cfg, loader, val_dr = load_val(args.stage, device=device,
                                       subset=args.subset,
                                       val_frac=args.val_frac)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"[info] device={device}  val_dr={val_dr:.6f}  stage={args.stage}")

    rows = {"mean": [], "bicubic": [], "model": [], "gt": []}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            lrs = batch["lrs"].to(device)
            hr_ref = batch["hr_ref"].to(device)
            gt = batch["target"].to(device)

            pred, _ = model(lrs, hr_ref, return_aux=True,
                            use_rsa=True, use_ref=False)
            pred = pred.clamp(0, 1)

            # 无技能基线：逐波段 GT 均值常数图
            mean_map = gt.mean(dim=(2, 3), keepdim=True).expand_as(gt)

            # 双三次上采样 LR frame0（无对齐、无多时相）
            lr0 = lrs[:, 0]
            bic = F.interpolate(lr0, size=gt.shape[-2:], mode="bicubic",
                                align_corners=False).clamp(0, 1)

            rows["mean"].append(stats(mean_map, gt, val_dr))
            rows["bicubic"].append(stats(bic, gt, val_dr))
            rows["model"].append(stats(pred, gt, val_dr))
            rows["gt"].append(stats(gt, gt, val_dr))

    n = len(rows["model"])
    print(f"\n[info] n={n} 个验证样本\n")
    print(f"{'预测器':10s} {'PSNR':>8s} {'SSIM':>8s} {'CC':>8s} "
          f"{'SAM°':>8s} {'pred_std/gt_std':>16s}")
    for k in ("mean", "bicubic", "model", "gt"):
        a = agg(rows[k])
        ratio = a["pred_std"] / max(a["gt_std"], 1e-12)
        print(f"{k:10s} {a['psnr']:8.2f} {a['ssim']:8.4f} {a['cc']:+8.4f} "
              f"{a['sam_deg']:8.3f} {ratio:16.3f}")

    m, b, mo = agg(rows["mean"]), agg(rows["bicubic"]), agg(rows["model"])
    print("\n===== 判定 =====")
    print(f"model − mean(无技能) : PSNR {mo['psnr']-m['psnr']:+.2f} dB, "
          f"SSIM {mo['ssim']-m['ssim']:+.4f}, CC {mo['cc']-m['cc']:+.4f}")
    print(f"model − bicubic      : PSNR {mo['psnr']-b['psnr']:+.2f} dB, "
          f"SSIM {mo['ssim']-b['ssim']:+.4f}, CC {mo['cc']-b['cc']:+.4f}")
    print(f"\n单图 GT 标准差 = {m['gt_std']:.5f}   val_dr = {val_dr:.5f}   "
          f"比值 = {val_dr/max(m['gt_std'],1e-12):.1f}×")
    if mo["psnr"] - m["psnr"] < 1.0:
        print("\n[警告] 模型相对无技能基线的 PSNR 增益 < 1dB —— "
              "当前评测协议下指标主要由'预测均值'贡献，结论不可信。")
    else:
        print("\n[OK] 模型显著优于无技能基线，指标反映真实重建能力。")


if __name__ == "__main__":
    main()
