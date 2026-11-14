"""诊断：模型输出 vs bicubic vs mean 的像素级统计（偏置/方差/结构）。

问题背景：sisr 技能地板长期低于 bicubic（CC 0.14 vs 0.30, PSNR -3.8dB），
pred_std/gt_std 恒 0.33 —— 输出被压缩。本脚本逐样本输出三个预测器的
(输出均值 - GT 均值) 与 (输出 std / GT std)，判断是偏置错、压缩还是结构错。

用法：python diag_output_stats.py --stage sisr --ckpt ckpt_ep119.pt
默认取 skill_floor 相同验证集（60 样本全部）。
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paired_mtxa import load_val
from models.sr_model import build_model


def bicubic_upsample(lr, size):
    """LR frame0 (B,3,64,64) -> HR 尺寸 size 的双三次上采样。"""
    return F.interpolate(lr, size=size, mode="bicubic", align_corners=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sisr")
    ap.add_argument("--ckpt", default=None, help="文件名，如 ckpt_ep119.pt；默认最高 epoch")
    ap.add_argument("--max_samples", type=int, default=60)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    ck, cfg, loader, val_dr = load_val(args.stage, device=device)
    # 选 ckpt：默认 fs[-1]；显式指定则重读
    if args.ckpt:
        import glob
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "experiments", "ckpts", args.stage)
        fs = sorted(glob.glob(os.path.join(d, "ckpt_ep*.pt")),
                    key=lambda p: int(p.split("_ep")[-1].split(".")[0]))
        path = [p for p in fs if p.endswith(args.ckpt)]
        assert path, f"ckpt {args.ckpt} not found in {d}"
        ck = torch.load(path[0], map_location=device, weights_only=False)

    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    # 聚合器
    stats = {"model": [], "bicubic": [], "mean": []}
    n = 0
    with torch.no_grad():
        for batch in loader:
            lrs = batch["lrs"].float().to(device)          # [1,T,3,64,64]
            tgt = batch["target"].float().to(device)       # [1,3,424,424]
            hr_ref = batch["hr_ref"].float().to(device)
            if args.stage == "sisr":
                lrs_in = lrs[:, :1]
            else:
                lrs_in = lrs
            pred = model(lrs_in, hr_ref, return_aux=False)
            if hasattr(pred, "shape") is False or not torch.is_tensor(pred):
                pred = pred[0] if isinstance(pred, (tuple, list)) else pred
            pred = pred.float()
            if tuple(pred.shape[-2:]) != tuple(tgt.shape[-2:]):
                pred = F.interpolate(pred, size=tuple(tgt.shape[-2:]),
                                     mode="bilinear", align_corners=True)
            bic = bicubic_upsample(lrs[:, 0], size=tuple(tgt.shape[-2:])).to(device)
            gt_mean = tgt.mean(dim=(1, 2, 3), keepdim=True)
            mn = torch.full_like(tgt, float(gt_mean.squeeze()))

            for name, out in (("model", pred), ("bicubic", bic), ("mean", mn)):
                dmu = (out.mean(dim=(1, 2, 3)) - gt_mean.squeeze()).abs()
                # 每通道 std
                o_std = out.std(dim=(2, 3))          # [B,3]
                g_std = tgt.std(dim=(2, 3))
                ratio = (o_std / (g_std + 1e-8)).mean(dim=1)
                mse = ((out - tgt) ** 2).mean(dim=(1, 2, 3))
                psnr = -10 * torch.log10(mse / val_dr ** 2)
                stats[name].append({
                    "bias": float(dmu.mean()), "std_ratio": float(ratio.mean()),
                    "psnr": float(psnr.mean())})
            n += 1
            if n >= args.max_samples:
                break

    print(f"[info] stage={args.stage} ckpt={args.ckpt or 'max'} n={n} "
          f"val_dr={val_dr:.4f}")
    print(f"{'预测器':<10}{'|bias|':>10}{'std_ratio':>12}{'PSNR':>10}")
    for name in ("mean", "bicubic", "model"):
        s = stats[name]
        bias = sum(x["bias"] for x in s) / len(s)
        sr = sum(x["std_ratio"] for x in s) / len(s)
        ps = sum(x["psnr"] for x in s) / len(s)
        print(f"{name:<10}{bias:>10.4f}{sr:>12.3f}{ps:>10.3f}")
    print("\n[注] bias=输出均值与GT均值绝对差; std_ratio=输出std/GTstd;")
    print("     理想: bias≈0, std_ratio≈1; std_ratio<1 -> 输出被压缩(回归均值)")


if __name__ == "__main__":
    main()
