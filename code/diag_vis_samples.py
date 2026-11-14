"""可视化诊断：bicubic / model / GT 三联图对比（RGB，0-1 -> 0-255 显示）。

用法：python code/diag_vis_samples.py --stage sisr_l1 --n 3
输出：experiments/vis_<stage>.png（每行一个样本：bicubic | model | GT，标题带 PSNR/CC）
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paired_mtxa import load_val
from models.sr_model import build_model

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sisr")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    ck, cfg, loader, val_dr = load_val(args.stage, device=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    def metr(a, b):
        a = a.float().clamp(0, 1)
        b = b.float()
        mse = ((a - b) ** 2).mean().item()
        psnr = -10 * np.log10(mse / val_dr ** 2 + 1e-12)
        # CC over flattened pixels
        x, y = a.flatten().numpy() if not torch.is_tensor(a) else a.cpu().flatten().numpy(), b.flatten().numpy()
        cc = np.corrcoef(x, y)[0, 1]
        return psnr, cc

    nrows = min(args.n, 4)
    fig, axes = plt.subplots(nrows, 3, figsize=(13.5, 4.5 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= nrows:
                break
            lrs = batch["lrs"].float().to(device)
            tgt = batch["target"].float().to(device)
            if args.stage.startswith("sisr"):
                lrs_in = lrs[:, :1]
            else:
                lrs_in = lrs
            pred = model(lrs_in, batch["hr_ref"].float().to(device), return_aux=False)
            if not torch.is_tensor(pred):
                pred = pred[0]
            if tuple(pred.shape[-2:]) != tuple(tgt.shape[-2:]):
                pred = F.interpolate(pred, size=tuple(tgt.shape[-2:]),
                                     mode="bilinear", align_corners=True)
            bic = F.interpolate(lrs[:, 0].float(), size=tuple(tgt.shape[-2:]),
                                mode="bicubic", align_corners=True).to(device)
            pred_c, bic_c, tgt_c = pred[0].cpu(), bic[0].cpu(), tgt[0].cpu()

            for k, (name, im) in enumerate(
                    (("bicubic", bic_c), ("model", pred_c), ("GT", tgt_c))):
                p, cc = metr(im, tgt_c) if name != "GT" else (np.inf, 1.0)
                title = name if name == "GT" else f"{name}\nPSNR {p:.2f} CC {cc:.2f}"
                axes[i, k].imshow(im.clamp(0, 1).permute(1, 2, 0).numpy())
                axes[i, k].set_title(title, fontsize=10)
                axes[i, k].axis("off")
            # 直方图：三通道像素分布（第 3 行追加）
            if i == nrows - 1:
                pass

    fig.suptitle(f"stage={args.stage}  ckpt={os.path.basename(ck['__meta__']) if '__meta__' in ck else '?'} "
                 f"(val_dr={val_dr:.3f})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(BASE, "experiments", f"vis_{args.stage}.png")
    fig.savefig(out, dpi=110)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
