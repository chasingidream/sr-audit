"""Inference + evaluation: load a checkpoint, run on the dataset, compute
PSNR/SSIM (+ optional LPIPS) and save SR / HR / bicubic-upscaled comparisons.
Usage:
    python infer.py --ckpt experiments/ckpts/ckpt_ep49.pt --out experiments/infer --num 50
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.default import CONFIG
from models.sr_model import build_model
from data.dataset import get_dataset
from metrics import calc_psnr_ssim
from losses import PerceptualLoss


def save_png(tensor, path):
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    if arr.shape[2] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="experiments/infer")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--lpips", action="store_true", help="also report LPIPS (downloads weights once)")
    args = ap.parse_args()

    base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    out = os.path.join(base, args.out)
    os.makedirs(out, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", CONFIG)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = get_dataset(cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    perc = PerceptualLoss("alex") if args.lpips else None
    psnrs, ssims, lps = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num:
                break
            lrs = batch["lrs"]
            hr = batch["hr_ref"]
            pred = model(lrs, hr)
            p, s = calc_psnr_ssim(pred.clamp(0, 1), hr.clamp(0, 1))
            psnrs.append(p.item())
            ssims.append(s.item())
            if perc is not None:
                lps.append(perc(pred.clamp(0, 1), hr.clamp(0, 1)).item())
            save_png(pred[0], os.path.join(out, f"sr_{i:04d}.png"))
            save_png(hr[0], os.path.join(out, f"hr_{i:04d}.png"))
            lr0 = F.interpolate(lrs[0, 0], scale_factor=cfg["scale"],
                                mode="bicubic", align_corners=True)
            save_png(lr0, os.path.join(out, f"bic_{i:04d}.png"))

    mp = sum(psnrs) / len(psnrs)
    ms = sum(ssims) / len(ssims)
    print(f"[infer] n={len(psnrs)} PSNR={mp:.2f} SSIM={ms:.4f}" +
          (f" LPIPS={sum(lps)/len(lps):.4f}" if lps else ""))
    print(f"[infer] samples saved to {out}")


if __name__ == "__main__":
    main()
