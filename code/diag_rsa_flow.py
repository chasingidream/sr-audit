"""RSA 诊断：检验 cost volume 是否在真实施加位移处有峰值。

这是判定 RSA 是否"架构性失效"的决定性测试：
- 若 cost volume 在施加位移 (tx,ty) 处有清晰峰值 -> RSA 本可学对齐，问题在训练/conf门控；
- 若峰值分散/无峰值 -> 帧间内容差异压过几何位移，RAFT cost-volume 不适用于真实多时相数据。

只在冻结 extractor 特征上算（与 RSA 权重无关），所以是架构前提检验。
"""
import importlib.util, sys, random
import numpy as np
import torch
import torch.nn.functional as F

BASE = ""
spec = importlib.util.spec_from_file_location("cfg", f"{BASE}/code/configs/worldstrat_ablation.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cfg = dict(mod.CONFIG)
sys.path.insert(0, f"{BASE}/code")

from data.dataset import get_dataset
from models.sr_model import build_model
from utils.warp import compute_cost_volume

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(cfg).to(device).eval()
extractor = model.extractor

ds = get_dataset(cfg, project_root=BASE, deterministic=True,
                 geo_jitter=cfg.get("geo_jitter"))
# 取验证子集（后 10%）里的样本
n = len(ds)
idxs = list(range(int(0.9 * n), min(n, int(0.9 * n) + 40)))
R = cfg.get("rsa_radius", 4)

def offset_of(idx_k, H, W):
    dy = idx_k // (2 * R + 1) - R
    dx = idx_k % (2 * R + 1) - R
    return dx, dy

print(f"R={R}  samples={len(idxs)}")
hit, total, pred_shifts, gt_shifts = 0, 0, [], []
with torch.no_grad():
    for i, idx in enumerate(idxs):
        item = ds[idx]
        gf = item["gt_flow"]  # [T,2,h,w]
        lrs = item["lrs"].to(device)  # [T,C,h,w]
        # 只测被扰动的帧 (gt_flow != 0)
        for t in range(1, lrs.shape[0]):
            if gf[t].abs().max() < 1e-6:
                continue
            f0 = extractor(lrs[0:1])            # [1,feat,h,w] 干净锚点
            ft = extractor(lrs[t:t+1])          # [1,feat,h,w] 扰动帧
            cost = compute_cost_volume(f0, ft, R)  # [1,(2R+1)^2,h,w]
            k = cost.argmax(dim=1).flatten().cpu().numpy()  # [N]
            offs = np.array([offset_of(int(kk), 0, 0) for kk in k])  # [N,2] (dx,dy)
            gtx = gf[t, 0].mean().item()  # tx (horizontal)
            gty = gf[t, 1].mean().item()  # ty (vertical)
            # 命中：峰值偏移与 GT 位移差 <=1px
            dist = np.sqrt((offs[:, 0] - gtx) ** 2 + (offs[:, 1] - gty) ** 2)
            hit += int((dist <= 1.0).sum())
            total += len(offs)
            pred_shifts.append(offs.mean(axis=0))
            gt_shifts.append((gtx, gty))
            if i < 5:
                print(f"  sample{idx} t{t}: GT=({gtx:.2f},{gty:.2f})px  "
                      f"costvol_peak_mean=({offs[:,0].mean():.2f},{offs[:,1].mean():.2f})px  "
                      f"hit@1px={100*(dist<=1).mean():.1f}%")

pred = np.array(pred_shifts); gt = np.array(gt_shifts)
print(f"\n=== SUMMARY over {len(gt)} jittered frames ===")
print(f"GT shift        mean=({gt[:,0].mean():.2f},{gt[:,1].mean():.2f})px  "
      f"max|shift|={np.sqrt((gt**2).sum(1)).max():.2f}px")
print(f"costvol peak    mean=({pred[:,0].mean():.2f},{pred[:,1].mean():.2f})px")
print(f"hit@1px (峰值落在GT位移±1px内) = {100*hit/total:.1f}%")
print("-> 若 hit@1px 很低: cost volume 无峰值, RSA 架构性失效 (帧间内容差异压过几何位移)")
