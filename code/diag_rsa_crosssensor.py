"""跨传感器 cost-volume 探针（验证 option 3 可行性）。

之前 diag_rsa_flow.py 证明：多时相(LR×LR) 的 cost volume 在施加位移处无峰值
（帧间内容差异压过几何位移）-> RSA 架构性失效。

本探针检验反向假设：跨传感器(HR参考×LR) 同景同时相，内容是对应的（同一地物，
仅传感器/分辨率不同）。若对 LR 特征施加已知位移后 cost volume 峰值落在该位移处，
则说明 RSA 在"跨传感器几何失配"场景下机制可行 -> option 3 走通。

实现：取干净 WorldStrat 样本，extractor 提 HR(SPOT) 与 LR(S2 frame0) 特征到同分辨率
64×64，对 LR 特征施加已知 (dx,dy) 位移，算 cost volume 峰值是否 = (dx,dy)。
"""
import importlib.util, sys
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
from utils.warp import compute_cost_volume, flow_warp

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(cfg).to(device)
extractor = model.extractor
R = cfg.get("rsa_radius", 4)

# 关键修正：加载训练过的 extractor（misaligned_rsa 在 GT 流监督下联合训练过），
# 否则随机初始化的 extractor 特征无内容对应结构，cost volume 不成峰是伪象。
import os, glob, re
ckpt_dir = os.path.join(BASE, "experiments/ckpts/misaligned_rsa")
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_ep*.pt")),
              key=lambda p: int(re.search(r"ckpt_ep(\d+)\.pt", p).group(1)))
if ckpts:
    ckpt = torch.load(ckpts[-1], map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[info] loaded extractor from {os.path.basename(ckpts[-1])}")
else:
    print("[warn] no misaligned_rsa ckpt found — using RANDOM extractor (结果无效!)")
model.eval()

ds = get_dataset(cfg, project_root=BASE, deterministic=True)  # 干净，无 geo_jitter
n = len(ds)
idxs = list(range(int(0.9 * n), min(n, int(0.9 * n) + 40)))

def offset_of(k):
    dy = k // (2 * R + 1) - R
    dx = k % (2 * R + 1) - R
    return dx, dy

print(f"R={R}  samples={len(idxs)}  (跨传感器 HR×LR 同景同时相)")
hit, total = 0, 0
shifts_tested = []
with torch.no_grad():
    for i, idx in enumerate(idxs):
        item = ds[idx]
        hr = item["hr_ref"].to(device)     # [C,H,H]  SPOT HR 参考
        lr0 = item["lrs"][0].to(device)    # [C,h,w]  Sentinel-2 LR frame0
        h, w = lr0.shape[-2:]
        f_hr = F.interpolate(extractor(hr.unsqueeze(0)), (h, w))   # HR 特征插值到 LR 分辨率
        f_lr = extractor(lr0.unsqueeze(0))                   # LR 特征 [1,feat,h,w]
        # 对 LR 特征施加随机已知位移 (dx,dy)
        dx = float(np.random.uniform(-(R - 1), (R - 1)))
        dy = float(np.random.uniform(-(R - 1), (R - 1)))
        flow = torch.zeros(1, 2, h, w, device=device)
        flow[0, 0] = dx; flow[0, 1] = dy
        f_lr_shifted = flow_warp(f_lr, flow)
        cost = compute_cost_volume(f_hr, f_lr_shifted, R)    # [1,(2R+1)^2,h,w]
        k = cost.argmax(dim=1).flatten().cpu().numpy()
        offs = np.array([offset_of(int(kk)) for kk in k])    # [N,2] (dx,dy)
        dist = np.sqrt((offs[:, 0] - dx) ** 2 + (offs[:, 1] - dy) ** 2)
        hit += int((dist <= 1.0).sum()); total += len(offs)
        shifts_tested.append((dx, dy))
        if i < 6:
            print(f"  sample{idx}: GT=({dx:.2f},{dy:.2f})px  "
                  f"peak_mean=({offs[:,0].mean():.2f},{offs[:,1].mean():.2f})px  "
                  f"hit@1px={100*(dist<=1).mean():.1f}%")

print(f"\n=== SUMMARY (跨传感器 HR×LR) ===")
print(f"hit@1px (峰值落在GT位移±1px内) = {100*hit/total:.1f}%")
if 100 * hit / total > 50:
    print("-> 跨传感器场景成本量能成峰: option 3 可行 (RSA 在内容对应场景下机制成立)")
else:
    print("-> 跨传感器仍不成峰: 即便同景也有传感器/辐射差异, 需直接走 option 2 (content-robust 机制)")
