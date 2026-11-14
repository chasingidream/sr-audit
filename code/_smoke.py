"""CPU smoke test: validates model forward/backward without CUDA or vcomp140.
Run: python _smoke.py
"""
import torch
from models.sr_model import build_model
from data.dataset import SyntheticDataset, get_dataset
from losses import CompositeLoss
from metrics import calc_psnr_ssim
import configs.default as C

cfg = dict(C.CONFIG)
cfg["dataset"] = {"type": "synthetic", "kwargs": {"n_samples": 64, "T": cfg["T"]}}
cfg["loss"]["w_perc"] = 0.0  # skip slow LPIPS weight download in smoke test

ds = get_dataset(cfg)
print("dataset len:", len(ds))
sample = ds[0]
print("sample lrs:", tuple(sample["lrs"].shape), "hr_ref:", tuple(sample["hr_ref"].shape))

loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
batch = next(iter(loader))
lrs = batch["lrs"]; hr = batch["hr_ref"]

model = build_model(cfg)
n = sum(p.numel() for p in model.parameters())
print(f"model params: {n/1e3:.1f}K")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = CompositeLoss(cfg["loss"])

model.train()
pred, aux = model(lrs, hr, return_aux=True)
print("pred shape:", tuple(pred.shape), "n_flows:", len(aux["flows"]),
      "flow shape:", tuple(aux["flows"][0].shape))
loss, parts = loss_fn(pred, hr, aux=aux, model=model, lrs=lrs)
loss.backward()
opt.step()
print("loss parts:", {k: round(v, 4) for k, v in parts.items()})

model.eval()
with torch.no_grad():
    p, s = calc_psnr_ssim(pred.clamp(0, 1), hr.clamp(0, 1))
print(f"PSNR={p.item():.2f} SSIM={s.item():.4f}")
print("SMOKE_OK")
