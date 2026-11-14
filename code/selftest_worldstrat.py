"""Self-test for the WorldStrat real-data ingestion pipeline.

This does NOT download the 107 GB WorldStrat dataset. It manufactures a small
*WorldStrat-FORMAT* sample (real folder layout + real .tif files, synthetic pixels)
so we can validate, end-to-end, that:
    build_manifest_worldstrat  ->  WorldStratDataset  ->  model forward/backward
works before the real data arrives.

Run:  python selftest_worldstrat.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import build_manifest_worldstrat, WorldStratDataset
from configs.default import CONFIG
from models.sr_model import build_model
from losses import CompositeLoss

import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

SAMPLE_ROOT = "data/raw/worldstrat_sample"
MANIFEST_OUT = "data/processed/worldstrat_sample_manifest.json"
SCALE, LR_SIZE, T = 4, 64, 5
HR_SIZE = LR_SIZE * SCALE  # 256


def _make_tif(path, arr):
    import tifffile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # uint16 like real Sentinel-2/SPOT (0..10000-ish), normalized later by /65535
    tifffile.imwrite(path, arr.astype(np.uint16))


def generate_sample():
    rng = np.random.default_rng(0)
    n_cov = 3
    for c in range(n_cov):
        cov = f"coverage_{c:03d}"
        # HR = SPOT (RGB), larger than hr_size so it gets cropped
        hr = rng.integers(0, 8000, size=(HR_SIZE + 24, HR_SIZE + 24, 3),
                          dtype=np.uint16)
        _make_tif(os.path.join(SAMPLE_ROOT, cov, "high_res", "0.tif"), hr)
        # LR = Sentinel-2 multi-date (3 dates < T=5, tests padding)
        n_dates = 3
        for d in range(n_dates):
            lr = rng.integers(0, 4000, size=(HR_SIZE + 24, HR_SIZE + 24, 3),
                              dtype=np.uint16)
            _make_tif(os.path.join(SAMPLE_ROOT, cov, "low_res", "Sentinel2",
                                   f"2020010{d}_{d}.tif"), lr)
    print(f"[gen] wrote {n_cov} coverages (HR + 3 Sentinel-2 dates each) "
          f"under {SAMPLE_ROOT}")


def main():
    generate_sample()
    manifest = build_manifest_worldstrat(SAMPLE_ROOT)
    assert manifest, "manifest is empty -- layout detection failed"
    print(f"[manifest] {len(manifest)} LR-HR pairs")
    print("  example:", json.dumps(manifest[0], indent=2).replace("\n", "\n  "))

    ds = WorldStratDataset(manifest, scale=SCALE, lr_size=LR_SIZE, T=T)
    item = ds[0]
    lrs, hr = item["lrs"], item["hr_ref"]
    print(f"[dataset] lrs {tuple(lrs.shape)}  hr_ref {tuple(hr.shape)}  "
          f"target {tuple(item['target'].shape)}")
    assert lrs.shape == (T, 3, LR_SIZE, LR_SIZE), lrs.shape
    assert hr.shape == (3, HR_SIZE, HR_SIZE), hr.shape
    assert item["target"].shape == hr.shape

    # model forward + backward on GPU (num_workers=0 safe; see train.py note)
    import torch
    cfg = dict(CONFIG)
    model = build_model(cfg).cuda()
    loss_fn = CompositeLoss(cfg["loss"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    lrs_d = lrs.unsqueeze(0).cuda()
    hr_d = hr.unsqueeze(0).cuda()
    pred, aux = model(lrs_d, hr_d, return_aux=True)
    loss, parts = loss_fn(pred, hr_d, aux=aux, model=model, lrs=lrs_d)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"[train] forward+backward OK  loss={float(loss):.4f}  "
          f"pred {tuple(pred.shape)}")
    print("\n[OK] WorldStrat ingestion pipeline validated end-to-end "
          "(format self-test with synthetic pixels).")


if __name__ == "__main__":
    main()
