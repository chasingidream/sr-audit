"""Data loaders (dual pipeline).

1. WorldStratDataset  - real cross-sensor pairs (Sentinel-2 LR .tif  <->  SPOT HR .tif).
2. SimulatedDataset  - HR image folder -> multi-temporal LR via random geometric +
                       radiometric degradation + bicubic downscale (quick ablation).
3. SyntheticDataset  - procedurally generated data so the pipeline can be smoke-tested
                       with zero downloads.

All datasets return a dict:
    {
      "lrs"    : [T, C, lr_size, lr_size]   low-res multi-temporal frames (same spatial size)
      "hr_ref" : [C, H, W]                  high-res reference (cross-sensor / guidance)
      "target" : [C, H, W]                  supervised HR (= hr_ref by default)
    }
where H = lr_size * scale (integer scale for PixelShuffle).
"""
import os
import glob
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------- low-level helpers --------------------------- #

def read_tif(path, bands=None, div=None):
    """读取 tif 并做辐射归一化。

    div: 显式缩放分母（DN -> 反射率）。None 时走旧启发式 /65535。
    ⚠ 旧启发式对 SPOT 是错的：WorldStrat 的 HR `_ps.tiff` 是 uint16 DN，
       DN = 反射率×10000（实测 max=8360 -> 反射率 0.836），除以 65535 会把
       HR 压小 6.55 倍，与 Sentinel-2 的反射率(0.02~0.65)差一个量级。
    """
    import tifffile
    arr = tifffile.imread(path)  # numpy, shape (H,W), (H,W,C) or (C,H,W)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[1] // 2:
            pass  # already (C,H,W)
        else:
            arr = np.transpose(arr, (2, 0, 1))  # (H,W,C) -> (C,H,W)
    if bands is not None:
        arr = arr[bands]
    if div is not None:
        arr = arr / float(div)
    elif arr.max() > 1.0:
        arr = arr / 65535.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr.astype(np.float32)


def to_tensor(arr):
    return torch.from_numpy(arr).float()


def random_geo_warp(img, rng=None, max_shift=0.04, max_rot=4.0, return_params=False):
    """Apply a small random affine (rotation + translation) to a [C,H,W] tensor.

    `rng` is a random.Random instance (seeded for deterministic val, unseeded for
    train augmentation). If None, falls back to the global `random` module so the
    synthetic/simulated datasets' legacy call sites keep working.

    If `return_params`, also returns the (tx, ty) translation in *pixel* units
    (the actual spatial shift applied). Used by the route-2 RSA validation, where
    the known GT flow = (tx, ty) constant per frame (set max_rot=0 for pure
    translation so the dense GT flow is exactly constant).
    """
    C, H, W = img.shape
    if rng is None:
        ang = random.uniform(-max_rot, max_rot) * np.pi / 180.0
        tx = random.uniform(-max_shift, max_shift) * W
        ty = random.uniform(-max_shift, max_shift) * H
    else:
        ang = rng.uniform(-max_rot, max_rot) * np.pi / 180.0
        tx = rng.uniform(-max_shift, max_shift) * W
        ty = rng.uniform(-max_shift, max_shift) * H
    cos, sin = np.cos(ang), np.sin(ang)
    theta = torch.tensor([[cos, -sin, tx / (W / 2.0)],
                          [sin, cos, ty / (H / 2.0)]], dtype=torch.float32)
    grid = F.affine_grid(theta.unsqueeze(0), [1, C, H, W], align_corners=True)
    out = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                        align_corners=True, padding_mode="reflection").squeeze(0)
    if return_params:
        return out, (float(tx), float(ty))
    return out


def radiometric_shift(img, max_gain=0.15, max_bias=0.1):
    gain = 1.0 + random.uniform(-max_gain, max_gain)
    bias = random.uniform(-max_bias, max_bias)
    return torch.clamp(img * gain + bias, 0.0, 1.0)


# --------------------------- base --------------------------- #

class BaseSRTriplet(Dataset):
    def __init__(self, scale=4, lr_size=64, T=5, deterministic=False):
        self.scale = scale
        self.lr_size = lr_size
        self.hr_size = lr_size * scale
        self.T = T
        self.deterministic = deterministic

    def _crop_hr(self, hr, idx=None):
        # hr: [C, Hr, Wr] -> crop [C, hr_size, hr_size]
        # 若 deterministic（验证集）：用 random.Random(idx) 固定种子，保证同一 idx
        # 每次取到完全一致的 crop，从而指标可复现；训练集（idx=None）走随机增广。
        C, Hr, Wr = hr.shape
        if Hr < self.hr_size or Wr < self.hr_size:
            hr = F.interpolate(hr.unsqueeze(0), size=(self.hr_size, self.hr_size),
                               mode="bicubic", align_corners=True).squeeze(0)
        else:
            if self.deterministic and idx is not None:
                rng = random.Random(idx)
                i = rng.randint(0, Hr - self.hr_size)
                j = rng.randint(0, Wr - self.hr_size)
            else:
                i = random.randint(0, Hr - self.hr_size)
                j = random.randint(0, Wr - self.hr_size)
            hr = hr[:, i:i + self.hr_size, j:j + self.hr_size]
        return hr

    def _lr_from_hr(self, hr, warp=True, radio=True):
        # downscale clean HR -> lr_size, then (optionally) degrade a *copy* to mimic
        # a different acquisition. Returns lr [C, lr_size, lr_size].
        lr = F.interpolate(hr.unsqueeze(0), size=(self.lr_size, self.lr_size),
                           mode="bicubic", align_corners=True).squeeze(0)
        return lr


# --------------------------- synthetic (no download) --------------------------- #

class SyntheticDataset(BaseSRTriplet):
    def __init__(self, n_samples=2000, scale=4, lr_size=64, T=5, seed=0):
        super().__init__(scale, lr_size, T)
        self.n_samples = n_samples
        rng = np.random.default_rng(seed)
        self.blobs = []
        for _ in range(n_samples):
            base = np.zeros((3, 256, 256), dtype=np.float32)
            for _ in range(random.randint(3, 8)):
                cy, cx = rng.integers(20, 236, 2)
                r = rng.integers(8, 36)
                oy, ox = np.ogrid[:256, :256]
                m = ((oy - cy) ** 2 + (ox - cx) ** 2) <= r * r
                c = rng.random(3)
                base[:, m] += c[:, None]
            base = np.clip(base + 0.05 * rng.random((3, 256, 256)), 0, 1)
            self.blobs.append(torch.from_numpy(base).float())

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        hr = self._crop_hr(self.blobs[idx % len(self.blobs)])
        lrs = []
        for t in range(self.T):
            x = hr
            if t > 0:
                x = random_geo_warp(x, max_shift=0.05, max_rot=7.0)
                x = radiometric_shift(x)
            lr = self._lr_from_hr(x)
            lrs.append(lr)
        lrs = torch.stack(lrs, 0)  # [T,C,lr,lr]
        return {"lrs": lrs, "hr_ref": hr, "target": hr}


# --------------------------- simulated from HR folder --------------------------- #

class SimulatedDataset(BaseSRTriplet):
    def __init__(self, hr_dir, scale=4, lr_size=64, T=5, exts=(".png", ".jpg", ".jpeg", ".tif")):
        super().__init__(scale, lr_size, T)
        self.files = []
        for e in exts:
            self.files += glob.glob(os.path.join(hr_dir, "**", f"*{e}"), recursive=True)
            self.files += glob.glob(os.path.join(hr_dir, "**", f"*{e.upper()}"), recursive=True)
        if not self.files:
            raise FileNotFoundError(f"No images found in {hr_dir}")

    def __len__(self):
        return len(self.files)

    def _load_hr(self, path):
        if path.lower().endswith((".tif", ".tiff")):
            arr = read_tif(path)
        else:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
        return to_tensor(arr)

    def __getitem__(self, idx):
        hr = self._crop_hr(self._load_hr(self.files[idx]))
        lrs = []
        for t in range(self.T):
            x = hr
            if t > 0:
                x = random_geo_warp(x, max_shift=0.05, max_rot=7.0)
                x = radiometric_shift(x)
            lrs.append(self._lr_from_hr(x))
        return {"lrs": torch.stack(lrs, 0), "hr_ref": hr, "target": hr}


# --------------------------- WorldStrat real pairs --------------------------- #

class WorldStratDataset(BaseSRTriplet):
    """Manifest-driven loader for WorldStrat paired scenes.

    manifest: list of dicts {"lr": [paths...], "hr": path, "bands": [..] optional}.
    Each entry yields one sample: T = len(lr) LR frames (Sentinel-2 dates) + 1 HR
    (SPOT). lr frames are resized to lr_size, hr to hr_size (integer scale)."""

    def __init__(self, manifest, scale=4, lr_size=64, T=None,
                 hr_bands=(0, 1, 2), lr_bands=(3, 2, 1), deterministic=False,
                 geo_jitter=None, geo_jitter_all=False,
                 hr_per_lr=6.63, hr_div=10000.0, lr_calib=None,
                 per_scene_calib=False, scene_calib_path=None,
                 norm="none", scene_norm_path=None):
        # HR (_ps.tiff) is RGBA -> take first 3 (RGB).
        # LR (L2A_data.tiff) packs 12 Sentinel-2 bands [B1,B2,B3,B4,B5,B6,B7,
        #   B8,B8A,B9,B11,B12] (no B10); true RGB = B4/B3/B2 = indices [3,2,1].
        super().__init__(scale, lr_size, T or 5, deterministic=deterministic)
        self.manifest = manifest
        self.hr_bands = list(hr_bands)
        self.lr_bands = list(lr_bands)
        self.geo_jitter = geo_jitter
        self.geo_jitter_all = geo_jitter_all
        self.cache = {}
        # --- 几何/辐射修正（2026-09-04）---
        # 真实 GSD：Sentinel-2 10m -> SPOT 1.5m，比例 6.63~6.67（实测 HR/LR 边长比），
        # 旧代码按 scale=4 且 LR 取整幅、HR 取随机窗 -> 二者覆盖不同地面范围，
        # 空间相关性≈0，指标只反映"预测均值"。现按 hr_per_lr 做地理配准裁剪。
        self.hr_per_lr = float(hr_per_lr)
        # HR uint16 DN = 反射率×10000；旧 /65535 会压小 6.55 倍
        self.hr_div = hr_div
        self.hr_size = int(round(lr_size * self.hr_per_lr))
        # 跨传感器辐射处理（三选一，优先级 norm > per_scene_calib > lr_calib）：
        #  norm="per_scene"   : 与 SEN2VENµS/PROBA-V 的 norm="per_scene" 同构——
        #    用该场景 LR 的 (mu,sd) 把 LR 与 HR 都标准化为 unit-variance。
        #    #23 落地诊断：逐场景【仿射】slope≈0.4 把 LR 量程压扁 -> 模型回归均值、
        #    落后双三次 0.83 dB；改成逐场景【归一化】后 LR 变 unit-variance（梯度条件
        #    极好），网络只需学"空间 SR + 平滑增益"，与兄弟数据集击败双三次的机制一致。
        #  per_scene_calib    : 逐场景仿射（诊断已证不如归一化，保留作对照）。
        #  lr_calib           : 全局常量（#23 证为负贡献）。
        self.lr_calib = tuple(lr_calib) if lr_calib is not None else None
        self.per_scene_calib = bool(per_scene_calib)
        self.scene_calib = None
        if self.per_scene_calib and scene_calib_path is not None:
            import json as _json
            with open(scene_calib_path) as _f:
                self.scene_calib = _json.load(_f)
            if len(self.scene_calib) < len(self.manifest):
                print(f"[warn] scene_calib 长度 {len(self.scene_calib)} < "
                      f"manifest {len(self.manifest)}，超出部分回退全局/单位仿射")
        self.norm = norm
        self.scene_norm = None
        if self.norm == "per_scene" and scene_norm_path is not None:
            import json as _json
            with open(scene_norm_path) as _f:
                self.scene_norm = _json.load(_f)   # list[scene] -> [mu, sd]
            if len(self.scene_norm) < len(self.manifest):
                print(f"[warn] scene_norm 长度 {len(self.scene_norm)} < "
                      f"manifest {len(self.manifest)}，超出部分用全局统计")

    def __len__(self):
        return len(self.manifest)

    def _read_cache(self, path, bands, div=None):
        key = (path, div)
        if key not in self.cache:
            self.cache[key] = to_tensor(read_tif(path, bands, div=div))
        return self.cache[key].clone()

    def _pair_box(self, hr_shape, lr_shape, idx):
        """按真实 GSD 比例，给出"覆盖同一片地面"的 LR/HR 裁剪窗口。

        在 LR 坐标系里选窗口 -> 按 ratio(=HR边长/LR边长) 映射到 HR 坐标系，
        保证两者地理范围一致（旧代码 LR 取整幅、HR 取随机窗，地理上无关）。
        """
        ratio = float(hr_shape[-1]) / float(lr_shape[-1])  # HR px per LR px
        hr_size = min(self.hr_size, hr_shape[-1], hr_shape[-2])
        lr_win = int(round(hr_size / ratio))
        lr_win = max(2, min(lr_win, lr_shape[-1], lr_shape[-2]))
        Hl, Wl = lr_shape[-2], lr_shape[-1]
        Hr, Wr = hr_shape[-2], hr_shape[-1]
        i_max, j_max = max(0, Hl - lr_win), max(0, Wl - lr_win)
        if self.deterministic:
            rng = random.Random(idx * 7919 + 13)
            i, j = rng.randint(0, i_max), rng.randint(0, j_max)
        else:
            i, j = random.randint(0, i_max), random.randint(0, j_max)
        hi = min(max(0, int(round(i * ratio))), max(0, Hr - hr_size))
        hj = min(max(0, int(round(j * ratio))), max(0, Wr - hr_size))
        return (i, j, lr_win), (hi, hj, hr_size)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        hr_full = self._read_cache(item["hr"], self.hr_bands, div=self.hr_div)
        lr0 = self._read_cache(item["lr"][0], self.lr_bands)
        lr_box, hr_box = self._pair_box(hr_full.shape, lr0.shape, idx)
        li, lj, lw = lr_box
        hi, hj, hs = hr_box
        hr = hr_full[:, hi:hi + hs, hj:hj + hs]
        if hr.shape[-2] < hs or hr.shape[-1] < hs:  # 兜底：极端小图
            hr = F.interpolate(hr.unsqueeze(0), size=(hs, hs),
                               mode="bicubic", align_corners=True).squeeze(0)
        if self.norm == "per_scene" and self.scene_norm is not None:
            # 与 SEN2VENµS/PROBA-V 的 norm='per_scene' 同构：用该场景 LR 的 (mu,sd)
            # 把 HR 也标准化（无 clamp，允许负/超 1 的标准化值）。
            mu, sd = self.scene_norm[idx]
            hr = (hr - mu) / (sd + 1e-6)

        lr_paths = item["lr"][:self.T]
        lrs, gt_flows = [], []
        for ti, p in enumerate(lr_paths):
            lr = self._read_cache(p, self.lr_bands)[:, li:li + lw, lj:lj + lw]
            if lw != self.lr_size:  # 仅做网格对齐（通常 ≤1px 差）
                lr = F.interpolate(lr.unsqueeze(0),
                                   size=(self.lr_size, self.lr_size),
                                   mode="bicubic",
                                   align_corners=True).squeeze(0)
            if self.norm == "per_scene" and self.scene_norm is not None:
                # 逐场景归一化（#23 落地修复）：LR 标准化为 unit-variance，与 HR 同变换。
                mu, sd = self.scene_norm[idx]
                lr = (lr - mu) / (sd + 1e-6)
            elif self.per_scene_calib and self.scene_calib is not None:
                # 逐场景仿射：每通道 slope_s*LR + intercept_s（诊断证不如归一化）
                sc = self.scene_calib[idx]
                for c in range(lr.shape[0]):
                    s, b = sc[c]
                    lr[c] = (lr[c] * s + b).clamp(0.0, 1.0)
            elif self.lr_calib is not None:  # 全局常量标定 LR -> HR 尺度
                s, b = self.lr_calib
                lr = (lr * s + b).clamp(0.0, 1.0)
            # 人为未对齐变体：对 frame1..T-1 施加随机仿射（frame0 保持干净作对齐锚点）。
            # geo_jitter_all=True 时对全部帧施加（跨传感器设定：HR 作锚点，所有 LR 帧
            # 都相对 HR 失配）。rng 固定到 (样本,帧) -> 同一样本每次 epoch 同扰动，对齐
            # 模块才能学到"稳定失配 -> 逆变换对齐"信号。回传 (tx,ty) 作 GT 流。
            if self.geo_jitter is not None and (ti > 0 or self.geo_jitter_all):
                rng = random.Random(idx * 1000 + ti)
                lr, (tx, ty) = random_geo_warp(lr, rng, return_params=True,
                                               **self.geo_jitter)
                gt_flows.append(torch.tensor([tx, ty], dtype=torch.float32)
                                .view(2, 1, 1).expand(2, self.lr_size, self.lr_size))
            else:
                gt_flows.append(torch.zeros(2, self.lr_size, self.lr_size))
            lrs.append(lr)
        while len(lrs) < self.T:  # pad with last if fewer than T dates
            lrs.append(lrs[-1])
            gt_flows.append(gt_flows[-1])
        return {"lrs": torch.stack(lrs, 0), "hr_ref": hr, "target": hr,
                "gt_flow": torch.stack(gt_flows, 0)}  # [T,2,h,w]


def build_manifest_worldstrat(root,
                              hr_globs=("**/high_res/*.tif", "**/high_res/*.tiff",
                                        "**/SPOT/*.tif", "**/SPOT/*.tiff"),
                              lr_globs=("**/low_res/Sentinel2/*.tif",
                                        "**/low_res/Sentinel2/*.tiff",
                                        "**/Sentinel2/*.tif", "**/Sentinel2/*.tiff",
                                        "**/low_res/*.tif", "**/low_res/*.tiff")):
    """Build a manifest from the standard WorldStrat unpacked layout.

    Standard layout (one folder per coverage / AOI):
        <root>/<coverage_id>/high_res/<hr>.tif             # SPOT HR (RGB, ~1.5 m)
        <root>/<coverage_id>/low_res/Sentinel2/<date>.tif  # Sentinel-2 LR (multi-date)
    Legacy / alternative variants (HR under SPOT/, LR directly under Sentinel2/,
    LR under low_res/) are detected automatically.

    Each manifest entry pairs ALL Sentinel-2 dates (LR) of a coverage with its
    single SPOT HR, yielding T = #dates LR frames for the multi-temporal model.
    """
    hr_files, lr_files = [], []
    for pat in hr_globs:
        hr_files += sorted(glob.glob(os.path.join(root, pat), recursive=True))
    for pat in lr_globs:
        lr_files += sorted(glob.glob(os.path.join(root, pat), recursive=True))

    _KNOWN = {"sentinel2", "sentinel-2", "high_res", "spot", "low_res"}
    def coverage(p):
        d = os.path.dirname(p)
        while os.path.basename(d).lower() in _KNOWN:
            d = os.path.dirname(d)
        return d

    lr_by_cov = {}
    for p in lr_files:
        lr_by_cov.setdefault(coverage(p), []).append(p)
    hr_by_cov = {}
    for p in hr_files:
        hr_by_cov.setdefault(coverage(p), p)  # first HR wins

    manifest = []
    for cov, hr in hr_by_cov.items():
        lrs = sorted(lr_by_cov.get(cov, []))
        if not lrs:
            continue  # need at least one LR frame
        manifest.append({"lr": lrs, "hr": hr})
    for cov in lr_by_cov:
        if cov not in hr_by_cov:
            print(f"[warn] coverage {os.path.basename(cov)} has LR but no HR "
                  f"detected; skipped (adjust hr_globs if your layout differs)")
    if not manifest:
        print(f"[warn] build_manifest_worldstrat: no LR-HR pairs found under {root!r}. "
              f"Found {len(hr_files)} HR and {len(lr_files)} LR candidate files. "
              f"Expected layout: <root>/<cov>/high_res/*.tif + <root>/<cov>/low_res/Sentinel2/*.tif")
    return manifest


def build_manifest_worldstrat_zenodo(root,
                                     hr_ps_glob="**/*_ps.tiff",
                                     hr_rgbn_glob="**/*_rgbn.tiff",
                                     lr_glob="**/L2A/*L2A_data.tiff"):
    """Build a manifest from the OFFICIAL WorldStrat Zenodo packages.

    Zenodo layout (hr_dataset.tar.gz + lr_dataset_l2a.tar.gz):
        <root>/<coverage>/<coverage>_ps.tiff           # SPOT pansharpened HR (RGB)
        <root>/<coverage>/<coverage>_rgbn.tiff         # SPOT RGBN (fallback HR)
        <root>/<coverage>/L2A/<coverage>-<date>-L2A_data.tiff   # Sentinel-2 L2A LR
            (multiple <date> frames per coverage -> multi-temporal)

    HR  = first `*_ps.tiff` of a coverage (fallback `*_rgbn.tiff`).
    LR  = all `*L2A_data.tiff` under that coverage's `L2A/` dir (multi-date).
    Pairing key = the coverage folder name (identical across hr/lr packages).
    """
    hr_ps = sorted(glob.glob(os.path.join(root, hr_ps_glob), recursive=True))
    hr_rgbn = sorted(glob.glob(os.path.join(root, hr_rgbn_glob), recursive=True))
    lr_files = sorted(glob.glob(os.path.join(root, lr_glob), recursive=True))

    def hr_cov(p):  # <cov>/<cov>_ps.tiff  ->  <cov>
        return os.path.basename(os.path.dirname(p))

    def lr_cov(p):  # <cov>/L2A/<cov>-n-L2A_data.tiff  ->  <cov>
        return os.path.basename(os.path.dirname(os.path.dirname(p)))

    # HR: prefer _ps, fall back to _rgbn for coverages lacking _ps.
    hr_by_cov = {}
    for p in hr_ps:
        hr_by_cov.setdefault(hr_cov(p), p)
    for p in hr_rgbn:
        c = hr_cov(p)
        if c not in hr_by_cov:
            hr_by_cov[c] = p

    lr_by_cov = {}
    for p in lr_files:
        lr_by_cov.setdefault(lr_cov(p), []).append(p)

    manifest = []
    for cov, hr in hr_by_cov.items():
        lrs = sorted(lr_by_cov.get(cov, []))
        if not lrs:
            print(f"[warn] coverage {cov} has HR but no Sentinel-2 LR; skipped")
            continue
        manifest.append({"lr": lrs, "hr": hr})
    for cov in lr_by_cov:
        if cov not in hr_by_cov:
            print(f"[warn] coverage {cov} has LR but no HR; skipped")
    if not manifest:
        print(f"[warn] build_manifest_worldstrat_zenodo: 0 pairs under {root!r}. "
              f"Found {len(hr_ps)+len(hr_rgbn)} HR and {len(lr_files)} LR files. "
              f"Expected: <cov>/<cov>_ps.tiff + <cov>/L2A/*L2A_data.tiff")
    else:
        print(f"[prep] {len(manifest)} LR-HR pairs; "
              f"LR dates/coverage: min={min(len(m['lr']) for m in manifest)}, "
              f"max={max(len(m['lr']) for m in manifest)}")
    return manifest


# --------------------------- factory --------------------------- #

# --------------------------- 外部数据集（P1 三轴对照） --------------------------- #

def _external_mod():
    """导入 code/datasets_external.py（只读加载器，与训练管线解耦）。"""
    import sys as _sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/
    if here not in _sys.path:
        _sys.path.insert(0, here)
    import datasets_external
    return datasets_external


class ExternalSRDataset(Dataset):
    """把 datasets_external 的只读加载器包装成训练管线所需的 dict 格式。

    返回 {"lrs":[T,C,h,w], "hr_ref":[C,H,W], "target":[C,H,W], "gt_flow":[T,2,h,w]}。
    与 WorldStratDataset 保持同一约定，train.py 会再做 90/10 的 Subset 切分。
    """

    def __init__(self, T=1, norm="per_scene", lr_size=None, scale=1,
                 deterministic=False):
        self.T = T
        # lr_size：把 LR 裁到统一尺寸。必要性有两点——
        # ① CrossAttnAlign 按帧展平做注意力，序列长 N=H*W，显存 ~N²：
        #    PROBA-V 原生 128×128 → N=16384，单 batch 注意力权重就要 8.6GB（8GB 卡 OOM）；
        #    裁到 64×64 → N=4096，约 0.5GB，与 WorldStrat 同量级。
        # ② 三个数据集统一为 LR 64×64 输入，跨数据集比较才公平。
        self.lr_size = lr_size
        self.scale = scale
        self.deterministic = deterministic
        # norm: "global"(默认) | "per_scene" | "none"
        # 必要性：PROBA-V 反射率动态范围仅 0.03~0.13，直接训练梯度尺度很差。
        # 默认用**数据集级**标准化（只做尺度平移缩放，不拟合 LR→HR 关系）——
        # per_scene 虽更符合"逐场景标定"的结论，但会让 train.py 的 val_dr
        # （全局 min/max 作 PSNR 满量程）被少数异常场景主导（实测 dr=19.7，PSNR 虚高）。
        # 注意：无论哪种，绝对 PSNR 都不跨数据集可比，只能比"模型 − 双三次"的**增益**。
        self.norm = norm
        self.mu, self.sd = None, None

    def _pack(self, lrs, hr, name=""):
        """lrs: (t,C,h,w) numpy -> 补齐/截断到 T 帧；hr: (C,H,W)"""
        t = lrs.shape[0]
        if t >= self.T:
            keep = lrs[: self.T]
        else:                       # 帧数不足 -> 循环补齐
            reps = int(np.ceil(self.T / t))
            keep = np.concatenate([lrs] * reps, 0)[: self.T]
        if self.norm == "per_scene":
            mu = float(keep.mean())
            sd = float(keep.std()) + 1e-6
            keep = (keep - mu) / sd
            hr = (hr - mu) / sd
        elif self.norm == "global":
            self._ensure_global_stats()
            keep = (keep - self.mu) / self.sd
            hr = (hr - self.mu) / self.sd
        return self._to_item(keep, hr, name)

    def _ensure_global_stats(self, n=24):
        """用少量样本的 LR 像素估计数据集级均值/标准差（统计量只取 LR 侧）。"""
        if self.mu is not None:
            return
        vals = []
        for i in range(min(n, len(self.base))):
            s = self.base.load(i)
            v = np.asarray(s["lrs"]).reshape(-1)
            vals.append(v[:: max(1, v.size // 20000)])
        v = np.concatenate(vals)
        self.mu, self.sd = float(v.mean()), float(v.std()) + 1e-6
        print(f"[data] global norm: mu={self.mu:.5f} sd={self.sd:.5f} "
              f"(from {min(n, len(self.base))} scenes)")

    def _to_item(self, keep, hr, name):
        lt = torch.from_numpy(np.ascontiguousarray(keep)).float()
        ht = torch.from_numpy(np.ascontiguousarray(hr)).float()
        _, _, h, w = lt.shape
        flow = torch.zeros(self.T, 2, h, w)
        return {"lrs": lt, "hr_ref": ht, "target": ht, "gt_flow": flow, "name": name}

    def _crop_pair(self, lrs, hr, idx):
        """按 lr_size 裁 LR，并按 scale 取对应的 HR 区域（保持地理同一片区域）。"""
        if not self.lr_size:
            return lrs, hr
        Hl, Wl = lrs.shape[-2], lrs.shape[-1]
        if Hl <= self.lr_size or Wl <= self.lr_size:
            return lrs, hr
        rng = np.random.RandomState(idx * 7919 + 13) if self.deterministic \
            else np.random.RandomState()
        i = rng.randint(0, Hl - self.lr_size + 1)
        j = rng.randint(0, Wl - self.lr_size + 1)
        s = self.scale
        lrs = lrs[..., i: i + self.lr_size, j: j + self.lr_size]
        hr = hr[..., i * s: (i + self.lr_size) * s, j * s: (j + self.lr_size) * s]
        return lrs, hr

    def __len__(self):
        return len(self.base)


class ProbaVSRDataset(ExternalSRDataset):
    """PROBA-V MISR：同传感器 + 多帧 + 3×（128 LR -> 384 HR）。

    三轴设计中的"无跨传感器辐射差"轴：用于隔离多时相融合的真实增益。
    帧选择：按 quality mask 的 clearance 降序取前 T 帧（数据自带 QM）。
    """

    def __init__(self, root=None, band="RED", T=9, scale=3, div=65535.0, split="train",
                 norm="per_scene", lr_size=64, deterministic=False):
        super().__init__(T, norm=norm, lr_size=lr_size, scale=scale,
                         deterministic=deterministic)
        self.scale = scale
        mod = _external_mod()
        if root is None:
            root = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), "data", "external", "probav")
        # 解压后首层可能直接是 train/ test/，也可能是 probav_data/，交给加载器自适应
        self.base = mod.ProbaVDataset(root, band=band, split=split, div=div)
        self.scale = scale

    def __getitem__(self, idx):
        s = self.base.load(idx)
        lrs, qms = s["lrs"], s["qms"]
        clr = qms.reshape(len(lrs), -1).mean(1)
        order = np.argsort(-clr)                 # clearance 降序
        lrs = lrs[order]
        lrs, hr = self._crop_pair(lrs, s["hr"], idx)
        return self._pack(lrs, hr, s["name"])


class Sen2VenusSRDataset(ExternalSRDataset):
    """SEN2VENµS：跨传感器 + 同日单帧 + 2×（128 LR -> 256 HR）。

    三轴设计中的"无时相内容差"轴：用于证明跨传感器本身不是失败主因。
    """

    def __init__(self, root=None, site=None, bands="b2b3b4b8", T=1, div=10000.0,
                 norm="per_scene", lr_size=64, deterministic=False):
        mod = _external_mod()
        if root is None:
            root = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), "data", "external", "sen2venus")
        self.base = mod.Sen2VenusDataset(root, site=site, bands=bands, div=div)
        super().__init__(T, norm=norm, lr_size=lr_size, scale=self.base.scale,
                         deterministic=deterministic)

    def __getitem__(self, idx):
        s = self.base.load(idx)
        lrs, hr = self._crop_pair(s["lrs"], s["hr"], idx)
        return self._pack(lrs, hr, s["name"])


class S2L8SRDataset(ExternalSRDataset):
    """S2↔L8：跨传感器 + 单帧 + 原生 3× (L8 30m 64×64 -> S2 10m 192×192)。

    P1-exp 关键补强：把"唯一正向轴"从同传感器的 PROBA-V 扩到真正的跨传感器轴，
    直接回应审稿 R1（"跨传感器真实增益证据薄"）。LR=L8 RGB，HR=S2 RGB，均 3 通道
    （满足模型 in_ch==out_ch），逐场景标准化（per_scene）消除跨传感器辐射偏置。
    """

    def __init__(self, base, T=1, norm="per_scene", lr_size=64, scale=3,
                 deterministic=False):
        self.base = base
        super().__init__(T, norm=norm, lr_size=lr_size, scale=scale,
                         deterministic=deterministic)

    def __getitem__(self, idx):
        s = self.base.load(idx)
        lrs, hr, i, j = self._crop_pair(s["lrs"], s["hr"], idx)
        item = self._pack(lrs, hr, s["name"])
        lr_v = s.get("lr_valid")
        hr_v = s.get("hr_valid")
        if lr_v is not None:
            sc = self.scale
            lv = lr_v[..., i: i + self.lr_size, j: j + self.lr_size]
            hv = hr_v[..., i * sc: (i + self.lr_size) * sc, j * sc: (j + self.lr_size) * sc]
            item["lr_valid"] = torch.from_numpy(np.ascontiguousarray(lv)).float()
            item["hr_valid"] = torch.from_numpy(np.ascontiguousarray(hv)).float()
        return item

    def _crop_pair(self, lrs, hr, idx):
        """S2↔L8 专用：保证 LR 裁 lr_size×lr_size 时，对应 HR 恰好 lr_size*scale×lr_size*scale。

        L8(30m) 与 S2(10m) 的像素数并非严格 3× 整数倍（常差 1px）。若按原共享逻辑直接用
        (i+lr_size)*scale 作上界，当随机裁切点使其超出 S2 实际尺寸时，Python 切片会把 HR
        静默裁成非 192（如 189），导致 batch 内 HR 尺寸不一致 -> collate 崩溃。这里以 HR
        实际尺寸反向约束 (i,j) 上界，使 HR 输出恒为 out×out，且 LR/HR 仍对应同一地面区域。
        返回 (lrs, hr, i, j)，i/j 供调用方裁切有效掩膜。
        """
        if not self.lr_size:
            return lrs, hr, 0, 0
        s = self.scale
        Hl, Wl = lrs.shape[-2], lrs.shape[-1]
        Hs, Ws = hr.shape[-2], hr.shape[-1]
        if Hl <= self.lr_size or Wl <= self.lr_size:
            return lrs, hr, 0, 0
        out = self.lr_size * s
        max_i = min(Hl - self.lr_size, (Hs - out) // s)
        max_j = min(Wl - self.lr_size, (Ws - out) // s)
        if max_i < 0 or max_j < 0:
            return lrs, hr, 0, 0
        rng = np.random.RandomState(idx * 7919 + 13) if self.deterministic \
            else np.random.RandomState()
        i = rng.randint(0, max_i + 1)
        j = rng.randint(0, max_j + 1)
        lrs = lrs[..., i: i + self.lr_size, j: j + self.lr_size]
        hr = hr[..., i * s: (i + self.lr_size) * s, j * s: (j + self.lr_size) * s]
        return lrs, hr, i, j


def _rebase_worldstrat_manifest(manifest, root):
    """把清单里的**相对路径**按 root 补全（实现见 `code/utils/paths.py`）。

    发布版清单（`data/processed/worldstrat_manifest*.json`）只含相对路径，不写本机
    绝对路径，以免泄漏用户名、也便于他人复现；root 依次取环境变量
    `WORLDSTRAT_ROOT` 与配置里的 `dataset.root`。本地开发用的绝对路径清单不受影响
    （`os.path.isabs` 为真时原样保留）。
    """
    try:
        from utils.paths import rebase_manifest
    except ImportError:                      # 作为包导入时（code.data.dataset）
        from ..utils.paths import rebase_manifest

    return rebase_manifest(manifest, root)


def get_dataset(cfg, project_root=None, deterministic=False, geo_jitter=None):
    d = cfg["dataset"]
    t = d["type"]
    if t == "probav":
        return ProbaVSRDataset(root=d.get("root"), band=d.get("band", "RED"),
                               T=cfg.get("T", 9), scale=cfg.get("scale", 3),
                               div=d.get("div", 65535.0), split=d.get("split", "train"),
                               norm=d.get("norm", "per_scene"),
                               lr_size=cfg.get("lr_size", 64),
                               deterministic=deterministic)
    if t == "sen2venus":
        return Sen2VenusSRDataset(root=d.get("root"), site=d.get("site"),
                                  bands=d.get("bands", "b2b3b4b8"),
                                  T=cfg.get("T", 1), div=d.get("div", 10000.0),
                                  norm=d.get("norm", "per_scene"),
                                  lr_size=cfg.get("lr_size", 64),
                                  deterministic=deterministic)
    if t == "s2l8":
        mod = _external_mod()
        root = d.get("root") or os.path.join(project_root, ".l8s2naip")
        base = mod.S2L8Dataset(root, lr_bands=d.get("lr_bands", (0, 1, 2, 3)),
                               hr_bands=d.get("hr_bands", (0, 1, 2, 3)),
                               div=d.get("div", 1.0))
        ds = S2L8SRDataset(base, T=cfg.get("T", 1), norm=d.get("norm", "global"),
                           lr_size=cfg.get("lr_size", 64), scale=cfg.get("scale", 3),
                           deterministic=deterministic)
        # global 归一化统计量必须先在全量 ROI 上计算（LR 侧像素统计，固定标量、无 GT
        # 泄漏、与 fold 划分无关），再按 use_ids 过滤 —— 否则 fold 评测会用到 held-out
        # 统计量（与训练不一致）。
        if ds.norm == "global":
            ds._ensure_global_stats(n=len(base))
        uids = d.get("use_ids")
        if uids is not None:
            keep = set(int(x) for x in uids)
            base.ids = [i for i in base.ids if i in keep]
        return ds
    if t == "synthetic":
        return SyntheticDataset(**d.get("kwargs", {}))
    if t == "simulated":
        return SimulatedDataset(d["hr_dir"], scale=cfg.get("scale", 4),
                                lr_size=cfg.get("lr_size", 64), T=cfg.get("T", 5))
    if t == "worldstrat":
        if "manifest" in d:
            manifest = d["manifest"]
            if isinstance(manifest, str):
                # 相对路径相对 project_root 解析
                if project_root and not os.path.isabs(manifest):
                    manifest = os.path.join(project_root, manifest)
                with open(manifest) as f:
                    manifest = json.load(f)
                # 发布版清单用相对路径；本地绝对路径清单不受影响（见 _rebase_...）
                manifest = _rebase_worldstrat_manifest(
                    manifest, os.environ.get("WORLDSTRAT_ROOT") or d.get("root"))
        else:
            builder = (build_manifest_worldstrat_zenodo
                       if d.get("layout", "zenodo") == "zenodo"
                       else build_manifest_worldstrat)
            manifest = builder(d["root"])
        if d.get("subset"):
            manifest = manifest[: d["subset"]]
        scene_calib_arg = d.get("scene_calib", None)
        if scene_calib_arg and project_root and not os.path.isabs(scene_calib_arg):
            scene_calib_arg = os.path.join(project_root, scene_calib_arg)
        scene_norm_arg = d.get("scene_norm", None)
        if scene_norm_arg and project_root and not os.path.isabs(scene_norm_arg):
            scene_norm_arg = os.path.join(project_root, scene_norm_arg)
        return WorldStratDataset(manifest, scale=cfg.get("scale", 4),
                                 lr_size=cfg.get("lr_size", 64), T=cfg.get("T", 5),
                                 hr_bands=d.get("hr_bands", (0, 1, 2)),
                                 lr_bands=d.get("lr_bands", (3, 2, 1)),
                                 deterministic=deterministic,
                                 geo_jitter=geo_jitter,
                                 geo_jitter_all=cfg.get("geo_jitter_all", False),
                                 hr_per_lr=d.get("hr_per_lr", 6.63),
                                 hr_div=d.get("hr_div", 10000.0),
                                 lr_calib=d.get("lr_calib", None),
                                 per_scene_calib=d.get("per_scene_calib", False),
                                 scene_calib_path=d.get("scene_calib", None),
                                 norm=d.get("norm", "none"),
                                 scene_norm_path=scene_norm_arg)
    raise ValueError(f"unknown dataset type: {t}")
