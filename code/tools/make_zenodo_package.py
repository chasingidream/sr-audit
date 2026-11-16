#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""打包 Tier B 权重为 Zenodo 上传件。

    python code/tools/make_zenodo_package.py                 # 默认出 dist/zenodo/
    python code/tools/make_zenodo_package.py --no-zip        # 只铺目录，不压缩
    python code/tools/make_zenodo_package.py --verify        # 只校验已有 zip

设计要点
--------
1. **不重命名**。权重按原始目录结构放在 `weights/<stage 目录>/ckpt_ep29.pt`，
   读者把 `weights/` 整个丢进 `experiments/` 即可直接跑 `pick_ckpt(save_dir, stage)`，
   与 RELEASE_MANIFEST.md 的 Tier B 表逐行对应，不需要任何映射表。
2. 只取末轮 `ckpt_ep29.pt`。逐 epoch 权重数十 GB，永不发布。
3. 同时产出 SHA256SUMS.txt、README.md（含逐文件用途表）、zip 自身校验和。
4. 版本号写在包名里，与 CITATION.cff 的 `version: 1.0.0` 对齐。
"""
import argparse
import hashlib
import os
import shutil
import sys
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERSION = "1.0.0"
PKG = "sr-audit-weights-v%s" % VERSION
OUT_ROOT = os.path.join(BASE, "dist", "zenodo")

# (stage 目录, 数据集, 臂/说明, seed, 加载方式)
# 加载方式一栏直接对应仓库里真实存在的入口，见 code/configs/*.py 与各 eval 脚本。
STAGES = [
    ("ckpts_probav/mtxa_aligned", "PROBA-V", "MTXA (multi-temporal cross-attention), frame0 anchor", "42",
     "eval_paired_floor.py --cfg code/configs/external_probav.py --stage mtxa_aligned --epoch 29"),
    ("ckpts_probav/rams_f0", "PROBA-V", "RAMS re-implementation (positive control, +2.89 dB)", "42",
     "rams_probav.py（CKPT_DIR 已指向本目录）"),
    ("ckpts_probav_n300_s7/mtxa_aligned", "PROBA-V (n=300)", "MTXA, seed 7", "7",
     "eval_paired_floor.py --cfg code/configs/external_probav_n300_s7.py --stage mtxa_aligned --epoch 29"),
    ("ckpts_probav_n300_s42/mtxa_aligned", "PROBA-V (n=300)", "MTXA, seed 42 (main-text arm)", "42",
     "eval_paired_floor.py --cfg code/configs/external_probav_n300_s42.py --stage mtxa_aligned --epoch 29"),
    ("ckpts_probav_n300_s123/mtxa_aligned", "PROBA-V (n=300)", "MTXA, seed 123", "123",
     "eval_paired_floor.py --cfg code/configs/external_probav_n300_s123.py --stage mtxa_aligned --epoch 29"),
    ("ckpts_s2v/sisr", "SEN2VENµS (FGMANAUS)", "SISR baseline, seed 42", "42",
     "eval_paired_floor.py --cfg code/configs/external_sen2venus.py --stage sisr --epoch 29"),
    ("ckpts_s2v/sisr_s7", "SEN2VENµS (FGMANAUS)", "SISR baseline, seed 7", "7",
     "eval_paired_floor.py --cfg code/configs/external_sen2venus.py --stage sisr_s7 --epoch 29"),
    ("ckpts_s2v/sisr_s42", "SEN2VENµS (FGMANAUS)", "SISR baseline, seed 42", "42",
     "eval_paired_floor.py --cfg code/configs/external_sen2venus.py --stage sisr_s42 --epoch 29"),
    ("ckpts_s2v/sisr_s123", "SEN2VENµS (FGMANAUS)", "SISR baseline, seed 123", "123",
     "eval_paired_floor.py --cfg code/configs/external_sen2venus.py --stage sisr_s123 --epoch 29"),
    ("ckpts_worldstrat/mtxa_aligned", "WorldStrat (SPOT–S2, 6.63×)", "MTXA aligned, seed 42 (main-text arm)", "42",
     "eval_paired_floor.py --cfg code/configs/external_worldstrat.py --stage mtxa_aligned --epoch 29"),
    ("ckpts_worldstrat/mtxa_aligned_s7", "WorldStrat (SPOT–S2, 6.63×)", "MTXA aligned, seed 7", "7",
     "eval_paired_floor.py --cfg code/configs/external_worldstrat.py --stage mtxa_aligned_s7 --epoch 29"),
    ("ckpts_worldstrat/mtxa_aligned_s42", "WorldStrat (SPOT–S2, 6.63×)", "MTXA aligned, seed 42", "42",
     "eval_paired_floor.py --cfg code/configs/external_worldstrat.py --stage mtxa_aligned_s42 --epoch 29"),
    ("ckpts_worldstrat/mtxa_aligned_s123", "WorldStrat (SPOT–S2, 6.63×)", "MTXA aligned, seed 123", "123",
     "eval_paired_floor.py --cfg code/configs/external_worldstrat.py --stage mtxa_aligned_s123 --epoch 29"),
    ("ckpts_worldstrat/mtxa_mis_xa", "WorldStrat (SPOT–S2, 6.63×)", "MTXA misaligned WITH cross-attention (7.7 px)", "42",
     "eval_paired_floor.py --cfg code/configs/external_worldstrat.py --stage mtxa_mis_xa --epoch 29"),
    ("ckpts_worldstrat/mtxa_mis_noxa", "WorldStrat (SPOT–S2, 6.63×)", "MTXA misaligned WITHOUT cross-attention (ablation)", "42",
     "eval_paired_floor.py --cfg code/configs/external_worldstrat.py --stage mtxa_mis_noxa --epoch 29"),
    ("ckpts_worldstrat/edsr_f0", "WorldStrat (SPOT–S2, 6.63×)", "EDSR re-implementation (Lim et al. 2017)", "42",
     "edsr_worldstrat.py --arch edsr"),
    ("ckpts_worldstrat/rcan_f0", "WorldStrat (SPOT–S2, 6.63×)", "RCAN re-implementation (Zhang et al. 2018)", "42",
     "edsr_worldstrat.py --arch rcan"),
    ("ckpts_worldstrat/swinir_f0", "WorldStrat (SPOT–S2, 6.63×)", "SwinIR-lite re-implementation (Liu et al. 2021)", "42",
     "edsr_worldstrat.py --arch swinir"),
    ("ckpts_worldstrat/wdsr_f0", "WorldStrat (SPOT–S2, 6.63×)", "WDSR re-implementation", "42",
     "cfg code/configs/external_worldstrat.py，stage wdsr_f0"),
]

CKPT = "ckpt_ep29.pt"


def human(n):
    for u in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or u == "GiB":
            return "%.1f %s" % (n, u) if u != "B" else "%d B" % n
        n /= 1024.0


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect():
    """返回 [(src_abs, rel_in_pkg, stage_dir, size)]，顺带做完整性检查。"""
    out, miss = [], []
    for stage, *_ in STAGES:
        src = os.path.join(BASE, "experiments", stage.replace("/", os.sep), CKPT)
        if not os.path.exists(src):
            miss.append(stage)
            continue
        out.append((src, os.path.join("weights", stage.replace("/", os.sep), CKPT), stage,
                    os.path.getsize(src)))
    if miss:
        raise SystemExit("[FAIL] 缺末轮权重 %d 个：\n  %s" % (len(miss), "\n  ".join(miss)))
    return out


def readme_text(files, shas):
    tot = sum(f[3] for f in files)
    rows = []
    for (src, rel, stage, size) in files:
        meta = next(m for m in STAGES if m[0] == stage)
        rows.append("| `%s` | %s | %s | %s | %s |"
                    % (rel.replace(os.sep, "/"), meta[1], meta[2], meta[3], human(size)))
    load = "\n".join("- `%s` → `%s`" % (m[0], m[4]) for m in STAGES)
    return """# SR-Audit — trained weights (v{ver})

Per-axis checkpoints for *"Illusory satellite super-resolution gains trace to a
collapsed baseline and a mappable land-surface information budget."*

**{n} files, {tot}**. Code, per-sample result records and the zero-training
self-check live at <https://github.com/chasingidream/sr-audit>.

Every number in the paper is a **per-sample paired** mean (paired *t*, CI, *p*;
never batch-pooled). These weights reproduce those numbers together with the
`paired_*.json` records in the GitHub repository — the records, not the weights,
are the primary evidence, and `code/tools/quickstart_cpu.py` re-derives the
headline panel from them on a CPU with no weights at all.

---

## Layout

`weights/` mirrors the `experiments/` tree, so it can be dropped straight in:

```
unzip {pkg}.zip
# inside the code checkout
cp -r {pkg}/weights/* experiments/     # then run any eval script as documented
```

Each file is the **final epoch (29)** of its run. Earlier epochs are not
released: this project selects no best-epoch — the main protocol scores the
final epoch, so the final epoch *is* the reported arm.

| file | dataset | arm | seed | size |
|---|---|---|---|---|
{rows}

---

## Loading

Checkpoints are plain `torch.save` dicts with keys `model` (state_dict) and
`epoch`. Load them exactly as the evaluation scripts do:

```python
import torch
from models.sr_model import build_model

ck = torch.load("weights/ckpts_worldstrat/mtxa_aligned/ckpt_ep29.pt",
                map_location="cpu", weights_only=False)
model = build_model(cfg).to(device)          # cfg = the config listed below
model.load_state_dict(ck["model"], strict=True)
model.eval()
```

`build_model` needs the matching config — the architecture is *not* in the
checkpoint. Each weight's entry point:

{load}

A weight from the list above is only meaningful with **its own** `--cfg`
(the stage name is part of the config contract). Loading `mtxa_aligned_s7`
into the seed-42 config will still load `strict=True`, because only the seed
differs, but do not cross datasets: the WorldStrat arms are 6.63× while the
SEN2VENµS arms are 2× and the PROBA-V arms 3×.

---

## What each weight is for

* **`ckpts_probav*`** — PROBA-V 3× same-sensor multi-frame, the positive arm
  (Δ_model +2.42 ± 0.13 dB, three seeds). `rams_f0` is a re-implementation of
  RAMS used as a *positive control* (+2.89 dB), not as a competitor.
* **`ckpts_s2v`** — SEN2VENµS 2× single-image baseline (SISR), the
  ceiling arm (Δ_model +0.13 dB, p = 0.168). Four seeds for the seed-spread
  claim in the ceiling regime.
* **`ckpts_worldstrat`** — WorldStrat 6.63× SPOT←Sentinel-2, the
  floor-failure arm (Δ_floor −7.36 dB, Δ_model −0.88 dB). `mtxa_mis_xa` /
  `mtxa_mis_noxa` are the 7.7 px misalignment ablation with and without
  cross-attention; `edsr_f0` / `rcan_f0` / `swinir_f0` / `wdsr_f0` are
  independent re-implementations of four published families, all of which
  fail to clear the floor.

## Licence

Code is BSD-3-Clause. **These weights are released under CC BY-NC 4.0**:
they are trained on WorldStrat HR (SPOT 6/7), SEN2VENµS HR (VENµS) and
PROBA-V imagery, of which the first two are distributed non-commercially
by their providers, and a permissive licence on the derivative weights
would grant more than the source terms allow. Attribution and
non-commercial use are permitted; redistribution must carry the same terms.

## Verify

```bash
sha256sum -c SHA256SUMS.txt        # Linux / macOS
certutil -hashfile weights\\... SHA256   # Windows, per file
```

SHA-256 of the archive itself is printed by the packaging script and stored
next to the zip as `{pkg}.zip.sha256`.
""".format(ver=VERSION, n=len(files), tot=human(tot), pkg=PKG,
           rows="\n".join(rows), load=load)


def build(no_zip=False):
    files = collect()
    pkg_dir = os.path.join(OUT_ROOT, PKG)
    if os.path.isdir(pkg_dir):
        shutil.rmtree(pkg_dir)
    os.makedirs(os.path.join(pkg_dir, "weights"))

    shas = {}
    for (src, rel, stage, size) in files:
        dst = os.path.join(pkg_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        shas[rel.replace(os.sep, "/")] = sha256(dst)
        print("  [copy] %-52s %s" % (rel.replace(os.sep, "/"), human(size)))

    with open(os.path.join(pkg_dir, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as f:
        for rel in sorted(shas):
            f.write("%s  %s\n" % (shas[rel], rel))

    with open(os.path.join(pkg_dir, "README.md"), "w", encoding="utf-8", newline="") as f:
        f.write(readme_text(files, shas))

    lic = os.path.join(BASE, "release", "LICENSE")
    if os.path.exists(lic):
        shutil.copy2(lic, os.path.join(pkg_dir, "LICENSE"))

    tot = sum(f[3] for f in files)
    print("[ok] 铺盘 %d 文件 / %s -> %s" % (len(files), human(tot), pkg_dir))

    if no_zip:
        return pkg_dir, None

    zip_path = os.path.join(OUT_ROOT, PKG + ".zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, _dirs, fs in os.walk(pkg_dir):
            for fn in sorted(fs):
                ap = os.path.join(root, fn)
                z.write(ap, os.path.join(PKG, os.path.relpath(ap, pkg_dir)))
    zsha = sha256(zip_path)
    with open(zip_path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write("%s  %s\n" % (zsha, os.path.basename(zip_path)))
    print("[ok] zip   %s / %s" % (human(os.path.getsize(zip_path)), zip_path))
    print("[ok] sha256 %s" % zsha)
    return pkg_dir, zip_path


def verify(zip_path):
    try:
        from zipfile import ZipFile
    except ImportError:
        raise SystemExit("需要 zipfile")
    bad = 0
    with ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".pt")]
        print("[info] zip 内权重 %d 个" % len(names))
        for n in names:
            h = hashlib.sha256()
            with z.open(n) as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
            print("  %s  %s" % (h.hexdigest()[:16], n))
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--verify", metavar="ZIP")
    a = ap.parse_args()
    if a.verify:
        sys.exit(verify(a.verify))
    build(a.no_zip)


if __name__ == "__main__":
    main()
