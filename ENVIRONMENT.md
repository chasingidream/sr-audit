# ENVIRONMENT.md — how to get this running, and the one trap that will bite you

Everything here is what we actually ran. Where a number is hardware-specific we say so.

---

## 1. Interpreter and packages

Tested with **Python 3.13.14 on Windows 11**, CUDA build of PyTorch. Python ≥ 3.10 should be
fine; nothing uses 3.13-only syntax.

```
numpy                    2.5.1
scipy                    1.18.0        # only for exact p-values / t CDFs
torch                    2.13.0+cu126
torchvision              0.28.0+cu126  # not imported at runtime; install to satisfy deps
tifffile                 2026.8.16
opencv-python-headless   5.0.0
scikit-image, matplotlib, einops, lpips, pandas, pillow, tensorboard
```

```
pip install -r code/requirements.txt
```

PyTorch: on Windows the default PyPI wheel is CPU-only. Install a CUDA build explicitly, e.g.

```
pip install --no-deps torch-2.13.0+cu126-cp313-cp313-win_amd64.whl
pip install --no-deps torchvision-0.28.0+cu126-cp313-cp313-win_amd64.whl
```

The project does not `import torchvision` at runtime; if it is missing, training and
evaluation still work.

**The CPU quickstart needs none of the above** — only `numpy`:

```
python code/tools/quickstart_cpu.py
```

---

## 2. ⚠ The Windows OpenMP trap: `import torch` segfaults with exit code 139

**Symptom.** On some Windows hosts, a CUDA build of PyTorch crashes on `import torch` (or at
the first CUDA call) with no traceback — exit code 139 / access violation. It looks like a
broken install, a driver problem, or a bad wheel. It is none of those.

**Root cause we isolated.** PyTorch's CUDA libraries (cuBLAS / cuDNN) load the system MSVC
OpenMP runtime `vcomp140.dll` from `C:\Windows\System32`, while PyTorch ships Intel OpenMP
(`libiomp5md.dll`) itself. Two OpenMP runtimes in one process corrupt the heap. The crash is
non-deterministic, which is what makes it so confusing.

**How we proved it.** Temporarily renaming `vcomp140.dll` makes `import torch` succeed
immediately — `torch.cuda.is_available() == True`, CUDA kernels run, the whole stack imports.
Renaming it back restores the crash. The DLL itself is a clean, official Microsoft 14.44
build; it is not corrupt. Setting `KMP_DUPLICATE_LIB_OK=TRUE` does **not** help, and there is
only one copy of `libiomp5md.dll` on the machine, so this is not the classic duplicate-OpenMP
case either.

**Fix (verified).** Rename the DLL out of the way for the duration of the process, then put
it back. PowerShell:

```powershell
$dll = "$env:WINDIR\System32\vcomp140.dll"
try {
    Rename-Item -Path $dll -NewName "vcomp140.dll.hide" -ErrorAction Stop
    python code\train.py --cfg code\configs\worldstrat_ablation.py --stage mtxa_aligned
} finally {
    Rename-Item -Path "$dll.hide" -NewName "vcomp140.dll" -ErrorAction SilentlyContinue
}
```

POSIX shell on Windows (Git Bash):

```bash
python code/_dll_guard.py run -- python code/train.py --cfg code/configs/external_worldstrat.py
```

`code/_dll_guard.py` implements the same hide/restore with an `atexit` handler. **If a run is
killed** (Ctrl-C, OOM, scheduler), `atexit` does not fire and the DLL stays renamed. Check and
restore manually:

```
dir C:\Windows\System32\vcomp140*
```

If you see `vcomp140.dll.hide` and no `vcomp140.dll`, rename it back. This is a host
configuration issue, not a property of this code — but leaving the DLL renamed will break
other software on the machine, so please restore it.

**Do not** work around this by installing a CPU-only PyTorch: the pipeline needs CUDA.

---

## 3. Data: what to download and where to put it

None of the imagery is redistributed here (see `README.md`). Download from the original
sources; all four are public.

| Dataset | Where | Expected layout |
|---|---|---|
| **WorldStrat** | Zenodo record 6810792 | one directory per coverage, e.g. `<root>/Landcover-1109896/{L2A/*.tiff, *_ps.tiff}`; `data/raw/metadata.csv` from the same record |
| **PROBA-V** | Zenodo record 6327426 | `<root>/train/<scene_id>/<scene_id>_red_{lr,hr}.png` (the README of that record describes the naming) |
| **SEN2VENµS** | Zenodo record 6514159 (or the Hugging Face `tacofoundation/sen2venus` mirror) | one directory per site; `code/s2v_extract_direct.py` reads the `.taco` shards directly |
| **L8-S2-NAIP** | Hugging Face `namelesscoder/L8-S2-NAIP` | we use only the Sentinel-2 / Landsat-8 export pairs |

**Set the WorldStrat root once.** The released manifests carry *relative* paths; they are
resolved against `WORLDSTRAT_ROOT` (default `~/worldstrat`):

```bash
export WORLDSTRAT_ROOT=/data/worldstrat            # POSIX
setx WORLDSTRAT_ROOT "%USERPROFILE%\worldstrat"    # Windows，之后重开 shell
```

`code/utils/paths.py` holds the resolver; `code/data/dataset.py` and every script that reads a
manifest go through it. Local manifests with absolute paths keep working unchanged.

The WorldStrat manifests in `data/processed/` are:

| File | Role |
|---|---|
| `worldstrat_manifest.json` | all 3,927 pairs; the first 600 are the training subset used in the paper |
| `worldstrat_manifest_holdout.json` | the 1,000-scene, five-station held-out panel (drawn from pairs 600–3927, never seen in training) |
| `worldstrat_scene_norm{,_holdout}.json` | per-scene (μ, σ) for the normalisation used by every reported number |
| `worldstrat_scene_calib.json` | per-scene radiometric calibration used by the `mtxa_aligned` arm |

---

## 4. Hardware and runtime

Measured on **one RTX 4060 Ti (8 GB)** and a 8-core desktop CPU:

| Job | Time |
|---|---|
| `quickstart_cpu.py` (all checks, with SHA-256) | ~4 s |
| Train one ablation arm, 30 epochs, WorldStrat (T = 5, LR 64, HR 424) | ~30 min |
| Train one SEN2VENµS station arm, 30 epochs | ~6 min |
| Paired evaluation, 60 scenes, 3 predictors | ~2 min |
| Paired evaluation of the 1,000-scene held-out panel, 4 seeds | ~29 min |

Memory: the WorldStrat arm fits in 8 GB at `out_size = 424`, batch 4. Larger crops need a
smaller batch. We ran each stage as its own process — sequentially loading two arms in one
process is what surfaces the OpenMP issue of §2 if it is present.

---

## 5. Determinism and seeds

* Three seeds per axis: **7, 42, 123** (plus the default seed-0 run). Seed variants appear as
  `paired_*_s7.json`, `_s42.json`, `_s123.json`.
* Reported statistics are **per-sample paired** tests (mean of per-scene differences, paired
  *t*, 95 % CI, Cohen's *d*), never batch-pooled. Batch pooling is also reported — precisely
  to show that it flips the sign of the WorldStrat result.
* The constant predictor is the per-scene **spatial mean of the ground truth**, computed on
  the same crop as the model and bicubic terms. Any implementation that uses the global mean,
  or a different crop, will produce a different Δ_floor; see `README.md` §Caveats.

---

## 6. Known assumptions that are baked into the configs

If you re-use the configs on a different sensor pair, these are the knobs that encode our
specific setup — they are not "defaults", they are measured properties of the data:

| Setting | Value | Why |
|---|---|---|
| `hr_per_lr` | 6.63 | WorldStrat S2→SPOT scale ratio |
| `hr_div` | 10000 | WorldStrat reflectance encoding (uint16 → 0–1) |
| `lr_bands` / `hr_bands` | `(3,2,1)` / `(0,1,2)` | B04/B03/B02 for LR, RGB for HR |
| `T` | 5 | LR frames per sample in the multi-frame arm |
| bicubic `align_corners` | `False` | Matches the manuscript's stated convention; using `True` shifts Δ_floor by ~0.2 dB |

`code/configs/external_*.py` are the per-dataset configurations; `worldstrat_ablation*.py`
are the ablation arms.
