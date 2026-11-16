# SR-Audit — skill-floor audit for satellite super-resolution

**Reference code and result records for the paper** *"Illusory satellite super-resolution gains
trace to a collapsed baseline and a mappable land-surface information budget."*

The claim this repository supports is deliberately narrow and testable: a reported
super-resolution gain is only meaningful relative to a **skill floor** — what a
zero-information constant predictor already achieves on the same target. When bicubic
upsampling itself scores *below* that constant (we call this **floor-failure**), any
positive model−bicubic number is an artefact of the collapsed baseline rather than
recovered information. Every axis in the paper is reported as a pair
(Δ_floor, Δ_model) plus the captured share β = Δ_model / (Δ_floor + Δ_model).

---

## What is in here

| Path | Contents |
|---|---|
| `code/` | Training and paired-evaluation pipeline, the five SR-Audit gates (G1–G5), the Δ_floor estimator, the scale-isolation (E1) and multi-seed (E2) experiments, independent re-implementations of four published methods (EDSR, WDSR, RCAN, SwinIR-lite) plus HighRes-net / RAMS / STANet / SEN2SR adapters, and all figure scripts |
| `results/` | Per-sample evaluation records (`paired_*.json`), gate outputs (`audit_*.json`), audit cards, data-layer confidence intervals, and `figs_final/` (PDF + 600 dpi PNG). File names `fig1`–`fig4` are main Figs. 1–4; **`figS12_atlas_tiers` is main-text Fig. 5** — the name predates its promotion from the supplement. The remaining `figS*` are Supplementary Figs. S1–S12 (incl. S6b) |
| `data/processed/` | WorldStrat manifests (train / held-out split) and per-scene normalisation and calibration parameters |
| `RELEASE_MANIFEST.md` | Full inventory with size and SHA-256 (first 16 hex digits) for every released file |

**Weights** (19 per-axis checkpoints, ~229 MiB) are archived on Zenodo — see
*Data and checkpoints* below. **Imagery is not redistributed**: WorldStrat HR (SPOT 6/7) and
SEN2VENµS HR (VENµS) are CC BY-NC, so we release per-sample numeric records, never pixels.

---

## Quickstart — no GPU, no imagery, no training

```bash
git clone https://github.com/chasingidream/sr-audit.git
cd sr-audit
pip install numpy          # the CPU check needs nothing else
python code/tools/quickstart_cpu.py
```

This re-derives the paper's numbers from the released records alone:

```
[PASS] 完整性：202 文件核对通过（含 SHA256）
[PASS] 算术闭合：28 个 paired_*.json 自洽
[PASS] 门值复判：29 个 audit_*.json 的 G1–G5 与记录一致

  axis                   Δ_floor   Δ_model       β          p      n  verdict
  PROBA-V 3×              +1.422    +2.332   62.1%   1.82e-05     60  positive
  SEN2VENµS FGMANAUS      +7.988    +0.130    1.6%       0.19     13  ceiling
  SEN2VENµS SO2           +6.443    +1.863   22.4%   2.31e-21     74  positive
  WorldStrat 6.63×        -7.357    -0.879     —         0.21     60  floor-failure
```

It checks three things: (i) every released file matches the manifest's size and SHA-256;
(ii) Δ(model−bicubic) and Δ(bicubic−mean) recomputed from the group means equal the reported
`mean_diff`, and the reported *t* equals mean/(sd/√n); (iii) the five gates recomputed from
the stored predictor metrics reproduce the stored PASS/FAIL/WARN verdicts and the overall
verdict. See [`README_audit.md`](README_audit.md) for what each gate means.

---

## Reproducing the full pipeline

Requires a CUDA GPU and the checkpoints from Zenodo. Full details, including the Windows
OpenMP trap that makes `import torch` segfault on some hosts, are in
[`ENVIRONMENT.md`](ENVIRONMENT.md).

```bash
# 1. data: see ENVIRONMENT.md §3 (WorldStrat, PROBA-V, SEN2VENµS)
export WORLDSTRAT_ROOT=/path/to/worldstrat

# 2. train one axis (30 epochs)
python code/train.py --cfg code/configs/worldstrat_ablation.py --stage mtxa_aligned

# 3. paired evaluation against the constant and bicubic predictors
python code/eval_paired_floor.py --cfg code/configs/external_worldstrat.py \
       --stage mtxa_aligned --save results/paired_worldstrat_rerun.json

# 4. the five gates
python code/sr_audit.py --stage mtxa_aligned --report results/audit_rerun.md
```

Training one axis is ~30 min on an 8 GB consumer GPU; the gate pass is a few minutes.

---

## Data and checkpoints

| Dataset | Source | Licence |
|---|---|---|
| WorldStrat (SPOT 6/7 + Sentinel-2) | Zenodo record 6810792 | HR **CC BY-NC** — not redistributed |
| PROBA-V super-resolution | Zenodo record 6327426 | CC BY |
| SEN2VENµS (VENµS + Sentinel-2) | Zenodo record 6514159 | HR **CC BY-NC** — not redistributed |
| L8-S2-NAIP (Sentinel-2 ↔ Landsat-8) | Hugging Face `namelesscoder/L8-S2-NAIP`; we use only its S2/L8 pairs | see source |

Trained per-axis checkpoints (~229 MiB): **Zenodo DOI to be inserted at submission.**
Until then the exact checkpoint inventory is listed in `RELEASE_MANIFEST.md` §Tier B.

---

## Licence and citation

Code and result records: **BSD-3-Clause** (see [`LICENSE`](LICENSE)).
Figures may be reused under the same licence with attribution.

Citation metadata is in [`CITATION.cff`](CITATION.cff). While the paper is under review,
please cite the preprint once it is posted; the archived release is `v1.0.0` of this
repository and matches the numbers in the submitted manuscript.

---

## Caveats worth knowing before you reuse this

* **Δ_floor is a property of the data and the evaluation protocol, not of a model.** Its
  *magnitude* changes with the crop window and the loader; its *sign* does not. We therefore
  report it as a range across protocols and treat the sign as the invariant.
* **β is undefined when Δ_floor ≤ 0.** In floor-failure rows β is left blank on purpose —
  the denominator Δ_floor + Δ_model is negative, so a percentage would be meaningless.
* **Absolute PSNR is not comparable across datasets** here: it depends on each axis'
  `data_range`. Only differences (Δ) are comparable, which is why every table reports Δ.
