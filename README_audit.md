# `README_audit.md` — the check you can run without a GPU

`code/tools/quickstart_cpu.py` re-derives the paper's headline numbers from the released
result records **alone**. No imagery, no checkpoints, no `torch`, no GPU. It needs only
`numpy` and the Python standard library; SHA-256 verification uses `hashlib`.

```bash
python code/tools/quickstart_cpu.py            # everything
python code/tools/quickstart_cpu.py --no-sha   # skip hashing, ~1 s
```

Expected output (the exact strings printed by the release this file ships with):

```
[info] 记录目录：results
[PASS] 完整性：202 文件核对通过（含 SHA256）
[PASS] 算术闭合：28 个 paired_*.json 自洽
[PASS] 门值复判：29 个 audit_*.json 的 G1–G5 与记录一致

  axis                   Δ_floor   Δ_model       β          p      n  verdict
  PROBA-V 3×              +1.422    +2.332   62.1%   1.82e-05     60  positive
  SEN2VENµS FGMANAUS      +7.988    +0.130    1.6%       0.19     13  ceiling
  SEN2VENµS SO2           +6.443    +1.863   22.4%   2.31e-21     74  positive
  WorldStrat 6.63×        -7.357    -0.879     —         0.21     60  floor-failure
```

---

## What the three checks actually prove

### 1. Integrity — the records are the ones we measured

Every file listed in `RELEASE_MANIFEST.md` is matched against its recorded size and the
first 16 hex digits of its SHA-256. This catches truncated downloads, editor-mangled JSON,
and any silent substitution of a result file.

Sizes are stored with one decimal, so the check allows half a unit of the last digit
(a `581.4 KiB` entry tolerates ±51 bytes). JSON payloads are compared by SHA-256, not size.

### 2. Arithmetic closure — the reported Δ are not hand-entered

Each `paired_*.json` stores, for the same 60 scenes (or n where noted), the group-mean PSNR
of three predictors — **mean** (the constant spatial-mean image), **bicubic**, **model** —
and the paired per-sample statistics of their differences. Because the mean of paired
differences equals the difference of paired means, the tool recomputes

```
Δ_model = PSNR(model) − PSNR(bicubic)          must equal  paired_per_sample.model_vs_bicubic.mean_diff
Δ_floor = PSNR(bicubic) − PSNR(mean)           must equal  paired_per_sample.bicubic_vs_mean.mean_diff
t       = mean_diff / (sd / sqrt(n))            must equal  paired_per_sample.model_vs_bicubic.t
```

If any of these disagree, either the table in the paper or the file was edited by hand. They
agree to floating-point precision. **This is the check that makes the floor claim auditable
rather than asserted:** Δ_floor is *derived* from three numbers that were each measured on
the same scenes, not typed in from a spreadsheet.

`β = Δ_model / (Δ_floor + Δ_model)` is printed only where Δ_floor > 0. In floor-failure rows
the denominator is negative, so β is left blank deliberately.

### 3. Gate re-evaluation — the PASS/FAIL labels are reproducible

`audit_*.json` stores the four-predictor metrics of each axis. The tool recomputes the five
gates with exactly the thresholds hard-coded in `code/sr_audit.py`:

| Gate | Question | Rule |
|---|---|---|
| **G1** pairing integrity | Is the HR reference actually correlated with the target? | `CC(bicubic, GT) ≥ 0.30` → PASS, else FAIL |
| **G2** structural fidelity | Did the model preserve structure rather than reorganise it? | `CC(model)/CC(bicubic) ≥ 0.9` PASS, `≥ 0.7` WARN, else FAIL |
| **G3** output statistics | Are the outputs plausible, or collapsed/out-of-range? | `0.5 ≤ std_ratio ≤ 1.8` → PASS, else FAIL |
| **G4** periodic artefacts | Is there a checkerboard / resampling comb? | FFT peak ratio `< 2.5` → PASS, else WARN |
| **G5** leakage risk | Could HR have entered the reconstruction? | model PSNR `> 45 dB` (or a leak-prone stage) → FAIL |

Overall verdict is FAIL if any gate FAILs, WARN if any WARNs, else PASS. The recomputed
per-gate verdicts and the overall verdict must match what is stored. **A PASS here does not
mean the method is good** — it means the numbers can be trusted enough to be interpreted.
That distinction is the point of the whole exercise.

### 4. Bonus: every number in the manuscript, in one command

`code/tools/verify_manuscript_numbers.py` is the harness we used while writing. It holds 120
assertions, each stored as *(claim id, how the manuscript phrases it, expected value, source
file, JSON path, tolerance)*. In this repository it resolves sources from `results/` and
prints:

```
PASS=119  MISMATCH=0  NO-SOURCE=0  合计=120
```

```bash
python code/tools/verify_manuscript_numbers.py
```

The 120th entry is a `NOTE`, not a pass. A `MISMATCH` means a sentence in the paper and a
file in this repository disagree; a `NO-SOURCE` would mean a claim's backing record is not
released. Both should be zero, and they are. This is the cheapest way to see whether any
individual sentence you care about is backed by a measurement.

---

## Why this is a fair reading of the paper

The paper's argument needs only three quantities per axis, all of which are in this
repository and all of which the quickstart recomputes:

1. **Δ_floor** — what a zero-information predictor already achieves. Negative means bicubic
   is *worse than a constant*, so the axis has no recoverable information left at that scale.
2. **Δ_model** — the model's gain over bicubic. On a collapsed floor this can be large and
   highly significant while meaning nothing.
3. **β** — the share of the *available* margin the model captures. Only defined when the
   floor is above the constant predictor.

Reading them together is what separates *positive* (real gain), *ceiling* (floor high, model
captures ~0), and *floor-failure* (floor below the constant) — the three regimes in Fig. 1.
The WorldStrat row is the one to look at first: Δ_floor = −7.36 dB, yet Δ_model is +1.83 dB
on the full 1,000-scene held-out panel (pooled, p ≤ 1.4 × 10⁻⁴⁵) — a "significant
improvement" over a baseline that a constant beats outright.

---

## Where to look next

| You want to check | Read |
|---|---|
| The gate definitions in full | `code/sr_audit.py` (§`audit()`) |
| How Δ_floor is computed, including the crop-window caveat | `code/eval_paired_floor.py` (`mean_scores`) |
| The scale-isolation experiment (E1) | `code/e1_scale_isolation_matched.py`, `results/e1_*_5pt.json` |
| The published-method re-implementations | `code/edsr_worldstrat.py`, `code/wdsr_worldstrat.py`, `results/sota_review/*.json` |
| Every number in the manuscript, mapped to a file | `code/tools/verify_manuscript_numbers.py` (§4 above — 119/120 pass in this repository) |
| Why `stratified_floor_ws.json` gives −3.3 dB while the paired files give −7.4 dB | `README.md` §Caveats — different crop windows, same formula, same sign |

## Honest limits of this quickstart

* It verifies **internal consistency and reproducibility of the reported statistics**, not
  that the model is any good. That judgement needs the gates *plus* the regime pair.
* It cannot re-derive a per-sample statistic from scratch, because per-scene PSNR arrays are
  large; the released files store their sufficient statistics (mean, sd, n, t, CI). If you
  want the raw per-scene values, re-run `code/eval_paired_floor.py` on the public datasets.
* `results/paired_worldstrat_pre0914.json` is retained on purpose: it is the record produced
  before a checkpoint was retrained, kept so that anyone comparing against an earlier draft
  can see why the numbers moved by ~0.24 dB. The current record is `paired_worldstrat.json`.
