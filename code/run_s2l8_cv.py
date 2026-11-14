#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S2<->L8 4-fold CV 驱动：逐 fold 训练 -> held-out 评测 -> 汇总 n=22 配对统计。

背景：S2<->L8 仅 22 对，若训练=评测同集，模型可记忆评测集，Δ_model 不可作泛化声明。
4-fold CV（round-robin 划分，6/6/5/5）给出诚实泛化估计：每 fold 用 16-17 ROI 训练、
held-out 5-6 ROI 评测，汇总 22 个 held-out 预测做配对 t 检验。

用法：
  python code/run_s2l8_cv.py [--epochs 80] [--skip-train]
输出：
  experiments/paired_s2l8_cv.json  （汇总 + 逐 fold + 逐 ROI）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy import stats as _st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
FOLDS = json.load(open(os.path.join(BASE, "experiments", "s2l8_cv_folds.json")))
NFOLD = len(FOLDS)


def sh(cmd):
    print("[run]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
    print(tail, flush=True)
    if r.returncode != 0:
        raise RuntimeError("cmd failed (%d): %s" % (r.returncode, " ".join(cmd)))
    return r


def paired(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    d = xs - ys
    n = len(d)
    mean = float(d.mean()); sd = float(d.std(ddof=1))
    se = sd / n ** 0.5
    t, p = _st.ttest_rel(xs, ys)
    lo, hi = _st.t.interval(0.95, n - 1, loc=mean, scale=se)
    return dict(mean_diff=mean, sd=sd, ci95=[float(lo), float(hi)],
                t=float(t), p=float(p), cohen_d=mean / sd, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--skip-train", action="store_true",
                    help="跳过训练（断点续跑评测）")
    args = ap.parse_args()

    fold_rows = []
    for k in range(NFOLD):
        held = FOLDS[str(k)]
        cfg = "code/configs/external_s2l8_f%d.py" % k
        if not args.skip_train:
            sh([PY, "-u", "code/train.py", "--cfg", cfg, "--stage", "sisr",
                "--epochs", str(args.epochs)])
        out = "experiments/paired_s2l8_f%d.json" % k
        sh([PY, "-u", "code/eval_s2l8.py", "--cfg", cfg, "--epoch", str(args.epochs - 1),
            "--ids", ",".join(str(x) for x in held), "--save", out])
        d = json.load(open(os.path.join(BASE, out)))
        for r in d["per_roi"]:
            r["fold"] = k
        fold_rows.extend(d["per_roi"])
        print("[fold %d] held=%s  Δ_model(bic)=%+.3f  Δfloor=%+.3f  G1=%.3f"
              % (k, held,
                  d["paired_model_vs_bicubic_gray"]["psnr"]["mean_diff"],
                  d["paired_bicubic_vs_mean_gray"]["psnr"]["mean_diff"],
                  d["g1_bicubic_gt"]["mean"]), flush=True)

    # 汇总 n=22（每 ROI 恰好一次，来自其 held-out fold 的模型）
    pm = [r["psnr_model"] for r in fold_rows]
    pb = [r["psnr_bicubic"] for r in fold_rows]
    pn = [r["psnr_mean"] for r in fold_rows]
    cc = [r["cc_bicubic"] for r in fold_rows]
    cc = [c for c in cc if c == c]
    summary = {
        "protocol": "4-fold CV (round-robin, 6/6/5/5), global-norm house recipe, "
                    "4-band matched (L8 B2/B3/B4/B5 <-> S2 B2/B3/B4/B8), "
                    "grayscale on valid overlap, n=22 held-out predictions",
        "n": len(pm),
        "delta_model_psnr": paired(pm, pb),
        "delta_floor_psnr": paired(pb, pn),
        "g1_cc_bicubic_gt": {"mean": float(np.mean(cc)),
                             "frac_pass": float(np.mean(np.array(cc) >= 0.30))},
        "group_mean": {"model": float(np.mean(pm)),
                        "bicubic": float(np.mean(pb)),
                        "mean": float(np.mean(pn))},
        "per_roi": fold_rows,
    }
    out_path = os.path.join(BASE, "experiments", "paired_s2l8_cv.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    dm = summary["delta_model_psnr"]; df_ = summary["delta_floor_psnr"]
    print("\n=== 4-fold CV 汇总（n=%d held-out）===" % summary["n"])
    print("  model=%.2f  bicubic=%.2f  mean=%.2f (灰度 PSNR)"
          % (summary["group_mean"]["model"], summary["group_mean"]["bicubic"],
             summary["group_mean"]["mean"]))
    print("  Δ_model = %+.2f ± %.2f dB  95%%CI[%+.2f,%+.2f]  p=%.2e  d=%+.2f"
          % (dm["mean_diff"], dm["sd"], dm["ci95"][0], dm["ci95"][1], dm["p"], dm["cohen_d"]))
    print("  Δ_floor = %+.2f ± %.2f dB  p=%.2e" % (df_["mean_diff"], df_["sd"], df_["p"]))
    print("  G1 CC(bic,GT)=%.3f  frac_pass=%.3f"
          % (summary["g1_cc_bicubic_gt"]["mean"], summary["g1_cc_bicubic_gt"]["frac_pass"]))
    print("  wrote", out_path)


if __name__ == "__main__":
    main()
