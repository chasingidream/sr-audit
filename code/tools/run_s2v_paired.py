#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为 SEN2VENµS 新站点生成 Fig.1 统一口径的 `paired_s2v_<key>_s<seed>.json`。

背景：`code/figs_final.py::axis_rows()` 读取的是 `eval_paired_floor.py` 产出的
`paired_*.json`（键 `paired_per_sample.{model_vs_bicubic,bicubic_vs_mean}`，
含 mean_diff / ci95 / p / n），而不是 `audit_axis.py` 的 `audit_axis_*.json`。
因此新站点除审计外还需补这一步，才能与 FGMANAUS / SO2 的点同口径地进 Fig. 1。

幂等：已存在 `paired_s2v_<key>_s<seed>.json` 则跳过；有 ckpt 才运行。
崩溃安全：每步后 best-effort 还原 vcomp140.dll。

用法：python code/tools/run_s2v_paired.py [--stations NARYN ...] [--seeds 42 123 7]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(ROOT, "experiments")
LOG_DIR = os.path.join(EXP, "logs_s2v_paired")

STATIONS = {"NARYN": "naryn", "ATTO": "atto", "SUDOUE-2": "sudoue_2"}
DEFAULT_SEEDS = [42, 123, 7]
_DLL = r"C:\Windows\System32\vcomp140.dll"


def restore_dll_safety() -> None:
    if os.path.exists(_DLL):
        return
    for cand in sorted(glob.glob(_DLL + ".guard*")):
        try:
            os.rename(cand, _DLL)
            print(f"[dll] restored {os.path.basename(cand)} -> vcomp140.dll")
            return
        except Exception:
            continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="*", default=list(STATIONS))
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    py = sys.executable
    done = []
    try:
        for station in args.stations:
            key = STATIONS.get(station, station.lower().replace("-", "_"))
            for seed in args.seeds:
                out = os.path.join(EXP, f"paired_s2v_{key}_s{seed}.json")
                ckpt_dir = os.path.join(EXP, f"ckpts_s2v_{key}_s{seed}", "sisr")
                if os.path.exists(out) and not args.force:
                    print(f"[skip] {station} s{seed} (paired exists)")
                    continue
                if not (os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir)):
                    print(f"[warn] {station} s{seed}: no ckpt at {ckpt_dir}; skip")
                    continue
                cfg = f"code/configs/external_sen2venus_{key}_s{seed}.py"
                log_path = os.path.join(LOG_DIR, f"{key}_s{seed}.log")
                cmd = [py, "code/eval_paired_floor.py", "--cfg", cfg,
                       "--stage", "sisr", "--save", out]
                print(f"[eval] {station} s{seed}")
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n===== eval {station} s{seed} "
                             f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                    lf.flush()
                    rc = subprocess.call(cmd, cwd=ROOT, stdout=lf,
                                         stderr=subprocess.STDOUT)
                print(f"       rc={rc}")
                if rc == 0:
                    done.append(dict(station=station, seed=seed, out=out))
                restore_dll_safety()
    finally:
        restore_dll_safety()

    json.dump(dict(done=done, at=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(EXP, "s2v_paired_progress.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[paired] wrote {len(done)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
