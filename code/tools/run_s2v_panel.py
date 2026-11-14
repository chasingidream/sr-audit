#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SEN2VENµS 跨站点方法层面板：顺序训练 + 审计（可续跑、崩溃安全）。

对每个 (站点, seed) 依次执行：
  1) python code/train.py --cfg code/configs/external_sen2venus_<key>_s<seed>.py \
         --stage sisr --seed <seed>
  2) python code/audit_axis.py --cfg <同上> --stage sisr \
         --save experiments/audit_axis_s2v_<key>_s<seed>.json

设计要点
  * **可续跑**：若 audits JSON 已存在则跳过该 (站点, seed)，重复调用安全。
  * **崩溃安全**：vcomp140.dll 由子进程 `_dll_guard` 隐藏；若子进程被强杀，
    atexit 不执行 → 本脚本在每步结束后 best-effort 把任何 `vcomp140.dll.guard*`
    改名回 `vcomp140.dll`，并在整体 finally 里再做一次兜底。
  * **单进程顺序执行**（不并行）——避免 3 个 orchestrator 争抢同一个 DLL 改名，
    也避免 GPU 争用；总墙钟 ~ 3–7 h，后台运行。
  * 每步追加写 experiments/logs_s2v_panel/<key>_s<seed>.log 与 progress.json。

用法：
  python code/tools/run_s2v_panel.py                      # 全部站点 × 3 seed
  python code/tools/run_s2v_panel.py --stations NARYN
  python code/tools/run_s2v_panel.py --seeds 42
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
LOG_DIR = os.path.join(EXP, "logs_s2v_panel")

# 站点显示名 -> 配置文件名键
STATIONS = {
    "NARYN": "naryn",
    "ATTO": "atto",
    "SUDOUE-2": "sudoue_2",
}
DEFAULT_SEEDS = [42, 123, 7]

_DLL = r"C:\Windows\System32\vcomp140.dll"


def restore_dll_safety() -> None:
    """best-effort：把残留的 vcomp140.dll.guard* 还原为 vcomp140.dll。"""
    try:
        if os.path.exists(_DLL):
            return
        for cand in sorted(glob.glob(_DLL + ".guard*")):
            try:
                os.rename(cand, _DLL)
                print(f"[dll] restored {os.path.basename(cand)} -> vcomp140.dll")
                return
            except Exception:
                continue
    except Exception:
        pass


def run_step(name: str, cmd: list[str], log_path: str) -> int:
    print(f"[step] {name}")
    print(f"       {' '.join(cmd)}")
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {name}  {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        lf.write(" ".join(cmd) + "\n")
        lf.flush()
        rc = subprocess.call(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"       rc={rc}  {dt:.1f}s")
    restore_dll_safety()
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="*", default=list(STATIONS),
                    help="子集，缺省全部")
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=None,
                    help="覆盖训练轮数（默认用 cfg 的 30）")
    ap.add_argument("--force", action="store_true", help="忽略已有审计、重跑")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    py = sys.executable
    prog_path = os.path.join(EXP, "s2v_panel_progress.json")
    progress = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "items": []}

    jobs = [(s, sd) for s in args.stations for sd in args.seeds]
    print(f"[panel] {len(jobs)} jobs: {jobs}")

    try:
        for station, seed in jobs:
            key = STATIONS.get(station, station.lower().replace("-", "_"))
            cfg = f"code/configs/external_sen2venus_{key}_s{seed}.py"
            audit_out = os.path.join(EXP, f"audit_axis_s2v_{key}_s{seed}.json")
            log_path = os.path.join(LOG_DIR, f"{key}_s{seed}.log")

            if os.path.exists(audit_out) and not args.force:
                print(f"[skip] {station} s{seed} (audit exists)")
                progress["items"].append(dict(station=station, seed=seed,
                                              status="cached"))
            else:
                base = [py, "code/train.py", "--cfg", cfg, "--stage", "sisr",
                        "--seed", str(seed)]
                if args.epochs is not None:
                    base += ["--epochs", str(args.epochs)]
                rc = run_step(f"train {station} s{seed}", base, log_path)
                if rc != 0:
                    print(f"[FAIL] train {station} s{seed} rc={rc}")
                    progress["items"].append(dict(station=station, seed=seed,
                                                  status="train_failed", rc=rc))
                    continue

                rc = run_step(f"audit {station} s{seed}",
                              [py, "code/audit_axis.py", "--cfg", cfg,
                               "--stage", "sisr", "--save", audit_out],
                              log_path)
                if rc != 0:
                    print(f"[FAIL] audit {station} s{seed} rc={rc}")
                    progress["items"].append(dict(station=station, seed=seed,
                                                  status="audit_failed", rc=rc))
                    continue
                progress["items"].append(dict(station=station, seed=seed,
                                              status="done"))

            progress["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            json.dump(progress, open(prog_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
    finally:
        restore_dll_safety()

    print("[panel] all jobs processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
