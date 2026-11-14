"""零训练、纯 CPU 的复现自检（README_audit.md 的 quickstart）。

只读发布出来的**记录文件**，不加载影像、不加载权重、不 import torch：

1. **完整性** — 逐个文件核对 `RELEASE_MANIFEST.md` 里的大小与 SHA256（前 16 位）；
2. **算术闭合** — 对每个 `paired_*.json`，用 `group_mean_per_sample` 重算
   Δ(model−bicubic)、Δ(bicubic−mean)，必须等于记录里的 `mean_diff`；
   再用 `mean/sd/n` 重算 t，必须等于记录里的 `t`；
3. **门值复判** — 对每个 `audit_*.json`，用 `predictors` 与 `sr_audit.py` 的阈值
   重算 G1–G5 的 PASS/FAIL，必须与记录里的 `gates` 逐条一致；
4. **头条数字** — 打印主文报告的 (Δ_floor, Δ_model, β, p, n, verdict) 面板。

跑法：

    python code/tools/quickstart_cpu.py            # 全部检查
    python code/tools/quickstart_cpu.py --no-sha   # 跳过 SHA256（数百文件时更快）

退出码 0 表示全部通过。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 发布仓用 results/，开发仓用 experiments/；两者都能跑
CANDIDATES = ["results", "experiments"]


def find_root_dir() -> str:
    for c in CANDIDATES:
        p = os.path.join(BASE, c)
        if os.path.isdir(p):
            return p
    raise SystemExit("找不到 results/ 或 experiments/ 目录")


def find_manifest() -> str | None:
    for p in (os.path.join(BASE, "RELEASE_MANIFEST.md"),
              os.path.join(BASE, "experiments", "RELEASE_MANIFEST.md")):
        if os.path.exists(p):
            return p
    return None


def sha256(path: str, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: float) -> str:
    for u in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TiB"


# --------------------------------------------------------------------------- #
# 1. 完整性
# --------------------------------------------------------------------------- #
def check_integrity(do_sha: bool) -> tuple[int, int]:
    mf = find_manifest()
    if not mf:
        print("[skip] 未找到 RELEASE_MANIFEST.md")
        return 0, 0
    txt = open(mf, encoding="utf-8").read()
    sec = txt.split("## Tier A")[1].split("## Tier B")[0]
    ok = bad = 0
    for line in sec.splitlines():
        m = re.match(
            r"\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*(B|KiB|MiB|GiB)\s*\|\s*`([0-9a-f]{8,})", line.strip())
        if not m:
            continue
        rel, size, unit, sha_prefix = m.groups()
        path = os.path.join(BASE, rel)
        if not os.path.exists(path):
            path = os.path.join(find_root_dir(), os.path.basename(rel))
            if not os.path.exists(path):
                print(f"  [MISSING] {rel}")
                bad += 1
                continue
        mult = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}[unit]
        want = float(size) * mult
        # 清单里的体积是 human() 的 1 位小数，容差取半个末位
        decimals = len(size.split(".")[1]) if "." in size else 0
        tol = 0.5 * (10 ** -decimals) * mult + 1
        if abs(os.path.getsize(path) - want) > tol:
            print(f"  [SIZE] {rel}: {os.path.getsize(path)} != {want:.0f}")
            bad += 1
            continue
        if do_sha and not sha256(path).startswith(sha_prefix):
            print(f"  [SHA] {rel}")
            bad += 1
            continue
        ok += 1
    print(f"[{'PASS' if bad == 0 else 'FAIL'}] 完整性：{ok} 文件核对通过"
          f"{'（含 SHA256）' if do_sha else '（仅大小，已跳过 SHA）'}"
          f"{f'，{bad} 处不符' if bad else ''}")
    return ok, bad


# --------------------------------------------------------------------------- #
# 2. 算术闭合
# --------------------------------------------------------------------------- #
def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def check_closure(root: str) -> tuple[int, int]:
    ok = bad = 0
    files = sorted(f for f in os.listdir(root) if f.startswith("paired_") and f.endswith(".json"))
    for fn in files:
        d = json.load(open(os.path.join(root, fn), encoding="utf-8"))
        gm = d.get("group_mean_per_sample")
        pp = d.get("paired_per_sample")
        if not gm or not pp:
            continue
        try:
            dm = gm["model"]["psnr"] - gm["bicubic"]["psnr"]
            df = gm["bicubic"]["psnr"] - gm["mean"]["psnr"]
            st_m = pp["model_vs_bicubic"]["psnr"]
            st_f = pp["bicubic_vs_mean"]["psnr"]
        except KeyError:
            continue
        prob = []
        if not _close(dm, st_m["mean_diff"]):
            prob.append(f"Δ_model 重算 {dm:+.4f} != 记录 {st_m['mean_diff']:+.4f}")
        if not _close(df, st_f["mean_diff"]):
            prob.append(f"Δ_floor 重算 {df:+.4f} != 记录 {st_f['mean_diff']:+.4f}")
        n, sd = st_m.get("n"), st_m.get("sd")
        if n and sd and n > 1:
            t = st_m["mean_diff"] / (sd / math.sqrt(n))
            if not _close(t, st_m["t"], 1e-4):
                prob.append(f"t 重算 {t:+.4f} != 记录 {st_m['t']:+.4f}")
        if prob:
            bad += 1
            print(f"  [FAIL] {fn}: " + "; ".join(prob))
        else:
            ok += 1
    print(f"[{'PASS' if bad == 0 else 'FAIL'}] 算术闭合：{ok} 个 paired_*.json 自洽"
          f"{f'，{bad} 处不符' if bad else ''}")
    return ok, bad


# --------------------------------------------------------------------------- #
# 3. 门值复判（阈值与 code/sr_audit.py 一致）
# --------------------------------------------------------------------------- #
def recompute_gates(res: dict) -> dict:
    p = res["predictors"]
    cc_b, cc_m = p["bicubic"]["cc"], p["model"]["cc"]
    r_cc = cc_m / cc_b if cc_b else 0.0
    g = {}
    g["G1_配对完整性"] = "PASS" if cc_b >= 0.30 else "FAIL"
    g["G2_结构保真"] = "PASS" if r_cc >= 0.9 else ("WARN" if r_cc >= 0.7 else "FAIL")
    g["G3_输出统计"] = "PASS" if 0.5 <= res["std_ratio"] <= 1.8 else "FAIL"
    g["G4_周期伪影"] = "PASS" if res["grid_ratio_model"] < 2.5 else "WARN"
    leak = p["model"]["psnr"] > 45.0
    g["G5_泄漏风险"] = "FAIL" if leak else "PASS"
    return g


def check_gates(root: str) -> tuple[int, int]:
    ok = bad = 0
    files = sorted(f for f in os.listdir(root) if f.startswith("audit_") and f.endswith(".json"))
    for fn in files:
        d = json.load(open(os.path.join(root, fn), encoding="utf-8"))
        if "predictors" not in d or "gates" not in d:
            continue
        mine = recompute_gates(d)
        theirs = {k: v["verdict"] for k, v in d["gates"].items()}
        diff = [f"{k}: 复判 {mine[k]} vs 记录 {theirs[k]}"
                for k in theirs if k in mine and mine[k] != theirs[k]]
        overall = ("FAIL" if "FAIL" in mine.values()
                   else "WARN" if "WARN" in mine.values() else "PASS")
        if diff or overall != d.get("overall"):
            bad += 1
            print(f"  [FAIL] {fn}: " + "; ".join(diff) +
                  (f"; 总判定 {overall} vs {d.get('overall')}" if overall != d.get("overall") else ""))
        else:
            ok += 1
    print(f"[{'PASS' if bad == 0 else 'FAIL'}] 门值复判：{ok} 个 audit_*.json 的 G1–G5 与记录一致"
          f"{f'，{bad} 处不符' if bad else ''}")
    return ok, bad


# --------------------------------------------------------------------------- #
# 4. 头条数字面板
# --------------------------------------------------------------------------- #
PANEL = [
    ("PROBA-V 3×",        "paired_probav.json",            "positive"),
    ("SEN2VENµS FGMANAUS", "paired_s2v_s42.json",          "ceiling"),
    ("SEN2VENµS SO2",     "paired_s2v_so2_s42.json",       "positive"),
    ("WorldStrat 6.63×",  "paired_worldstrat.json",        "floor-failure"),
    ("S2↔L8 3×",          "paired_s2l8.json",              "—"),
]


def print_panel(root: str) -> None:
    print("\n  %-20s %9s %9s %7s %10s %6s  %s" %
          ("axis", "Δ_floor", "Δ_model", "β", "p", "n", "verdict"))
    for label, fn, verdict in PANEL:
        p = os.path.join(root, fn)
        if not os.path.exists(p):
            print(f"  {label:<20} {'—':>9} (缺 {fn})")
            continue
        d = json.load(open(p, encoding="utf-8"))
        gm, pp = d.get("group_mean_per_sample"), d.get("paired_per_sample")
        if not gm or not pp:
            continue
        df = gm["bicubic"]["psnr"] - gm["mean"]["psnr"]
        dm = pp["model_vs_bicubic"]["psnr"]["mean_diff"]
        pv = pp["model_vs_bicubic"]["psnr"]["p"]
        n = pp["model_vs_bicubic"]["psnr"]["n"]
        beta = f"{dm / (df + dm) * 100:5.1f}%" if df > 0 and (df + dm) else "   —  "
        print(f"  {label:<20} {df:+9.3f} {dm:+9.3f} {beta:>7} {pv:10.3g} {n:6d}  {verdict}")
    print("\n  β = Δ_model / (Δ_floor + Δ_model)，仅当 Δ_floor > 0 时有定义；"
          "floor-failure 时留空。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sha", action="store_true", help="跳过 SHA256 校验")
    args = ap.parse_args()

    root = find_root_dir()
    print(f"[info] 记录目录：{os.path.relpath(root, BASE)}")
    n_ok = n_bad = 0
    a, b = check_integrity(not args.no_sha)
    n_ok += a; n_bad += b
    a, b = check_closure(root)
    n_ok += a; n_bad += b
    a, b = check_gates(root)
    n_ok += a; n_bad += b
    print_panel(root)

    print(f"\n合计：{n_ok} 项通过" + (f"，{n_bad} 项不符" if n_bad else "，无不符"))
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
