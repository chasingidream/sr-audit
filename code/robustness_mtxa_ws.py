"""
稳健性复核：CrossAttnAlign 三档 (xa / noxa / aligned / bicubic) 在 WorldStrat 6.63x 上的
配对结论对离群样本是否稳健。

输入：experiments/eval_mtxa_worldstrat.json（逐样本 psnr/ssim/sam_deg，n=60 四条件同序）
方法：
  - 纯 numpy/scipy，不 import torch（规避 vcomp140 segfault）。
  - 以"退化块 / 泄漏"近似 SR-Audit G-gates（G3 std_ratio、G4 FFT 需原始张量，超出 JSON 口径，标注说明）：
      * G_deg_sam   : 任一条件 SAM_deg > 170  → 光谱彻底失效 / no-data / 云污染
      * G_deg_psnr  : 任一条件 PSNR < 10      → 灾难性失败
      * G_deg_ssim  : 任一条件 SSIM < 0       → 结构破坏
      * G5_leak     : 任一条件 PSNR > 45      → SR-Audit G5 泄漏门（>45dB 作废）
  - 对每个过滤视图重算：xa-noxa / xa-aligned / xa-bicubic 的逐样本配对差，
    给出 均值 / 中位 / 10% 截尾均值 / 配对 t 检验 (95%CI + p + Cohen's d)。
输出：experiments/robustness_mtxa_worldstrat.json + 控制台可读表。
"""
import json
import os
import numpy as np
from scipy import stats

_BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(_BASE, "..", "experiments", "eval_mtxa_worldstrat.json")
OUT = os.path.join(_BASE, "..", "experiments", "robustness_mtxa_worldstrat.json")


def load():
    with open(PATH) as f:
        d = json.load(f)
    ps = d["per_sample"]
    keys = ["mtxa_aligned", "mtxa_mis_noxa", "mtxa_mis_xa", "bicubic"]
    arr = {}
    for k in keys:
        arr[k] = {
            "psnr": np.array([s["psnr"] for s in ps[k]]),
            "ssim": np.array([s["ssim"] for s in ps[k]]),
            "sam_deg": np.array([s["sam_deg"] for s in ps[k]]),
        }
    return arr


def trim_mean(x, p=0.1):
    x = np.asarray(x, float)
    k = int(np.floor(len(x) * p))
    if k == 0:
        return float(np.mean(x))
    xs = np.sort(x)
    return float(np.mean(xs[k : len(xs) - k]))


def paired(diff, a, b):
    """diff = a - b (paired, same order)."""
    n = len(diff)
    if n < 3:
        return None
    mean = float(np.mean(diff))
    t, p = stats.ttest_rel(a, b)
    sd = float(np.std(diff, ddof=1))
    se = sd / np.sqrt(n)
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    cohen = float(mean / sd) if sd > 0 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(diff)),
        "trim10_mean": trim_mean(diff, 0.1),
        "sd": sd,
        "ci95": [float(ci[0]), float(ci[1])],
        "t": float(t),
        "p": float(p),
        "cohen_d": cohen,
    }


def summarize(a, n, x, b):
    """三对照：xa vs noxa / xa vs aligned / xa vs bicubic。"""
    return {
        "xa_minus_noxa": paired(x - n, x, n),
        "xa_minus_aligned": paired(x - a, x, a),
        "xa_minus_bicubic": paired(x - b, x, b),
    }


def main():
    arr = load()
    a, n, x, b = arr["mtxa_aligned"], arr["mtxa_mis_noxa"], arr["mtxa_mis_xa"], arr["bicubic"]
    N = len(x["psnr"])
    assert N == 60

    # 退化 / 泄漏 掩码（任一条件触发即视为该样本异常）
    m_sam = (a["sam_deg"] > 170) | (n["sam_deg"] > 170) | (x["sam_deg"] > 170) | (b["sam_deg"] > 170)
    m_psnr = (a["psnr"] < 10) | (n["psnr"] < 10) | (x["psnr"] < 10) | (b["psnr"] < 10)
    m_ssim = (a["ssim"] < 0) | (n["ssim"] < 0) | (x["ssim"] < 0) | (b["ssim"] < 0)
    m_leak = (a["psnr"] > 45) | (n["psnr"] > 45) | (x["psnr"] > 45) | (b["psnr"] > 45)

    views = {
        "A_full": np.ones(N, bool),
        "B_deg_sam_gt170": ~m_sam,
        "C_deg_psnr_lt10": ~m_psnr,
        "D_deg_ssim_lt0": ~m_ssim,
        "E_g5_leak_gt45": ~m_leak,
        "F_clean_allgates": ~(m_sam | m_psnr | m_ssim | m_leak),
    }

    out = {
        "n_total": N,
        "removed_counts": {
            "deg_sam_gt170": int(m_sam.sum()),
            "deg_psnr_lt10": int(m_psnr.sum()),
            "deg_ssim_lt0": int(m_ssim.sum()),
            "g5_leak_gt45": int(m_leak.sum()),
        },
        "views": {},
    }

    print(f"{'view':22s} {'n':>3s} | {'xa-noxa Δ':>10s} {'p':>7s} | {'xa-align Δ':>10s} {'p':>7s} | {'xa-bic Δ':>10s} {'p':>7s}")
    print("-" * 95)
    for name, mask in views.items():
        idx = np.where(mask)[0]
        sub = lambda v: v[idx]
        r = summarize(sub(a["psnr"]), sub(n["psnr"]), sub(x["psnr"]), sub(b["psnr"]))
        out["views"][name] = r
        def fmt(d):
            if d is None:
                return "  n/a", "  n/a"
            return f"{d['mean']:+.2f}", f"{d['p']:.3f}"
        dn, pn = fmt(r["xa_minus_noxa"])
        da, pa = fmt(r["xa_minus_aligned"])
        db, pb = fmt(r["xa_minus_bicubic"])
        print(f"{name:22s} {len(idx):3d} | {dn:>10s} {pn:>7s} | {da:>10s} {pa:>7s} | {db:>10s} {pb:>7s}")

    # 额外：中位 / 截尾均值 视角（仅 full 与 clean）的 xa-noxa 核心对照
    for name in ["A_full", "F_clean_allgates"]:
        mask = views[name]
        idx = np.where(mask)[0]
        diff = (x["psnr"] - n["psnr"])[idx]
        out["views"][name]["xa_minus_noxa"]["median"] = float(np.median(diff))
        out["views"][name]["xa_minus_noxa"]["trim10_mean"] = trim_mean(diff, 0.1)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
