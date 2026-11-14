"""
mtxa 三档的逐样本配对显著性检验。

为什么需要它：
  验证集只有 60 个样本，且 mtxa_mis_xa 的验证 PSNR 在不同 epoch 间摆动达 ~2 dB
  （28.50 / 30.60 / 28.83 / 29.79 / 29.57 / 29.06）。直接用单点均值比较
  （如 mis_xa − mis_noxa = +0.35 dB）无法区分"真实贡献"与"噪声"，因为
  样本难度差异带来的方差远大于组间差异。

  配对检验把同一个验证样本在三档下的指标逐一配对作差，抵消样本难度，
  只保留方法差异 —— 这是小验证集下唯一有统计效力的比较方式。

做法：
  对每个验证样本单独前向（batch_size=1），记录 PSNR/SSIM/SAM/CC，
  然后对 (mis_xa − mis_noxa)、(mis_xa − aligned) 做配对 t 检验，
  给出均值差、标准差、95% 置信区间、t 值与 p 值，以及 Cohen's d 效应量。

判据：p < 0.05 且置信区间不跨 0 → 差异显著。
"""
import argparse
import json
import math
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, os.path.join(BASE, "code", "models"))

import torch  # noqa: E402
from data.dataset import get_dataset  # noqa: E402
from models.sr_model import build_model  # noqa: E402
from eval_metrics import evaluate_all  # noqa: E402

try:
    from scipy import stats as _st
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

STAGES = ["mtxa_aligned", "mtxa_mis_noxa", "mtxa_mis_xa"]
METRICS = ["psnr", "ssim", "sam_deg", "cc"]


def load_val(stage, cfg_override=None, device="cuda", subset=None, val_frac=0.9,
             ckpt_dir=None):
    """按 train.py 相同方式重建验证集，返回 loader(batch=1) 与 val_dr。

    subset/val_frac 用于在**训练未见过**的更大验证集上复核（统计加固）：
    模型只在 subset=600 的前 540 个样本上训练，故把 val 取到更靠后的
    区间（如 subset=3000, val_frac=0.9 -> 样本 2700..2999）即为干净留出集。

    ckpt_dir: 覆盖 ckpt 根目录。默认 None -> experiments/ckpts（pipeline 修复前的
              旧跑，norm=None）；审计**当前** per-scene 管线须传
              "experiments/ckpts_worldstrat"。可传相对 BASE 或绝对路径。
    """
    import glob
    if ckpt_dir:
        root = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(BASE, ckpt_dir)
        d = os.path.join(root, stage)
    else:
        d = os.path.join(BASE, "experiments", "ckpts", stage)
    fs = sorted(glob.glob(os.path.join(d, "ckpt_ep*.pt")),
                key=lambda p: int(p.split("_ep")[-1].split(".")[0]))
    ck = torch.load(fs[-1], map_location="cpu", weights_only=False)
    cfg = dict(ck.get("cfg", {}))
    if cfg_override:
        cfg.update(cfg_override)
    if subset:
        ds = dict(cfg.get("dataset", {}))
        ds["subset"] = int(subset)
        cfg["dataset"] = ds

    _is_mis = (stage.startswith("misaligned")
               or stage.startswith("xs_misaligned")
               or stage.startswith("mtxa_mis"))
    geo_jitter = cfg.get("geo_jitter") if _is_mis else None
    cfg["geo_jitter_all"] = stage.startswith("xs_misaligned")

    train_ds = get_dataset(cfg, project_root=BASE, deterministic=False,
                           geo_jitter=geo_jitter)
    val_ds = get_dataset(cfg, project_root=BASE, deterministic=True,
                         geo_jitter=geo_jitter)
    n_train = max(1, int(val_frac * len(train_ds)))
    val_subset = torch.utils.data.Subset(val_ds, list(range(n_train, len(val_ds))))
    loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False,
                                         num_workers=0)

    mn, mx = float("inf"), float("-inf")
    for vb in loader:
        t = vb["target"].float()
        mn = min(mn, float(t.min()))
        mx = max(mx, float(t.max()))
    val_dr = mx - mn if (mx - mn) > 1e-8 else 1.0
    return ck, cfg, loader, val_dr


def per_sample_scores(stage, device, scale=4, subset=None, val_frac=0.9):
    ck, cfg, loader, val_dr = load_val(stage, device=device, subset=subset,
                                       val_frac=val_frac)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    use_rsa = not stage.endswith("noxa")
    use_ref = False
    rows = []
    with torch.no_grad():
        for batch in loader:
            lrs = batch["lrs"].to(device)
            hr_ref = batch["hr_ref"].to(device)
            gt = batch["target"].to(device)
            pred, _ = model(lrs, hr_ref, return_aux=True,
                            use_rsa=use_rsa, use_ref=use_ref)
            m = evaluate_all(pred.clamp(0, 1), gt, data_range=val_dr,
                             scale=scale, with_lpips=False)
            rows.append({k: float(m[k]) for k in METRICS if m.get(k) is not None})
    print(f"  [{stage}] {len(rows)} samples, val_dr={val_dr:.6f}")
    return rows


def paired_test(a, b, name_a, name_b):
    """对同名指标做配对 t 检验，返回可读结果。"""
    out = {}
    for k in METRICS:
        xs = [r[k] for r in a if k in r]
        ys = [r[k] for r in b if k in r]
        n = min(len(xs), len(ys))
        if n < 2:
            continue
        xs, ys = xs[:n], ys[:n]
        diffs = [x - y for x, y in zip(xs, ys)]
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        sd = math.sqrt(var)
        se = sd / math.sqrt(n)
        if _HAS_SCIPY and sd > 0:
            t_stat, p_val = _st.ttest_rel(xs, ys)
            t_stat, p_val = float(t_stat), float(p_val)
            # 正态近似 95% CI（n=60 够用）
            ci = (mean - 1.96 * se, mean + 1.96 * se)
        else:
            t_stat, p_val, ci = float("nan"), float("nan"), (mean - 1.96 * se,
                                                             mean + 1.96 * se)
        d = mean / sd if sd > 0 else float("nan")
        out[k] = dict(mean_diff=mean, sd=sd, ci95=ci, t=t_stat, p=p_val,
                      cohen_d=d, n=n)
    return out


def fmt(r):
    sig = "显著" if (r["p"] == r["p"] and r["p"] < 0.05) else "不显著"
    star = "" if r["p"] != r["p"] else ("***" if r["p"] < 0.001 else
                                        ("**" if r["p"] < 0.01 else
                                         ("*" if r["p"] < 0.05 else "")))
    return (f"    Δ={r['mean_diff']:+.4f}  sd={r['sd']:.4f}  "
            f"95%CI=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]  "
            f"t={r['t']:+.2f}  p={r['p']:.4f}{star}  d={r['cohen_d']:+.2f}  "
            f"→ {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=STAGES)
    ap.add_argument("--scale", type=float, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save", default=None, help="保存逐样本结果到 json")
    ap.add_argument("--subset", type=int, default=None,
                    help="覆盖 dataset.subset，在更大的留出验证集上复核")
    ap.add_argument("--val_frac", type=float, default=0.9,
                    help="前 val_frac 比例视为训练段，其后为验证段")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")
    print(f"[info] device={device}  scipy={_HAS_SCIPY}")

    scores = {}
    _SUF = ("_rbc_zi", "_rbc", "_ct", "_l1", "_zi")
    import re as _re
    for s in args.stages:
        sc = per_sample_scores(s, device, scale=args.scale,
                               subset=args.subset,
                               val_frac=args.val_frac)
        scores[s] = sc
        b = s
        _ch = True
        while _ch:                       # 迭代剥离变体/多种子后缀 -> 基础名
            _ch = False
            for x in _SUF:
                if b.endswith(x):
                    b = b[:-len(x)]
                    _ch = True
                    break
            if _re.search(r"_s\d+$", b):
                b = _re.sub(r"_s\d+$", "", b)
                _ch = True
        scores.setdefault(b, sc)   # 别名键 -> 同一份分数（配对按基础名匹配）

    print("\n===== 逐样本配对检验 =====")
    pairs = [("mtxa_mis_xa", "mtxa_mis_noxa", "cross-attn 自对齐的贡献 (xa − noxa)"),
             ("mtxa_mis_xa", "mtxa_aligned", "距无失配上界 (xa − aligned)")]
    results = {}
    for a, b, title in pairs:
        if a not in scores or b not in scores:
            continue
        print(f"\n-- {title} --")
        r = paired_test(scores[a], scores[b], a, b)
        results[f"{a}_vs_{b}"] = r
        for k, v in r.items():
            print(f"  {k}:")
            print(fmt(v))

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"per_sample": scores, "paired": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n[saved] {args.save}")


if __name__ == "__main__":
    main()
