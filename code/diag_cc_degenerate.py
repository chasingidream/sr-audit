"""诊断 CC 指标在 WorldStrat 上是否退化，以及 PSNR 排名是否稳健。

背景：n=300 配对检验里 cross-attn 的 CC 出现"显著变差"(Δ=-0.026, p<1e-4)，
但 CC 逐样本值普遍接近 0，而 PSNR 却有 29dB —— 二者不自洽。
怀疑根因：WorldStrat HR 单块影像方差极小（相对全局 val_dr≈0.345），
Pearson 相关在近常量 patch 上是噪声，CC 在本数据上退化。

本脚本验证两件事：
  1. 记录每个样本的 GT 标准差，看 CC 是否随 GT 方差崩塌；
  2. 按 GT 方差分层，看 CC/PSNR 的配对差异在"高方差样本"上是否还存在；
  3. 同时算逐图动态范围的 PSNR_local，检验排名是否依赖全局 dr 的选择。

用法：
  python code/diag_cc_degenerate.py --subset 3000 --save experiments/cc_diag.json
"""
import argparse
import json
import math
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, os.path.join(BASE, "code", "configs"))

import torch  # noqa: E402
from eval_paired_mtxa import load_val, per_sample_scores  # noqa: E402
from models.sr_model import build_model  # noqa: E402

GT_STD_KEY = "gt_std"


def collect(stage, device, subset=None, val_frac=0.9):
    """返回逐样本记录：cc / psnr_global / psnr_local / gt_std。"""
    ck, cfg, loader, val_dr = load_val(stage, device=device, subset=subset,
                                       val_frac=val_frac)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    use_rsa = not stage.endswith("noxa")
    rows = []
    with torch.no_grad():
        for batch in loader:
            lrs = batch["lrs"].to(device)
            hr_ref = batch["hr_ref"].to(device)
            gt = batch["target"].to(device)
            pred, _ = model(lrs, hr_ref, return_aux=True,
                            use_rsa=use_rsa, use_ref=False)

            p = pred.clamp(0, 1).float()
            g = gt.float()

            # 全局 dr 的 PSNR（benchmark 口径）
            mse_g = ((p - g) ** 2).mean().item()
            psnr_g = 10 * math.log10((val_dr ** 2) / max(mse_g, 1e-12))

            # 逐图 dr 的 PSNR（检验是否依赖全局 dr）
            dr_l = float(g.max() - g.min())
            psnr_l = 10 * math.log10((dr_l ** 2) / max(mse_g, 1e-12)) \
                if dr_l > 1e-8 else float("nan")

            # GT 标准差（patch 信息量）
            gt_std = float(g.std().item())

            # 逐波段 Pearson 相关（与 eval_metrics.calc_cc 同口径）
            ccs = []
            for b in range(g.shape[1]):
                a = (p[:, b] - p[:, b].mean()).reshape(-1)
                c = (g[:, b] - g[:, b].mean()).reshape(-1)
                den = math.sqrt((a * a).sum().item() * (c * c).sum().item())
                ccs.append(0.0 if den < 1e-8 else float((a * c).sum().item() / den))
            rows.append(dict(cc=float(sum(ccs) / len(ccs)),
                             psnr=psnr_g, psnr_local=psnr_l,
                             gt_std=gt_std, val_dr=float(val_dr)))
    print(f"  [{stage}] {len(rows)} samples, val_dr={val_dr:.6f}")
    return rows


def paired(xs, ys):
    n = min(len(xs), len(ys))
    xs, ys = xs[:n], ys[:n]
    d = [x - y for x, y in zip(xs, ys)]
    m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float("nan")
    return dict(mean=m, sd=sd, ci=(m - 1.96 * se, m + 1.96 * se), t=t,
                n=n, cohen_d=(m / sd if sd > 0 else float("nan")))


def report(name, A, B, keys):
    print(f"\n-- {name} --")
    out = {}
    for k in keys:
        r = paired([r[k] for r in A], [r[k] for r in B])
        out[k] = r
        print(f"  {k:12s} Δ={r['mean']:+.4f}  sd={r['sd']:.4f}  "
              f"95%CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  "
              f"t={r['t']:+.2f}  d={r['cohen_d']:+.2f}  n={r['n']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs=2, default=["mtxa_mis_xa", "mtxa_mis_noxa"])
    ap.add_argument("--subset", type=int, default=3000)
    ap.add_argument("--val_frac", type=float, default=0.9)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")
    print(f"[info] device={device}")

    data = {s: collect(s, device, subset=args.subset, val_frac=args.val_frac)
            for s in args.stages}
    a, b = args.stages

    keys = ["psnr", "psnr_local", "cc", "gt_std"]
    res = {"all": report(f"全样本 {a} − {b}", data[a], data[b], keys)}

    # 按 GT 方差分层：中位数切分
    stds = sorted(r["gt_std"] for r in data[b])
    med = stds[len(stds) // 2]
    print(f"\n[info] GT std 中位数 = {med:.5f}  "
          f"(min={stds[0]:.5f}, max={stds[-1]:.5f})")
    hi = [i for i, r in enumerate(data[b]) if r["gt_std"] >= med]
    print(f"[info] 高方差层 n={len(hi)}")
    res["high_var"] = report(
        f"高方差层(gt_std≥{med:.4f}) {a} − {b}",
        [data[a][i] for i in hi], [data[b][i] for i in hi], keys)

    # CC 与 GT 方差的关系：分四档看 CC 均值
    q = [stds[int(len(stds) * f)] for f in (0.0, 0.25, 0.5, 0.75)]
    print("\n-- CC 随 GT 方差分层（看是否崩塌）--")
    buckets = []
    for i in range(4):
        lo, hi_b = q[i], (q[i + 1] if i < 3 else float("inf"))
        idx = [j for j, r in enumerate(data[b]) if lo <= r["gt_std"] < hi_b]
        if not idx:
            continue
        cc_b = sum(data[b][j]["cc"] for j in idx) / len(idx)
        cc_a = sum(data[a][j]["cc"] for j in idx) / len(idx)
        ps_b = sum(data[b][j]["psnr"] for j in idx) / len(idx)
        print(f"  gt_std∈[{lo:.4f},{hi_b:.4f}) n={len(idx):3d}  "
              f"CC_noxa={cc_b:+.3f}  CC_xa={cc_a:+.3f}  PSNR_noxa={ps_b:.2f}")
        buckets.append(dict(lo=lo, hi=(hi_b if hi_b != float("inf") else None),
                            n=len(idx), cc_noxa=cc_b, cc_xa=cc_a,
                            psnr_noxa=ps_b))
    res["cc_by_gtstd"] = buckets
    res["median_gt_std"] = med

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n[saved] {args.save}")


if __name__ == "__main__":
    main()
