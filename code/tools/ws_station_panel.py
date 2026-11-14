# -*- coding: utf-8 -*-
"""WorldStrat 第二数据集「站点面板」：把留出推理的逐样本 PSNR 汇总为站点判决表。

与 SEN2VENµS 面板（code/s2v_station_panel.py）同构：数据层 Δ_floor + 方法层 Δ_model
-> 捕获比 β -> regime 判决（positive / ceiling / floor-failure）。

关键口径（与主文 Methods "Separating the three regimes by magnitude" 一致）：
  Δ_floor   = PSNR(bicubic) − PSNR(mean_GT)   **同一裁剪**（非跨文件口径）
  Δ_model   = PSNR(model)   − PSNR(bicubic)   逐 seed 独立
  β         = Δ_model / (Δ_floor + Δ_model)   地板失效时未定义（Δ_floor+Δ_model<0 亦无意义）
  positive  = 每一 seed 的 Δ_model 都显著为正（p<0.05 且 Δ_model>0）
  ceiling   = 没有 seed 显著，且 β < BETA_CEIL
  floor-failure = Δ_floor ≤ 0

用法：
  python code/tools/ws_station_panel.py \
      --eval experiments/ws_eval_holdout.json \
      --split experiments/ws_station_split.json \
      --out experiments/ws_station_panel.json
"""
import argparse
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BETA_CEIL = 0.02      # 与 s2v_station_panel.py 一致（1.8%–7.7% 空隙下任取皆同）
BASELINES = ("bicubic", "mean")


def load_eval(p):
    d = json.load(open(p, encoding="utf-8"))
    ps = d["per_sample"]
    stages = [k for k in ps if k not in BASELINES]
    return d, stages


def ttest1(xs):
    """单样本 t 检验（H0: 均值=0）；返回 (mean, sd, p, n)。"""
    n = len(xs)
    if n < 2:
        return (statistics.fmean(xs) if xs else float("nan"),
                0.0, float("nan"), n)
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    if sd == 0:
        return m, 0.0, 0.0, n
    try:
        from scipy import stats as st
        t, p = st.ttest_1samp(xs, 0.0)
        return m, sd, float(p), n
    except Exception:
        se = sd / math.sqrt(n)
        t = m / se
        # 粗略双尾 p（正态近似），仅供无 scipy 时兜底
        p = math.erfc(abs(t) / math.sqrt(2))
        return m, sd, p, n


def series(ps, key, idxs):
    return [ps[key][i]["psnr"] for i in idxs if i < len(ps[key])]


def analyse(ps, stages, idxs, label):
    """对给定样本索引集合算 Δ_floor / 各 seed Δ_model / regime。"""
    bic = series(ps, "bicubic", idxs)
    mean = series(ps, "mean", idxs) if "mean" in ps else None
    dfloor = [b - m for b, m in zip(bic, mean)] if mean else []
    dfloor_stat = ttest1(dfloor) if dfloor else (float("nan"), 0.0, float("nan"), 0)

    seeds = []
    for st in stages:
        pred = series(ps, st, idxs)
        dm = [p - b for p, b in zip(pred, bic)]
        if not dm:
            continue
        m, sd, p, n = ttest1(dm)
        seeds.append(dict(stage=st, dm=m, dm_sd=sd, p=p, n=n,
                          dm_frac_pos=sum(1 for d in dm if d > 0) / len(dm)))

    dm_vals = [s["dm"] for s in seeds]
    dm = statistics.fmean(dm_vals) if dm_vals else None
    dm_sd = statistics.stdev(dm_vals) if len(dm_vals) > 1 else 0.0
    dfl = dfloor_stat[0] if dfloor else None
    # 模型相对常数预测器的余量 = Δ_floor + Δ_model（>0 才算真正过了地板）
    below_floor = (dfl + dm) if (dfl is not None and dm is not None) else None
    # β 只在 Δ_floor > 0（地板未塌陷）时才有意义：地板失效时 Δ_floor+Δ_model<0，
    # 比值给出无意义的大负数，故显式置空（与主文对 floor-failure 的处理一致）。
    beta = (dm / (dfl + dm)) if (dm is not None and dfl is not None
                                 and dfl > 0 and (dfl + dm) != 0) else None

    sig = [(s["p"] == s["p"] and s["p"] < 0.05 and s["dm"] > 0) for s in seeds]
    if dfl is not None and dfl <= 0:
        regime = "floor-failure"
    elif not seeds or beta is None:
        regime = "—"
    elif all(sig):
        regime = "positive"
    elif not any(sig) and beta < BETA_CEIL:
        regime = "ceiling"
    else:
        regime = "mixed"
    # 过地板判据：模型是否显著优于常数预测器
    clears_floor = (below_floor is not None and below_floor > 0)

    return dict(label=label, n=len(idxs),
                dfloor=dfl, dfloor_sd=dfloor_stat[1], dfloor_p=dfloor_stat[2],
                dmodel=dm, dmodel_sd_seeds=dm_sd, below_floor=below_floor,
                clears_floor=clears_floor, beta=beta, regime=regime,
                seeds=seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="experiments/ws_eval_holdout.json")
    ap.add_argument("--split", default="experiments/ws_station_split.json")
    ap.add_argument("--out", default="experiments/ws_station_panel.json")
    args = ap.parse_args()

    ev, stages = load_eval(os.path.join(ROOT, args.eval))
    sp = json.load(open(os.path.join(ROOT, args.split), encoding="utf-8"))
    station_of = sp["eval_station"]
    n = len(station_of)
    for st in stages + ["bicubic"]:
        if len(ev["per_sample"][st]) < n:
            print(f"[warn] {st} 只有 {len(ev['per_sample'][st])} 样本 < 站点表 {n}，按交集截断")
    n = min(n, *(len(ev["per_sample"][k]) for k in list(stages) + ["bicubic"]))

    order = sp["station_order"]
    rows = []
    for k in order:
        idxs = [i for i in range(n) if station_of[i] == k]
        if not idxs:
            continue
        rows.append(analyse(ev["per_sample"], stages, idxs, k))
    rows.append(analyse(ev["per_sample"], stages, list(range(n)), "ALL(pooled)"))

    print(f"[info] eval={args.eval}  stages={stages}  n={n}")
    hdr = (f"{'station':14s} {'n':>5s} {'Dfloor':>8s} {'p_floor':>8s} "
           f"{'Dmodel':>8s} {'+-sd(seed)':>10s} {'Dfloor+Dmodel':>13s} {'beta':>8s}  regime")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        f = lambda v, d=2: ("%.*f" % (d, v)) if isinstance(v, (int, float)) and v == v else "—"
        b = f"{r['beta']*100:.2f}%" if isinstance(r["beta"], (int, float)) and r["beta"] == r["beta"] else "—"
        print(f"{r['label']:14s} {r['n']:5d} {f(r['dfloor']):>8s} {f(r['dfloor_p'],4):>8s} "
              f"{f(r['dmodel']):>8s} {f(r['dmodel_sd_seeds']):>10s} "
              f"{f(r['below_floor']):>13s} {b:>8s}  {r['regime']}")

    print("\n逐 seed Δ_model (dB) / p：")
    for r in rows:
        parts = [f"{s['stage'].replace('mtxa_aligned','aligned').replace('_s',' s')}:"
                 f"{s['dm']:+.3f}(p={s['p']:.1e})" for s in r["seeds"]]
        print(f"  {r['label']:14s} " + "  ".join(parts))

    cleared = [r["label"] for r in rows if r["clears_floor"]]
    print(f"\n过地板站点：{cleared if cleared else '无（全部 floor-failure）'}")

    json.dump(dict(meta=dict(eval=args.eval, split=args.split, stages=stages,
                             n=n, beta_ceil=BETA_CEIL), rows=rows),
              open(os.path.join(ROOT, args.out), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
