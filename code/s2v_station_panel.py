# -*- coding: utf-8 -*-
"""SEN2VENµS 跨站点面板：把数据层 Δ_floor 与方法层 Δ_model 汇总为站点判决表。

数据来源（全部为本仓库已产出的记录，不重算）：
  * 数据层 Δ_floor：experiments/s2v_crossstation_dfloor.json
      sites[<SITE>].dfloor.psnr.mean_diff   （配对口径，与主文 +7.99 / +6.44 可比）
  * 方法层 Δ_model：experiments/audit_axis_s2v_*.json
      audit_gain_db（逐样本配对 model − bicubic）、audit_p、passes_floor、overall、n_test

派生量：捕获比 β = Δ_model /(Δ_floor + Δ_model)  —— 方法在双三次之上新增的份额。

用法：python code/s2v_station_panel.py [--out experiments/s2v_station_panel.json]
"""
import argparse
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")

# ceiling 判据的 β 上界。主文中 10 个非失稳臂的捕获比在 (1.8%, 7.7%) 区间内**无臂落地**，
# 故任何落在该区间内的阈值都给出同一划分；这里取区间下缘以下的 0.02（=2%）。
BETA_CEIL = 0.02

# 站点 -> 模型层审计文件（glob 模式，按 seed 汇总）
AUDIT_GLOBS = {
    "FGMANAUS": ["audit_axis_s2v_sisr.json"],
    "SO2": ["audit_axis_s2v_so2_s*.json"],
    "NARYN": ["audit_axis_s2v_naryn_s*.json"],
    "ATTO": ["audit_axis_s2v_atto_s*.json"],
    "SUDOUE-2": ["audit_axis_s2v_sudoue_2_s*.json"],
}


def load_dfloor():
    p = os.path.join(EXP, "s2v_crossstation_dfloor.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p, encoding="utf-8"))
    out = {}
    for site, rec in d.get("sites", {}).items():
        try:
            out[site] = dict(
                dfloor=rec["dfloor"]["psnr"]["mean_diff"],
                n_val=rec.get("n_val"),
                n_total=rec.get("n_total"),
                p=rec["dfloor"]["psnr"].get("p"),
            )
        except Exception:  # noqa: BLE001
            pass
    return out


def load_models():
    out = {}
    for site, pats in AUDIT_GLOBS.items():
        seeds = []
        for pat in pats:
            for f in sorted(glob.glob(os.path.join(EXP, pat))):
                try:
                    d = json.load(open(f, encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                g = d.get("audit_gain_db", d.get("gain_db"))
                if g is None:
                    continue
                seeds.append(dict(file=os.path.basename(f), dm=float(g),
                                  p=d.get("audit_p", d.get("paired_psnr", {}).get("p")),
                                  passes=d.get("passes_floor"),
                                  overall=d.get("overall"), n=d.get("n_test")))
        if seeds:
            out[site] = seeds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXP, "s2v_station_panel.json"))
    args = ap.parse_args()

    df = load_dfloor()
    md = load_models()
    sites = sorted(set(df) | set(md), key=lambda s: (s not in df, s))

    rows = []
    for s in sites:
        drec = df.get(s, {})
        ms = md.get(s, [])
        dm_vals = [m["dm"] for m in ms]
        dm = statistics.fmean(dm_vals) if dm_vals else None
        sd = statistics.stdev(dm_vals) if len(dm_vals) > 1 else 0.0
        dfl = drec.get("dfloor")
        beta = (dm / (dfl + dm)) if (dm is not None and dfl is not None and (dfl + dm) != 0) else None
        # 判据（与主文 Methods "Separating the three regimes by magnitude" 一致）：
        #   positive  = 每一 seed 的双三次以上增益都显著为正 (p < 0.05 且 Δ_model > 0)
        #   ceiling   = 没有一个 seed 显著，且 β < BETA_CEIL（捕获比极小 = bicubic 已吃掉余量）
        #   floor-failure = Δ_floor ≤ 0
        # 注意 passes_floor 只表示"模型优于常数预测器"，不能用作 regime 判据。
        sig = [(m["p"] is not None and m["p"] < 0.05 and m["dm"] > 0.0) for m in ms]
        if dfl is not None and dfl <= 0:
            regime = "floor-failure"
        elif not ms or beta is None:
            regime = "—"
        elif all(sig):
            regime = "positive"
        elif not any(sig) and beta < BETA_CEIL:
            regime = "ceiling"
        else:
            regime = "mixed"
        rows.append(dict(site=s, n_total=drec.get("n_total"), n_val=drec.get("n_val"),
                         dfloor=dfl, n_seeds=len(ms), dm=dm, dm_sd=sd, beta=beta,
                         regime=regime, seeds=ms))

    hdr = f"{'site':10s} {'n_tot':>6s} {'n_val':>6s} {'Dfloor':>8s} {'Dmodel':>9s} {'+-sd':>6s} {'beta':>7s}  regime"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f = lambda v, n=2: ("%.*f" % (n, v)) if isinstance(v, (int, float)) else "—"
        b = f"{r['beta']*100:.2f}%" if isinstance(r["beta"], (int, float)) else "—"
        print(f"{r['site']:10s} {str(r['n_total']):>6s} {str(r['n_val']):>6s} "
              f"{f(r['dfloor']):>8s} {f(r['dm']):>9s} {f(r['dm_sd']):>6s} {b:>7s}  {r['regime']}")

    json.dump(dict(rows=rows), open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
