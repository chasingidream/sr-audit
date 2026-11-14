#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WorldStrat 第二数据集「站点面板」——留出景清单 + 确定性地理站点划分 + 逐景归一化。

背景（R2 残余）：方法层跨站点证据目前全部来自 SEN2VENµS 一个数据集。
本工具为第二数据集 WorldStrat 建一个**平行站点面板**，且严格无泄漏：
  * 训练（external_worldstrat.py, subset=600）只用 manifest[0:600]（= 全部 ASMSpotter +
    全部 Amnesty POI + 61 景 Landcover）；
  * 本面板评测 manifest[600:3927]（= 2,346 景 Landcover + 981 景 UNHCR），**全部未见**。

站点定义 = **确定性大陆盒**（不用 k-means：实测 k-means 在 6 个种子下给出 3 个不同
局部最优 → 站点不可复现，不能进论文）。盒边界：
    lon < -30                 -> Americas
    lon >= 120                -> Asia-Pacific
    60 <= lon < 120           -> Asia
    -30 <= lon < 60, lat > 35 -> Europe
    -30 <= lon < 60, lat <=35 -> Africa
零超参、纯查表、处处可复现。共 5 站，与 SEN2VENµS 五站点面板一一对应。

抽样：每站按 **确定性等距步长**（np.linspace 取整）抽 ≤ CAP 景，避免 RNG 与顺序偏置。

逐景归一化：**复用** code/compute_worldstrat_scene_norm.py 的 read_tif / LR_BANDS，
保证与训练口径**逐位一致**（LR 取 (3,2,1)，/65535，pooled 全帧全像素）。

产物：
  data/processed/worldstrat_manifest_holdout.json   评测用有序 manifest（list[{lr,hr}]）
  data/processed/worldstrat_scene_norm_holdout.json 与其对齐的 list[[mu,sd]]
  experiments/ws_station_split.json                 站点划分 + 逐站 n + 评测序站标签
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "code"))
# 复用训练侧完全相同的读取/归一化实现，杜绝口径漂移
import compute_worldstrat_scene_norm as cwn  # noqa: E402
from utils.paths import rebase_manifest  # noqa: E402

MANIFEST = os.path.join(BASE, "data", "processed", "worldstrat_manifest.json")
FLOOR = os.path.join(BASE, "experiments", "stratified_floor_ws.json")
OUT_MAN = os.path.join(BASE, "data", "processed", "worldstrat_manifest_holdout.json")
OUT_NORM = os.path.join(BASE, "data", "processed", "worldstrat_scene_norm_holdout.json")
OUT_SPLIT = os.path.join(BASE, "experiments", "ws_station_split.json")

TRAIN_SUBSET = 600          # external_worldstrat.py 的 dataset.subset
CAP = 200                   # 每站评测上限
STATION_ORDER = ["Americas", "Asia", "Asia-Pacific", "Africa", "Europe"]
# 退化（空白/近常量）景过滤阈值：LR 全帧全像素均值 <= 1e-4（反射率尺度 0..1）
# 这类景 sd≈1e-6 -> (x-mu)/sd 数值爆炸（实测 GT 极差可达 1.9e5），会污染 val_dr 与
# 逐样本 PSNR。训练 600 景中此类为 0（复核见 docs/R2_..._结果_2026-10-07.md）。
MU_MIN = 1e-4


def cov_of(entry):
    """manifest 条目 -> 场景 id（= 目录名），与 stratified_floor_ws.json 的 cov 对齐。"""
    p = entry["hr"].replace("\\", "/")
    return p.split("/")[-2]


def station_of(lat, lon):
    if lon < -30:
        return "Americas"
    if lon >= 120:
        return "Asia-Pacific"
    if lon >= 60:
        return "Asia"
    return "Europe" if lat > 35 else "Africa"


def fit_norm(item):
    """pooled 逐景 (mu, sd)——与 compute_worldstrat_scene_norm 逐位一致。"""
    x = np.concatenate([cwn.read_tif(p, cwn.LR_BANDS).reshape(-1)
                        for p in item["lr"]])
    return float(x.mean()), float(x.std()) + 1e-6


def select_station(manifest, idxs, cap, mu_min, cache):
    """在站点成员（manifest 索引，保序）上做确定性等距抽样 + 退化过滤 + 补齐。

    先取等距 cap 个，丢弃退化景；不足 cap 时按原序补齐（仍过滤退化）。
    返回 (选中的 manifest 索引升序列表, 被判为退化的 manifest 索引列表)。
    """
    n = len(idxs)
    if n == 0:
        return [], []
    prim = sorted({int(v) for v in np.linspace(0, n - 1, min(cap, n)).round()})
    chosen, rejected = [], []

    def consider(j):
        i = idxs[j]
        if i not in cache:
            cache[i] = fit_norm(manifest[i])
        mu, _sd = cache[i]
        if mu > mu_min:
            chosen.append(i)
            return True
        rejected.append(i)
        return False

    prim_set = set(prim)
    for j in prim:
        consider(j)
    if len(chosen) < min(cap, n):
        for j in range(n):
            if len(chosen) >= cap:
                break
            if j in prim_set:
                continue
            consider(j)
    return sorted(chosen), sorted(set(rejected))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--mu_min", type=float, default=MU_MIN)
    ap.add_argument("--no_norm", action="store_true",
                    help="只出划分清单，不写逐景 (mu,sd) 文件（仍会算以过滤退化景）")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    manifest = rebase_manifest(manifest)   # 发布版清单为相对路径，按数据根补全
    floor = json.load(open(FLOOR, encoding="utf-8"))["scenes"]
    assert len(manifest) == len(floor), (len(manifest), len(floor))
    for i, (m, s) in enumerate(zip(manifest, floor)):
        assert cov_of(m) == s["cov"], f"顺序不对齐 @ {i}: {cov_of(m)} vs {s['cov']}"

    hold_idx = list(range(TRAIN_SUBSET, len(manifest)))
    print(f"[info] 总景 {len(manifest)}；训练 [0:{TRAIN_SUBSET})；"
          f"留出 [{TRAIN_SUBSET}:{len(manifest)}) = {len(hold_idx)} 景")

    # 逐站成员（保留原 manifest 序）
    members = {k: [] for k in STATION_ORDER}
    for i in hold_idx:
        s = floor[i]
        members[station_of(s["lat"], s["lon"])].append(i)

    cache, picked, rejected_all = {}, [], []
    for k in STATION_ORDER:
        sel, rej = select_station(manifest, members[k], args.cap, args.mu_min, cache)
        picked.extend(sel)
        rejected_all.extend(rej)
    picked = sorted(set(picked))
    print(f"[info] 评测样本 {len(picked)} 景（cap={args.cap}）；"
          f"退化景剔除 {len(rejected_all)} 景（LR μ≤{args.mu_min:g}）")

    eval_manifest = [manifest[i] for i in picked]
    eval_station = [station_of(floor[i]["lat"], floor[i]["lon"]) for i in picked]
    norm = [list(cache[i]) for i in picked]

    # 逐站统计
    stations = {}
    for k in STATION_ORDER:
        allk = members[k]
        rejk = set(rejected_all) & set(allk)
        evk = [j for j, i in enumerate(picked) if eval_station[j] == k]
        if not allk:
            continue
        dfa = np.array([floor[i]["delta_floor"] for i in allk])
        dfe = np.array([floor[i]["delta_floor"]
                        for i in picked if i in set(allk)])
        lats = np.array([floor[i]["lat"] for i in allk])
        lons = np.array([floor[i]["lon"] for i in allk])
        stations[k] = dict(
            n_holdout=len(allk), n_eval=len(evk), n_rejected=len(rejk),
            dfloor_holdout_mean=float(dfa.mean()),
            dfloor_holdout_sd=float(dfa.std(ddof=1)) if len(dfa) > 1 else 0.0,
            dfloor_eval_mean=float(dfe.mean()) if len(dfe) else None,
            frac_pos_holdout=float((dfa > 0).mean()),
            lat_range=[float(lats.min()), float(lats.max())],
            lon_range=[float(lons.min()), float(lons.max())],
        )

    json.dump(eval_manifest, open(OUT_MAN, "w", encoding="utf-8"))
    print(f"[saved] {OUT_MAN}  ({len(eval_manifest)} 景)")

    if not args.no_norm:
        json.dump(norm, open(OUT_NORM, "w", encoding="utf-8"))
        arr = np.array(norm)
        print(f"[saved] {OUT_NORM}  mu μ={arr[:,0].mean():.4f} "
              f"sd μ={arr[:,1].mean():.4f} min={arr[:,1].min():.4f}")

    json.dump(dict(
        box_rule="lon<-30:Americas; lon>=120:Asia-Pacific; 60<=lon<120:Asia; "
                 "-30<=lon<60 & lat>35:Europe; else Africa",
        degenerate_rule=f"LR pooled mu <= {args.mu_min:g}",
        train_subset=TRAIN_SUBSET, n_holdout_total=len(hold_idx),
        per_station_cap=args.cap, n_eval=len(picked),
        n_rejected=len(rejected_all),
        rejected_covs=[floor[i]["cov"] for i in sorted(set(rejected_all))],
        station_order=STATION_ORDER, stations=stations,
        eval_covs=[cov_of(m) for m in eval_manifest],
        eval_station=eval_station,
    ), open(OUT_SPLIT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[saved] {OUT_SPLIT}")

    print("\n=== 站点划分（留出全集 / 评测样本）===")
    hdr = (f"{'station':14s} {'n_hold':>7s} {'n_eval':>7s} {'n_rej':>6s} "
           f"{'Dfloor(hold)':>13s} {'frac>0':>7s}")
    print(hdr); print("-" * len(hdr))
    for k in STATION_ORDER:
        if k not in stations:
            continue
        s = stations[k]
        print(f"{k:14s} {s['n_holdout']:7d} {s['n_eval']:7d} {s['n_rejected']:6d} "
              f"{s['dfloor_holdout_mean']:+13.2f} {100*s['frac_pos_holdout']:6.1f}%")


if __name__ == "__main__":
    main()
