#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p2_ws_segmentation.py —— P2-1b：保真度敏感下游（k-means 伪分割）@ WS 6.63×。

背景（用户指令 2026-09-26）：EuroSAT 粗分类对 SR 保真度不敏感（受控
floor-failure 负对照 ΔmAcc=-0.014 不显著），换成 WorldStrat 6.63× 跨传感器
真实 floor-failure 场景 + 像素级分割任务，干净演示「audit-fail → 下游有害」。

协议：
  1. 伪 GT：对 GT（per_scene 归一化 RGB）做全局 k-means（K=6，60 场景混池
     下采样拟合 + 全像素最近中心标注）→ 跨场景语义一致的分割标签；
  2. 场景切分：rng(0) 置换 60 场景 → 40 训练 / 20 评测（跨臂、跨种子固定）；
  3. 逐臂 in-domain：每臂（gt/bicubic/ours/edsr/wdsr/hrnet，用 *_reg.npy
     统一配准版）独立训练一个小型 U-Net（消除 frozen-classifier 分布漂移
     混淆，P2-1 已证两种协议同结论，此处取更公平者）；
  4. 多种子（默认 0/1/2）：mean±std + 与 bicubic 臂逐种子配对差；
  5. monotonicity test：下游 mIoU 差是否随 SR-Audit 增益排序单调
     （hrnet +0.995 > bicubic 0 > ours −0.88 > edsr −2.44 > wdsr −3.04）。

用法：
  python code/p2_ws_segmentation.py                # 全六臂 × 3 seeds
  python code/p2_ws_segmentation.py --arms bicubic,ours --seeds 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))

_DLL = r"C:\Windows\System32\vcomp140.dll"
_HIDDEN = _DLL + ".hide"


def _hide_dll():
    if os.path.exists(_HIDDEN) and not os.path.exists(_DLL):
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            pass
    if os.path.exists(_DLL):
        try:
            os.rename(_DLL, _HIDDEN)
            return True
        except Exception:
            print("[warn] could not hide vcomp140.dll")
    return False


def _restore_dll():
    if os.path.exists(_HIDDEN):
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            pass


_hide_dll()
import atexit  # noqa: E402
atexit.register(_restore_dll)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT_DIR = os.path.join(BASE, "experiments", "p2_ws_seg")

# 已发布审计增益（experiments/sota_review/*.json；hrnet=官方口径）
AUDIT_GAIN = {"gt": None, "bicubic": 0.0, "ours": -0.88, "edsr": -2.44,
              "wdsr": -3.04, "hrnet": +0.995}


# ---------------------------------------------------------------- 伪 GT
def make_pseudo_labels(gt, k=6, seed=0, stride=4):
    """全局 k-means 伪分割标签。gt: (N,3,H,W) float32（归一化空间）。

    混池下采样（stride）拟合中心 → 全像素最近中心标注。返回 (N,H,W) uint8。
    """
    from scipy.cluster.vq import kmeans2
    N, C, H, W = gt.shape
    feats = gt[:, :, ::stride, ::stride].transpose(0, 2, 3, 1).reshape(-1, C)
    feats = np.ascontiguousarray(feats, dtype=np.float64)
    cen, _ = kmeans2(feats, k, iter=25, minit="++", seed=seed)
    print(f"[kmeans] K={k} seed={seed} centers(mu):\n{np.round(cen.mean(1), 3)}")
    labels = np.empty((N, H, W), dtype=np.uint8)
    for i in range(N):
        px = gt[i].transpose(1, 2, 0).reshape(-1, C).astype(np.float64)
        d = ((px[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        labels[i] = d.argmin(1).reshape(H, W).astype(np.uint8)
    return labels, cen


# ---------------------------------------------------------------- 小型 U-Net
class UNetSmall(nn.Module):
    """3-level 小型 U-Net（~0.8M params），全分辨率 logits 输出。"""

    def __init__(self, in_ch=3, n_cls=6, w=(32, 64, 128)):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))
        self.e1 = blk(in_ch, w[0])
        self.e2 = blk(w[0], w[1])
        self.e3 = blk(w[1], w[2])
        self.d2 = blk(w[2] + w[1], w[1])
        self.d1 = blk(w[1] + w[0], w[0])
        self.head = nn.Conv2d(w[0], n_cls, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.e1(x)                       # 424
        x2 = self.e2(self.pool(x1))           # 212
        x3 = self.e3(self.pool(x2))           # 106
        y2 = self.d2(torch.cat([F.interpolate(x3, scale_factor=2, mode="bilinear",
                                               align_corners=False), x2], 1))
        y1 = self.d1(torch.cat([F.interpolate(y2, scale_factor=2, mode="bilinear",
                                              align_corners=False), x1], 1))
        return self.head(y1)


# ---------------------------------------------------------------- 训练 / 评测
def train_arm(images, labels, train_idx, n_cls, seed, iters, batch, crop, device):
    """逐臂 in-domain 训练。images: (N,3,H,W) GPU, labels: (N,H,W) GPU uint8."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    model = UNetSmall(3, n_cls).to(device)
    # 类权重（逆频率，训练集统计，clip 防爆）
    cnt = np.bincount(labels[train_idx].reshape(-1).cpu().numpy(),
                      minlength=n_cls).astype(np.float64)
    w = cnt.sum() / np.maximum(cnt, 1) ** 1.0
    w = torch.from_numpy((w / w.mean()).clip(0.2, 5.0)).float().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters,
                                                       eta_min=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    model.train()
    for it in range(iters):
        si = np.random.choice(train_idx, batch)
        ys = np.random.randint(0, images.shape[-2] - crop + 1, batch)
        xs = np.random.randint(0, images.shape[-1] - crop + 1, batch)
        xb = torch.stack([images[s, :, y:y + crop, x:x + crop]
                          for s, y, x in zip(si, ys, xs)])
        yb = torch.stack([labels[s, y:y + crop, x:x + crop]
                          for s, y, x in zip(si, ys, xs)]).long()
        # 随机翻转增广
        if np.random.rand() < 0.5:
            xb, yb = torch.flip(xb, (-1,)), torch.flip(yb, (-1,))
        if np.random.rand() < 0.5:
            xb, yb = torch.flip(xb, (-2,)), torch.flip(yb, (-2,))
        loss = lossf(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def eval_arm(model, images, labels, eval_idx, n_cls, device):
    """全场景推理，逐场景 confusion → mIoU(场景内出现类) / 像素精度。"""
    model.eval()
    ious, accs = [], []
    with torch.no_grad():
        for i in eval_idx:
            logits = model(images[i:i + 1])
            pred = logits.argmax(1)[0].cpu().numpy()
            gt = labels[i].cpu().numpy()
            present = np.unique(gt)
            inter = np.zeros(n_cls, np.int64)
            union = np.zeros(n_cls, np.int64)
            for c in present:
                p = pred == c
                g = gt == c
                inter[c] = np.logical_and(p, g).sum()
                union[c] = np.logical_or(p, g).sum()
            iou_c = inter[present] / np.maximum(union[present], 1)
            ious.append(float(iou_c.mean()))
            accs.append(float((pred == gt).mean()))
    return float(np.mean(ious)), float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser(description="WS 6.63x k-means 伪分割下游（P2-1b）")
    ap.add_argument("--arms", default="gt,bicubic,ours,edsr,wdsr,hrnet")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--n_train_scene", type=int, default=40)
    ap.add_argument("--label_seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    # ---------- 数据 ----------
    gt = np.load(os.path.join(OUT_DIR, "gt.npy"))
    meta = json.load(open(os.path.join(OUT_DIR, "meta.json"), encoding="utf-8"))
    N = gt.shape[0]
    print(f"[info] N={N} device={device} arms={arms} seeds={seeds}")

    labels_np, _ = make_pseudo_labels(gt, k=args.k, seed=args.label_seed)
    # 场景切分（跨臂/跨种子固定）
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    train_idx, eval_idx = perm[:args.n_train_scene], perm[args.n_train_scene:]
    print(f"[split] train={len(train_idx)} eval={len(eval_idx)} "
          f"(RandomState(0), 固定)")

    labels = torch.from_numpy(labels_np).to(device)
    imgs = {}
    for a in arms:
        fn = "gt.npy" if a == "gt" else f"{a}_reg.npy"
        x = torch.from_numpy(np.load(os.path.join(OUT_DIR, fn))).to(device)
        # 逐臂【全局】标准化（数据集级仿射，跨场景一致）：仅解决 hrnet 原生
        # 反射率空间 [0,1] 小动态范围在同 LR 下欠训练的问题。⚠ 不能用逐场景
        # 标准化——k-means 标签定义在 per-scene 归一化 GT 上，逐场景变换会使
        # "输入↔标签"映射逐场景不同，oracle 也会被拉平（实测 gt 0.72→0.48）。
        mu = x.mean()
        sd = x.std() + 1e-6
        imgs[a] = (x - mu) / sd
        print(f"[norm] {a}: global mu={float(mu):.4f} sd={float(sd):.4f}")

    # ---------- 逐臂 × 逐种子 ----------
    res = {}
    for a in arms:
        rows = []
        for seed in seeds:
            m = train_arm(imgs[a], labels, train_idx, args.k, seed,
                          args.iters, args.batch, args.crop, device)
            miou, acc = eval_arm(m, imgs[a], labels, eval_idx, args.k, device)
            rows.append({"seed": seed, "miou": round(miou, 5),
                         "acc": round(acc, 5)})
            print(f"  [{a} seed{seed}] mIoU={miou:.4f} acc={acc:.4f} "
                  f"({time.time()-t0:.0f}s)")
        mi = np.array([r["miou"] for r in rows])
        ac = np.array([r["acc"] for r in rows])
        res[a] = {"rows": rows,
                  "miou_mean": round(float(mi.mean()), 5),
                  "miou_std": round(float(mi.std(ddof=1)), 5) if len(mi) > 1 else 0.0,
                  "acc_mean": round(float(ac.mean()), 5),
                  "acc_std": round(float(ac.std(ddof=1)), 5) if len(ac) > 1 else 0.0}

    # ---------- 配对差 vs bicubic + 单调性 ----------
    if "bicubic" in res and len(seeds) > 1:
        b = np.array([r["miou"] for r in res["bicubic"]["rows"]])
        import scipy.stats as st
        for a in arms:
            if a == "bicubic":
                continue
            x = np.array([r["miou"] for r in res[a]["rows"]])
            d = x - b
            t, p = st.ttest_rel(x, b)
            res[a]["delta_miou_vs_bicubic"] = {
                "mean": round(float(d.mean()), 5),
                "sd": round(float(d.std(ddof=1)), 5),
                "p": float(f"{p:.3g}") if p == p else None,
            }
    mono = {}
    have = [(a, AUDIT_GAIN[a], res[a]["miou_mean"]) for a in arms
            if a in res and AUDIT_GAIN.get(a) is not None
            and a != "bicubic" and "delta_miou_vs_bicubic" in res[a]]
    if len(have) >= 3:
        import scipy.stats as st
        gains = [h[1] for h in have]
        dms = [h[2] - res["bicubic"]["miou_mean"] for h in have]
        rho, p_sp = st.spearmanr(gains, dms)
        mono = {"spearman_rho": round(float(rho), 3), "p": round(float(p_sp), 4),
                "points": {h[0]: {"audit_gain": h[1], "d_miou": round(d, 5)}
                           for h, d in zip(have, dms)}}

    # ---------- 落盘 ----------
    out = {"task": "kmeans pseudo-segmentation (K=%d), per-arm in-domain U-Net" % args.k,
           "data": "ws val-60 *_reg.npy (统一 ±16px 粗到细配准)",
           "split": {"n_train_scene": len(train_idx), "n_eval_scene": len(eval_idx)},
           "seeds": seeds, "iters": args.iters, "results": res,
           "monotonicity": mono}
    with open(os.path.join(OUT_DIR, "segmentation_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[saved] segmentation_results.json")

    print("\n== 下游分割（mIoU，逐臂 in-domain，%d seeds）==" % len(seeds))
    bic_m = res["bicubic"]["miou_mean"] if "bicubic" in res else None
    for a in arms:
        r = res[a]
        dl = ""
        if "delta_miou_vs_bicubic" in r:
            dl = f"  Δ={r['delta_miou_vs_bicubic']['mean']:+.4f}" \
                 f" (p={r['delta_miou_vs_bicubic']['p']})"
        print(f"  {a:<8} mIoU {r['miou_mean']:.4f}±{r['miou_std']:.4f}  "
              f"acc {r['acc_mean']:.4f}{dl}")
    if mono:
        print(f"  monotonicity (audit gain ↔ ΔmIoU): Spearman ρ="
              f"{mono['spearman_rho']} (p={mono['p']})")

    # ---------- 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
        xs = [a for a in arms if a in res]
        mu = [res[a]["miou_mean"] for a in xs]
        sd = [res[a]["miou_std"] for a in xs]
        colors = {"gt": "#888888", "bicubic": "#4C72B0", "ours": "#DD8452",
                  "edsr": "#55A868", "wdsr": "#C44E52", "hrnet": "#8172B3"}
        ax1.bar(xs, mu, yerr=sd, capsize=4,
                color=[colors.get(a, "#999") for a in xs])
        if bic_m is not None:
            ax1.axhline(bic_m, ls="--", c="k", lw=0.8, alpha=0.6)
        ax1.set_ylabel("eval mIoU")
        ax1.set_title("Pseudo-segmentation mIoU per arm "
                      f"(K={args.k}, {len(seeds)} seeds)")
        ax1.grid(axis="y", alpha=0.3)
        if mono:
            pts = mono["points"]
            ax2.scatter([v["audit_gain"] for v in pts.values()],
                        [v["d_miou"] for v in pts.values()], s=55,
                        color=[colors.get(a, "#999") for a in pts])
            for a, v in pts.items():
                ax2.annotate(a, (v["audit_gain"], v["d_miou"]),
                             textcoords="offset points", xytext=(5, 3),
                             fontsize=9)
            ax2.axhline(0, ls="--", c="k", lw=0.8, alpha=0.6)
            ax2.set_xlabel("SR-Audit gain (dB)")
            ax2.set_ylabel("Δ eval mIoU vs bicubic")
            ax2.set_title(f"Monotonicity (ρ={mono['spearman_rho']})")
            ax2.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(OUT_DIR, "segmentation_miou.png")
        fig.savefig(p, dpi=130)
        print(f"[saved] {p}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plotting failed: {e}")


if __name__ == "__main__":
    main()
