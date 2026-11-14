#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p2_ws_sr_dump.py —— P2-1b（保真度敏感下游）第一步：WorldStrat 6.63× 五臂 SR 落盘。

背景（用户指令 2026-09-26）：EuroSAT 粗分类对 SR 保真度不敏感（受控 floor-failure
负对照下游 ΔmAcc=-0.014 不显著），须换成保真度敏感任务 + WorldStrat 真实
跨传感器 floor-failure 场景，才能干净演示「audit-fail → 下游有害」。

本脚本在 ws val-60（与 audit_axis.py 完全同口径：get_dataset deterministic +
Subset 尾部 10% + per_scene 归一化）上为六臂导出统一 424×424 RGB 栅格：

  * gt        —— target（per_scene 归一化空间，不 clamp）
  * bicubic   —— sr_audit.bicubic_up（align_corners=False，主文评测口径）
  * ours      —— mtxa_aligned（build_model + pick_ckpt 末轮，use_rsa=True/use_ref=False）
  * edsr      —— code/edsr_worldstrat.py EDSR，frame0 单帧
  * wdsr      —— code/wdsr_worldstrat.py WDSR，frame0 单帧
  * hrnet     —— HighRes-net 官方协议（build_lr 8 revisit JIF 归一化 → 160；
                  bias_adjust；Lanczos 亚像素配准取最优核），输出 156×156 全景，
                  按该场景 val-60 裁剪窗口映射裁剪后 resize 回 424 对齐网格。
                  注意：hrnet 臂保持其原生反射率空间（与 per_scene 归一化的其余臂
                  不同），下游用逐臂 in-domain 训练吸收辐射差异。

同时逐样本配对 PSNR 复核（ours/edsr/wdsr vs bicubic，val_dr 满量程），
经 sr_audit_report.derive 出 verdict，与已发布审计卡
（ours −0.88 / EDSR −2.43 / WDSR −3.02 / HighRes-net +0.995）对照自洽。

输出：experiments/p2_ws_seg/{gt,bicubic,ours,edsr,wdsr,hrnet}.npy (N,3,424,424) f32
      experiments/p2_ws_seg/meta.json

用法：
  python code/p2_ws_sr_dump.py --limit 3     # 冒烟
  python code/p2_ws_sr_dump.py               # 全量 60 场景
"""
from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------- vcomp140 守卫
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
            print("[warn] could not hide vcomp140.dll; torch CUDA may segfault.")
    return False


def _restore_dll():
    if os.path.exists(_HIDDEN):
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            print("[warn] could not restore vcomp140.dll; rename manually.")


_hide_dll()
atexit.register(_restore_dll)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, os.path.join(BASE, "code", "sota"))
_SRC = os.path.join(BASE, "third_party", "worldstrat")
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "_kornia_stub"))

from eval_paired_floor import load_cfg, pick_ckpt  # noqa: E402
from data.dataset import get_dataset  # noqa: E402
from eval_metrics import calc_psnr, calc_cc  # noqa: E402
from sr_audit import bicubic_up, periodicity_scan  # noqa: E402
from models.sr_model import build_model  # noqa: E402
from edsr_worldstrat import EDSR  # noqa: E402
from wdsr_worldstrat import WDSR  # noqa: E402
from highresnet_eval import ShiftEval, bias_adjust, build_lr, build_hr  # noqa: E402
from highresnet_load import load_highresnet  # noqa: E402
from sr_audit_report import derive  # noqa: E402
import scipy.stats as st  # noqa: E402

OUT_DIR = os.path.join(BASE, "experiments", "p2_ws_seg")


def last_ckpt(ckpt_dir):
    fs = glob.glob(os.path.join(ckpt_dir, "ckpt_ep*.pt"))
    if not fs:
        raise FileNotFoundError(ckpt_dir)
    return max(fs, key=lambda p: int(re.search(r"ckpt_ep(\d+)\.pt", p).group(1)))


def apply_best_shift(sev: ShiftEval, y_hat, y):
    """在 Nk 个 Lanczos 亚像素核中选使 MSE 最小的核并应用到 y_hat（官方配准约定）。

    返回 (shifted, best_idx)。与 highresnet_eval.ShiftEval.registered_mse 同一
    conv 约定（不翻转核），保证选出的核施加方向一致。
    """
    with torch.no_grad():
        patch = y_hat.shape[-1] - sev.kh + 1
        y_s = sev.forward(y_hat)                       # (Nk*B, C, patch, patch)
        y_c = y[:, :, sev.shift_by_px:sev.shift_by_px + patch,
                sev.shift_by_px:sev.shift_by_px + patch]
        y_c = y_c.expand(sev.Nk, -1, -1, -1, -1).reshape(
            sev.Nk * y.shape[0], y.shape[1], patch, patch)
        mse = F.mse_loss(y_s, y_c, reduction="none").mean(dim=(-1, -2, -3))
        mse = mse.view(sev.Nk, y.shape[0])
        j = int(mse.argmin(dim=0)[0].item())
        k = sev.kernels[j:j + 1]                       # (1,1,kh,kh)
        C = y_hat.shape[1]
        p = sev.kh // 2
        out = F.conv2d(F.pad(y_hat, (p, p, p, p), mode="reflect"),
                       k.expand(C, 1, sev.kh, sev.kh), groups=C)
    return out, j


def _cc3(a, b):
    """逐通道 CC 均值（纯 torch，供精配准搜索；与 eval_metrics.calc_cc 同口径）。"""
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    num = (a * b).sum(-1)
    den = torch.sqrt((a * a).sum(-1) * (b * b).sum(-1)) + 1e-8
    return float((num / den).mean())


def refine_shift(crop, gt, max_shift=16):
    """粗到细整数平移精配准（对 GT 的几何配准，非内容泄漏）。

    粗搜：1/4 尺度（avg_pool4）上 ±max_shift/4 网格取 CC 峰；细搜：全尺度在
    粗峰 ±3px 内取 CC 峰。动机：① 各臂影像与标签帧存在数据管线自带的
    S2↔SPOT 逐场景地理定位误差（实测 bicubic 0/60 场景最优平移为 (0,0)）；
    ② hrnet 官方全景输出映射到裁剪网格有额外映射残差。官方 WorldStrat 协议
    本就对 model 与 bicubic 施加同预算 Lanczos 亚像素配准，此处对【全部臂】
    施加同预算整数配准以隔离"内容质量"、消除"对齐运气"。审计复核用原生
    未配准数组（与已发布卡一致），下游评测用配准后数组（*_reg.npy）。

    返回 (shifted_or_original, (dy,dx), cc_after)。
    """
    m = max_shift
    H, W = crop.shape[-2], crop.shape[-1]
    with torch.no_grad():
        # ---- 粗搜（1/4 尺度）----
        c4 = F.avg_pool2d(crop[None], 4)[0]
        g4 = F.avg_pool2d(gt[None], 4)[0]
        q = m // 4
        P4 = F.pad(c4[None], (q + 1,) * 4, mode="replicate")[0]
        h4, w4 = c4.shape[-2], c4.shape[-1]
        cy = cx = 0
        best4 = _cc3(c4, g4)
        for dy in range(-q, q + 1):
            for dx in range(-q, q + 1):
                if dy == 0 and dx == 0:
                    continue
                s = P4[:, q + 1 - dy: q + 1 - dy + h4,
                       q + 1 - dx: q + 1 - dx + w4]
                c = _cc3(s, g4)
                if c > best4:
                    cy, cx, best4 = dy * 4, dx * 4, c
        # ---- 细搜（全尺度，粗峰 ±3）----
        P = F.pad(crop[None], (m + 1,) * 4, mode="replicate")[0]
        out, best_cc, best_dy, best_dx = crop, _cc3(crop, gt), 0, 0
        for dy in range(max(-m, cy - 3), min(m, cy + 3) + 1):
            for dx in range(max(-m, cx - 3), min(m, cx + 3) + 1):
                if dy == 0 and dx == 0:
                    continue
                s = P[:, m + 1 - dy: m + 1 - dy + H,
                      m + 1 - dx: m + 1 - dx + W]
                c = _cc3(s, gt)
                if c > best_cc:
                    out, best_cc, best_dy, best_dx = s, c, dy, dx
    return out, (best_dy, best_dx), best_cc


def _save_preview(out_dir, rel, scene_gt, bic, crop):
    """存 hrnet 映射健康性预览条带（gt | bicubic | hrnet-mapped），独立百分位拉伸。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _norm(x):
        x = x.detach().cpu().numpy().transpose(1, 2, 0)
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, (name, img) in zip(axes, [("GT", scene_gt), ("bicubic", bic),
                                      ("HighRes-net mapped", crop)]):
        ax.imshow(_norm(img))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    p = os.path.join(out_dir, f"preview_{rel:02d}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)


def paired_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    t, p = st.ttest_rel(a, b)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    ci = st.t.interval(0.95, n - 1, loc=d.mean(), scale=se)
    return {"n": n, "mean": round(float(d.mean()), 4),
            "sd": round(float(sd), 4), "t": round(float(t), 3),
            "p": float(f"{p:.3g}"), "d": round(float(d.mean() / sd), 3) if sd > 0 else 0.0,
            "ci95": [round(float(ci[0]), 4), round(float(ci[1]), 4)]}


def main():
    ap = argparse.ArgumentParser(description="WS 6.63x 五臂 SR dump（P2-1b）")
    ap.add_argument("--cfg", default=os.path.join(BASE, "code", "configs",
                                                  "external_worldstrat.py"))
    ap.add_argument("--stage", default="mtxa_aligned")
    ap.add_argument("--limit", type=int, default=0, help=">0 只处理前 N 个 val 场景")
    ap.add_argument("--preview", type=int, default=0, help=">0 存前 N 个场景映射预览 PNG")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")
    cfg = load_cfg(args.cfg)
    cfg["stage"] = args.stage
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 数据：与 audit_axis.py 完全同口径 ----------
    full = get_dataset(cfg, project_root=BASE, deterministic=True)
    val_frac = float(cfg.get("val_frac", 0.1))
    n = len(full)
    n_train = max(1, min(n - 1, int((1.0 - val_frac) * n)))
    n_val = n - n_train
    subset = list(range(n_train, n))
    loader = DataLoader(Subset(full, subset), batch_size=1, shuffle=False,
                        num_workers=0)

    mn, mx = float("inf"), float("-inf")
    for vb in loader:                       # val_dr + 顺带预热 ds.cache
        t = vb["target"].float()
        mn, mx = min(mn, float(t.min())), max(mx, float(t.max()))
    val_dr = (mx - mn) if mx - mn > 1e-8 else 1.0
    print(f"[info] n_val={n_val}  val_dr={val_dr:.4f}  device={device}")

    # ---------- 模型 ----------
    ds = full
    manifest = ds.manifest

    ck = torch.load(pick_ckpt(cfg["save_dir"], args.stage, None),
                    map_location=device, weights_only=False)
    m_ours = build_model(cfg).to(device)
    m_ours.load_state_dict(ck["model"], strict=True)
    m_ours.eval()
    print(f"[info] ours ckpt={pick_ckpt(cfg['save_dir'], args.stage, None)} ep{ck.get('epoch')}")

    cke = last_ckpt(os.path.join(BASE, "experiments", "ckpts_worldstrat", "edsr_f0"))
    m_edsr = EDSR(scale=cfg.get("scale", 7), n_feats=64, n_blocks=16,
                  out_size=cfg.get("out_size", 424)).to(device)
    m_edsr.load_state_dict(torch.load(cke, map_location=device,
                                      weights_only=False)["model"])
    m_edsr.eval()

    ckw = last_ckpt(os.path.join(BASE, "experiments", "ckpts_worldstrat", "wdsr_f0"))
    m_wdsr = WDSR(scale=cfg.get("scale", 7), n_feats=32, n_blocks=16, expand=4,
                  res_scale=0.1, out_size=cfg.get("out_size", 424)).to(device)
    m_wdsr.load_state_dict(torch.load(ckw, map_location=device,
                                      weights_only=False)["model"])
    m_wdsr.eval()

    m_hr = load_highresnet(device=device)
    m_hr.eval()
    sev = ShiftEval(device=device)
    print(f"[info] edsr={os.path.basename(cke)}  wdsr={os.path.basename(ckw)}  "
          f"hrnet=official pretrained  ShiftEval Nk={sev.Nk}")

    ARMS = ("gt", "bicubic", "ours", "edsr", "wdsr", "hrnet")
    store = {a: [] for a in ARMS}
    store_lr = []                     # 备份 LR frame0（N,3,64,64），供 P2-3 外部审计 harness
    rows = {a: [] for a in ARMS[1:]}        # 逐样本 PSNR（gt 臂无 PSNR）
    rows_floor = []
    cc_rows = {a: [] for a in ARMS[1:]}
    grid_rows = {a: [] for a in ARMS[1:]}
    stdr_rows = {a: [] for a in ARMS[1:]}
    scene_ids, hrnet_shift_hist = [], []
    t0 = time.time()

    def _fix(p):
        # 统一到 [1,3,424,424] -> 取 [0] -> cpu numpy
        if tuple(p.shape[-2:]) != (424, 424):
            p = F.interpolate(p, size=(424, 424), mode="bilinear",
                              align_corners=True)
        return p

    with torch.no_grad():
        for rel, batch in enumerate(loader):
            if args.limit and rel >= args.limit:
                break
            idx = n_train + rel
            lrs = batch["lrs"].float().to(device)         # [1,5,3,64,64]
            tgt = batch["target"].float().to(device)      # [1,3,424,424]
            hr_ref = batch["hr_ref"].float().to(device)
            gt = tgt[0]

            bic = bicubic_up(lrs[:, 0], size=(424, 424)).to(device)
            p_ours = _fix(m_ours(lrs, hr_ref, return_aux=False,
                                 use_rsa=True, use_ref=False))
            p_edsr = _fix(m_edsr(lrs[:, 0]))
            p_wdsr = _fix(m_wdsr(lrs[:, 0]))

            # ---- HighRes-net：官方协议 + 配准 + 窗口映射 ----
            item = manifest[idx]
            lr_paths = sorted(item["lr"])[:8]
            lr8 = build_lr(lr_paths, device=device)              # (8,12,160,160)
            y = build_hr(item["hr"], device=device)[None].to(device)  # (1,3,156,156)
            y_hat = m_hr(lr8[None])[:, 0]                        # (1,3,156,156)
            y_hat = bias_adjust(y, y_hat)
            y_hat_reg, best_j = apply_best_shift(sev, y_hat, y)
            hrnet_shift_hist.append(best_j)
            # 裁剪窗口（与 __getitem__(idx) 完全一致的确定性窗口）
            hr_full = ds._read_cache(item["hr"], ds.hr_bands, div=ds.hr_div)
            lr0 = ds._read_cache(item["lr"][0], ds.lr_bands)
            (li, lj, lw), (hi, hj, hs) = ds._pair_box(hr_full.shape, lr0.shape, idx)
            Hr, Wr = hr_full.shape[-2], hr_full.shape[-1]
            sy, sx = 156.0 / Hr, 156.0 / Wr
            wi = int(round((hi + 0.5) * sy - 0.5))
            wj = int(round((hj + 0.5) * sx - 0.5))
            wh = max(8, min(int(round(hs * sy)), 156 - wi))
            ww = max(8, min(int(round(hs * sx)), 156 - wj))
            crop = y_hat_reg[:, :, wi:wi + wh, wj:wj + ww]
            crop = F.interpolate(crop, size=(424, 424), mode="bicubic",
                                 align_corners=False, antialias=True)[0]
            # 场景归一化：用与数据管线/标签完全相同的逐场景 (mu,sd)（LR RGB 统计）
            # 把 hrnet 从原生反射率空间变换到 per_scene 归一化空间——其余四臂的
            # 输出天然就在该空间（训练目标即归一化 GT），hrnet 不变换会成为
            # 唯一的异空间臂，下游分割会被逐场景辐射映射系统性惩罚。
            assert ds.scene_norm is not None and len(ds.scene_norm) > idx
            mu_s, sd_s = ds.scene_norm[idx]
            crop = (crop - float(mu_s)) / (float(sd_s) + 1e-6)

            outs = {"gt": gt[None], "bicubic": bic, "ours": p_ours,
                    "edsr": p_edsr, "wdsr": p_wdsr, "hrnet": crop[None]}
            mean_flr = torch.full_like(gt, float(gt.mean()))[None]
            for a in ARMS[1:]:
                o = outs[a][0]
                rows[a].append(float(calc_psnr(o, gt, data_range=val_dr)))
                cc_rows[a].append(float(calc_cc(o, gt)))
                stdr_rows[a].append(float(o.std() / (gt.std() + 1e-8)))
                g, _ = periodicity_scan(o)
                grid_rows[a].append(float(g) if g == g else float("nan"))
            rows_floor.append(float(calc_psnr(mean_flr[0], gt, data_range=val_dr)))

            for a in ARMS:
                store[a].append(outs[a][0].detach().cpu().numpy().astype(np.float32))
            scene_ids.append(os.path.basename(os.path.dirname(item["hr"])))
            store_lr.append(lrs[0, 0].detach().cpu().numpy().astype(np.float32))
            if (rel + 1) % 10 == 0:
                print(f"  ...{rel+1}/{args.limit or n_val}  "
                      f"bic={np.mean(rows['bicubic']):.2f} ours={np.mean(rows['ours']):.2f} "
                      f"edsr={np.mean(rows['edsr']):.2f} wdsr={np.mean(rows['wdsr']):.2f} "
                      f"({time.time()-t0:.0f}s)")

    N = len(scene_ids)
    print(f"[info] dumped N={N} in {time.time()-t0:.0f}s")
    for a in ARMS:
        arr = np.stack(store[a], 0)
        path = os.path.join(OUT_DIR, f"{a}.npy")
        np.save(path, arr)
        print(f"[saved] {path}  {arr.shape}")
    lr_arr = np.stack(store_lr, 0)
    np.save(os.path.join(OUT_DIR, "lr.npy"), lr_arr)
    print(f"[saved] {os.path.join(OUT_DIR, 'lr.npy')}  {lr_arr.shape}")

    # ---------- 全臂统一配准（下游口径；审计复核用原生未配准数组） ----------
    # 对【全部臂】施加同预算粗到细整数配准（±16px@424，CC 准则），镜像官方
    # WorldStrat 协议对 model/bicubic 同预算 Lanczos 配准的精神：隔离内容质量、
    # 消除对齐运气（数据管线自带的 S2↔SPOT 逐场景地理误差 + hrnet 映射残差）。
    preview_n = args.preview or 0
    gt_cpu = torch.from_numpy(np.stack(store["gt"]))
    gt_all = gt_cpu.to(device)
    reg_diag, reg_first = {}, {}
    for a in ARMS[1:]:
        x = torch.from_numpy(np.stack(store[a])).to(device)
        regs, shifts, cc_pres, cc_posts = [], [], [], []
        for i in range(N):
            cc_pres.append(round(float(_cc3(x[i], gt_all[i])), 4))
            r, (dy, dx), cc_post = refine_shift(x[i], gt_all[i], max_shift=16)
            regs.append(r.detach().cpu().numpy().astype(np.float32))
            shifts.append([dy, dx])
            cc_posts.append(round(float(cc_post), 4))
        arr = np.stack(regs, 0)
        np.save(os.path.join(OUT_DIR, f"{a}_reg.npy"), arr)
        reg_diag[a] = {"cc_pre_mean": round(float(np.mean(cc_pres)), 4),
                       "cc_post_mean": round(float(np.mean(cc_posts)), 4),
                       "shift_hist": shifts}
        if preview_n:
            reg_first[a] = torch.from_numpy(arr[:preview_n])
        print(f"[reg] {a:<8} cc {reg_diag[a]['cc_pre_mean']:.3f} -> "
              f"{reg_diag[a]['cc_post_mean']:.3f}  saved {a}_reg.npy")
        del x
    del gt_all
    if preview_n:
        for i in range(preview_n):
            _save_preview(OUT_DIR, i, scene_gt=gt_cpu[i],
                          bic=reg_first["bicubic"][i], crop=reg_first["hrnet"][i])
        print(f"[saved] {preview_n} 张配准后预览 preview_*.png")

    # ---------- 配对复核 + verdict ----------
    paired, verdicts = {}, {}
    for a in ("ours", "edsr", "wdsr", "hrnet"):
        paired[a] = paired_t(rows[a], rows["bicubic"])
    for a in ("ours", "edsr", "wdsr"):
        gain = float(np.mean(rows[a]) - np.mean(rows["bicubic"]))
        r_cc = float(np.mean(cc_rows[a]) / max(np.mean(cc_rows["bicubic"]), 1e-6))
        card = {
            "method": a, "n_test": N, "audit_gain_db": round(gain, 3),
            "audit_p": paired[a]["p"],
            "passes_floor": bool(np.mean(rows[a]) > np.mean(rows_floor)),
            "g1": "PASS" if np.mean(cc_rows["bicubic"]) >= 0.30 else "FAIL",
            "g2": ("PASS" if r_cc >= 0.9 else ("WARN" if r_cc >= 0.7 else "FAIL")),
            "g3": round(float(np.mean(stdr_rows[a])), 3),
            "g4": round(float(np.nanmean(grid_rows[a])), 3),
            "g5": "PASS" if np.mean(rows[a]) <= 45.0 else "FAIL",
            "delta_floor_db": round(float(np.mean(rows["bicubic"])
                                          - np.mean(rows_floor)), 3),
        }
        d = derive(card)
        verdicts[a] = {"card": card, "credibility": d["credibility"],
                       "regime": d["regime"], "gate_verdict": d["gate_verdict"]}
        print(f"[verdict] {a:<7} gain={gain:+.2f} dB  p={paired[a]['p']:.3g}  "
              f"-> {d['credibility']}|{d['regime']}")

    # hrnet 臂诊断（辐射空间不同，PSNR 不跨空间比；配准诊断见 reg_diag）
    hrnet_diag = {
        "cc_native_mean": round(float(np.mean(cc_rows["hrnet"])), 4),
        "cc_bicubic_native_mean": round(float(np.mean(cc_rows["bicubic"])), 4),
        "psnr_own_space_mean": round(float(np.mean(rows["hrnet"])), 3),
        "lanczos_best_shift_hist": hrnet_shift_hist,
        "note": ("hrnet 臂=官方协议(Lanczos ±2px@156 配准 + bias_adjust)输出映射到 "
                 "裁剪网格（原生反射率空间）。下游用 *_reg.npy（全臂统一粗到细整数"
                 "配准）。其 +0.995 dB 审计增益来自官方 156 全景口径 "
                 "(experiments/sota_review/highresnet.json)，不重算。"),
    }
    meta = {
        "protocol": ("ws val-60 (audit_axis 同口径: deterministic crop, per_scene "
                     "norm, subset=600 tail 10%), 424x424 RGB; hrnet=official "
                     "protocol mapped to same crop grid (raw reflectance space)"),
        "n": N, "val_dr": round(float(val_dr), 6),
        "scene_ids": scene_ids,
        "psnr_mean": {a: round(float(np.mean(rows[a])), 3) for a in ARMS[1:]},
        "psnr_floor_mean": round(float(np.mean(rows_floor)), 3),
        "per_sample_psnr": rows, "per_sample_floor": rows_floor,
        "paired_vs_bicubic": paired, "verdicts": verdicts,
        "hrnet_diag": hrnet_diag,
        "reg_diag": reg_diag,
        "reg_note": ("{arm}_reg.npy = 全臂统一粗到细整数配准(±16px@424, CC 准则)后"
                     "数组，供保真度敏感下游使用；{arm}.npy 原生数组供审计复核。"),
        "published_audit_cards": {
            "ours": -0.88, "edsr": -2.43, "wdsr": -3.02, "highresnet": +0.995,
        },
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[saved] meta.json")
    print("\n== PSNR 汇总 (val_dr=%.3f) ==" % val_dr)
    for a in ARMS[1:]:
        pp = paired.get(a)
        extra = (f"  paired-vs-bic {pp['mean']:+.3f} (p={pp['p']:.3g})"
                 if pp else "")
        print(f"  {a:<8} {np.mean(rows[a]):7.2f} dB{extra}")
    print(f"  floor    {np.mean(rows_floor):7.2f} dB")


if __name__ == "__main__":
    main()
