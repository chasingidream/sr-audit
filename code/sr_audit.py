"""sr_audit.py —— 遥感超分评测审计（统一入口，Nature 子刊方案 P0 交付物）。

把此前零散的诊断脚本合并成"一键审计"，把项目两年踩过的坑固化为可复用协议：
    1) 配对完整性  CC(bicubic(LR), GT)  —— 伪配对时 ~0.03，正确时 ≳0.3
    2) 技能地板    mean / bicubic / model / GT 四预测器对比
    3) 输出统计    |bias|（均值偏移）与 std_ratio（离群/压缩）
    4) 周期伪影    FFT 网格能量比 grid_ratio（上采样器棋盘/网格的定量指标）
    5) 泄漏风险    HR 是否进网络 + PSNR 是否高得不合理

用法：
    python code/sr_audit.py --stage mtxa_mis_xa_rbc_zi
    python code/sr_audit.py --stage sisr --subset 3000 --val_frac 0.9 --report experiments/audit_sisr.md

输出：JSON（--save）+ Markdown 报告（--report）+ 控制台摘要，含 PASS/WARN/FAIL 门控。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paired_mtxa import load_val                      # noqa: E402
from models.sr_model import build_model                    # noqa: E402
from eval_metrics import calc_psnr, calc_ssim, calc_cc, calc_sam  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 泄漏风险档：HR 直接进网络重建（本项目已证实 mt 档虚增 +11.09dB）
LEAK_STAGES = ("mt", "mt+rsa", "end2end", "full",
               "xs_aligned", "xs_misaligned_noxa", "xs_misaligned_xa")


def bicubic_up(lr, size):
    """LR (B,3,h,w) -> size 的双三次上采样。

    align_corners=False 与主文评测路径 (`eval_paired_floor.py` / `eval_skill_floor_cfg.py`)
    保持一致；此前用 True，使 G1 的 CC 偏低 0.004（0.299 vs 0.303），在 0.30 阈值上造成
    伪"边缘/FAIL"。两条路径必须同口径，否则门判定不可比。
    """
    return F.interpolate(lr, size=size, mode="bicubic", align_corners=False)


def _harmonic_ratio(mag, period):
    """给定周期 p（像素），返回该周期对应频率线能量 / 邻近频率基线。"""
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    fy, fx = int(round(H / period)), int(round(W / period))
    if fy < 2 or fx < 2 or fy >= cy or fx >= cx:
        return float("nan")
    off = max(2, fy // 8)
    rs = []
    for r in (cy + fy, cy - fy):
        if 0 <= r - off and r + off < H:
            base = torch.stack([mag[r - off, :].mean(), mag[r + off, :].mean()])
            rs.append(mag[r, :].mean() / (base.mean() + 1e-12))
    for c in (cx + fx, cx - fx):
        if 0 <= c - off and c + off < W:
            base = torch.stack([mag[:, c - off].mean(), mag[:, c + off].mean()])
            rs.append(mag[:, c].mean() / (base.mean() + 1e-12))
    if not rs:
        return float("nan")
    return float(torch.stack(rs).mean())


def periodicity_scan(img, periods=range(2, 17)):
    """周期伪影扫描：在 2..16 px 周期里找最强的谐波突出度。

    返回 (best_ratio, best_period)。用"邻近频率基线"作分母（FFT 能量重尾，
    全局中位数过小会让所有图像都得到巨大比值，没有区分度）。
    主导周期本身是诊断信息：
      period≈2  -> 经典 PixelShuffle 棋盘；period≈scale -> 上采样块网格。
    经验阈值：ratio >2.5 视为存在周期伪影。
    """
    x = img.float().mean(0)
    mag = torch.fft.fftshift(torch.abs(torch.fft.fft2(x)))
    best, bp = -1.0, None
    for p in periods:
        r = _harmonic_ratio(mag, p)
        if r == r and r > best:      # 跳过 nan
            best, bp = r, p
    return float(best), (int(bp) if bp else -1)


def base_stage(stage):
    """剥离变体/多种子后缀得到基础 stage 名（与 train.py 门控一致）。"""
    import re
    s = stage
    _ch = True
    while _ch:
        _ch = False
        for x in ("_l1", "_ct", "_rbc", "_zi"):
            if s.endswith(x):
                s = s[:-len(x)]
                _ch = True
                break
        if re.search(r"_s\d+$", s):
            s = re.sub(r"_s\d+$", "", s)
            _ch = True
    return s


def audit(stage, device="cuda", subset=None, val_frac=0.9, max_samples=0,
          n_vis=0, ckpt_dir=None):
    ck, cfg, loader, val_dr = load_val(stage, device=device, subset=subset,
                                       val_frac=val_frac, ckpt_dir=ckpt_dir)
    scale = cfg.get("scale", 4)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    # 门控必须与 eval_paired_mtxa / diag_skill_floor 一致，否则未训练的
    # guidance 卷积会被当作噪声注入（use_ref 默认取 cfg 的 True → 污染评测）。
    _base = base_stage(stage)
    use_rsa = not _base.endswith("noxa")
    use_ref = False

    acc = {k: [] for k in ("mean", "bicubic", "model")}
    pair_cc, bias_list, stdr_list, grid_m, grid_b = [], [], [], [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            lrs = batch["lrs"].float().to(device)
            tgt = batch["target"].float().to(device)
            lrs_in = lrs[:, :1] if _base.startswith("sisr") else lrs
            pred = model(lrs_in, batch["hr_ref"].float().to(device),
                         return_aux=False, use_rsa=use_rsa, use_ref=use_ref)
            if not torch.is_tensor(pred):
                pred = pred[0]
            if tuple(pred.shape[-2:]) != tuple(tgt.shape[-2:]):
                pred = F.interpolate(pred, size=tuple(tgt.shape[-2:]),
                                     mode="bilinear", align_corners=True)
            bic = bicubic_up(lrs[:, 0], size=tuple(tgt.shape[-2:])).to(device)
            gt_mean = float(tgt.mean())
            mnc = torch.full_like(tgt, gt_mean)

            # clamp(0,1) 仅当数据本身已在 [0,1]（val_dr≈1）时才成立。
            # per-scene 归一化配置（如 WorldStrat，val_dr≈15）的数据在 [0,1] 之外，
            # 对预测做 clamp 而 GT 不 clamp 会造成灾难性口径错配
            # （std_ratio→~0.1、bias→1.0、CC 崩塌），使 G1/G2/G3 假 FAIL。
            _do_clamp = abs(val_dr - 1.0) < 1e-6

            def _c(x):
                return x.clamp(0, 1) if _do_clamp else x

            for name, out in (("mean", mnc), ("bicubic", bic), ("model", pred)):
                o = _c(out)
                acc[name].append({
                    "psnr": float(calc_psnr(o, tgt, data_range=val_dr)),
                    "ssim": float(calc_ssim(o, tgt, data_range=val_dr)),
                    "cc": float(calc_cc(o, tgt)),
                    "sam": float(calc_sam(o, tgt)),
                })
            pair_cc.append(float(calc_cc(_c(bic), tgt)))
            p = _c(pred)
            bias_list.append(abs(float(p.mean() - tgt.mean())))
            stdr_list.append(float((p.std(dim=(1, 2)).mean()) /
                                   (tgt.std(dim=(1, 2)).mean() + 1e-8)))
            r_m, p_m = periodicity_scan(p[0])
            r_b, p_b = periodicity_scan(bic[0].clamp(0, 1))
            grid_m.append((r_m, p_m))
            grid_b.append((r_b, p_b))
            n += 1
            if max_samples and n >= max_samples:
                break

    def agg(d):
        return {k: float(np.mean([x[k] for x in d])) for k in d[0]}

    res = {
        "stage": stage,
        "n_samples": n,
        "val_dr": float(val_dr),
        "predictors": {k: agg(v) for k, v in acc.items()},
        "pair_cc": float(np.mean(pair_cc)),
        "bias": float(np.mean(bias_list)),
        "std_ratio": float(np.mean(stdr_list)),
        "grid_ratio_model": float(np.nanmean([a for a, _ in grid_m])),
        "grid_ratio_bicubic": float(np.nanmean([a for a, _ in grid_b])),
        "grid_period_model": int(round(float(np.median([b for _, b in grid_m])))),
        "grid_period_bicubic": int(round(float(np.median([b for _, b in grid_b])))),
    }

    # ---------------- 门控 ----------------
    g = {}
    g["G1_配对完整性"] = ("PASS" if res["pair_cc"] >= 0.30 else "FAIL",
                          f"CC(bic,GT)={res['pair_cc']:.3f} (阈值≥0.30)")
    r_cc = res["predictors"]["model"]["cc"] / max(res["predictors"]["bicubic"]["cc"], 1e-6)
    g["G2_结构保真"] = ("PASS" if r_cc >= 0.9 else ("WARN" if r_cc >= 0.7 else "FAIL"),
                        f"model CC / bicubic CC = {r_cc:.2f}")
    g["G3_输出统计"] = ("PASS" if 0.5 <= res["std_ratio"] <= 1.8 else "FAIL",
                        f"std_ratio={res['std_ratio']:.2f}, |bias|={res['bias']:.4f}")
    g["G4_周期伪影"] = ("PASS" if res["grid_ratio_model"] < 2.5 else "WARN",
                        f"ratio={res['grid_ratio_model']:.2f} @ period="
                        f"{res['grid_period_model']}px (bicubic "
                        f"{res['grid_ratio_bicubic']:.2f} @ "
                        f"{res['grid_period_bicubic']}px)")
    leak = stage in LEAK_STAGES or res["predictors"]["model"]["psnr"] > 45.0
    g["G5_泄漏风险"] = ("FAIL" if leak else "PASS",
                        "HR 参与重建或 PSNR>45dB" if leak else "未见泄漏")
    res["gates"] = {k: {"verdict": v[0], "detail": v[1]} for k, v in g.items()}
    res["overall"] = "FAIL" if any(v[0] == "FAIL" for v in g.values()) else (
        "WARN" if any(v[0] == "WARN" for v in g.values()) else "PASS")
    return res


def to_markdown(res):
    p = res["predictors"]
    L = [f"# SR 审计报告 — `{res['stage']}`",
         "", f"- 样本数：{res['n_samples']}　val_dr：{res['val_dr']:.4f}",
         f"- **总判定：{res['overall']}**", "",
         "## 1. 四预测器技能地板",
         "| 预测器 | PSNR | SSIM | CC | SAM° |",
         "|---|---|---|---|---|"]
    for k in ("mean", "bicubic", "model"):
        d = p[k]
        L.append(f"| {k} | {d['psnr']:.2f} | {d['ssim']:.4f} | {d['cc']:+.4f} | {d['sam']:.3f} |")
    L += ["", "## 2. 审计门控", "| 门 | 判定 | 依据 |", "|---|---|---|"]
    for k, v in res["gates"].items():
        L.append(f"| {k} | **{v['verdict']}** | {v['detail']} |")
    L += ["", "## 3. 原始量",
          f"- 配对 CC(bic, GT) = {res['pair_cc']:.4f}",
          f"- 输出 |bias| = {res['bias']:.4f}，std_ratio = {res['std_ratio']:.3f}",
          f"- 周期伪影：model ratio {res['grid_ratio_model']:.2f} @ "
          f"{res['grid_period_model']}px；bicubic {res['grid_ratio_bicubic']:.2f} @ "
          f"{res['grid_period_bicubic']}px（period≈2=棋盘，≈scale=块网格）"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--val_frac", type=float, default=0.9)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--save", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--ckpt_dir", default=None,
                    help="ckpt 根目录；默认 experiments/ckpts（旧 norm=None 跑）。"
                         "审计当前 per-scene 管线请传 experiments/ckpts_worldstrat")
    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    res = audit(args.stage, device=device, subset=args.subset,
                val_frac=args.val_frac, max_samples=args.max_samples,
                ckpt_dir=args.ckpt_dir)
    print(f"[audit] stage={res['stage']}  n={res['n_samples']}  "
          f"overall={res['overall']}")
    for k, v in res["gates"].items():
        print(f"  {v['verdict']:<5} {k:<14} {v['detail']}")
    p = res["predictors"]
    print(f"  PSNR: mean {p['mean']['psnr']:.2f} | bicubic {p['bicubic']['psnr']:.2f}"
          f" | model {p['model']['psnr']:.2f}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[saved] {args.save}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(to_markdown(res))
        print(f"[saved] {args.report}")


if __name__ == "__main__":
    main()
