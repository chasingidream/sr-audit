"""P2-3: 把 SR-Audit 协议接到任意外部 SR 方法的输出 (LR, SR, GT)。

目标（见 docs/P2_下游验证与通用性规划.md）：证明 SR-Audit 不挑方法身份——给任意
外部方法的 (低分辨率输入 LR, 其超分输出 SR, 高分辨率真值 GT)，本脚本机械地复算
G1–G5 五道闸门 + Δ_floor + 配对 model−bicubic 增益/显著性，产出一个 audit card，
再喂给 code/sr_audit_report.py 的 derive() 得到可信徽章（VERIFIED/QUALIFIED/VOID/
UNVERIFIED）与科学 regime（positive/saturated/floor-failure）。

设计要点（口径一致性）：
- G1/G4 闸门逻辑**逐字移植**自 code/sr_audit.py 的 periodicity_scan / _harmonic_ratio，
  只把 torch.fft 换成 numpy.fft（阈值、邻近频率基线法、period≈2/scale 诊断全部一致）。
- 双三次上采样用 scipy zoom(order=3) 作为 code/sr_audit.py bicubic_up
  (torch F.interpolate bicubic, align_corners=False) 的可移植等价实现；两者 CC/周期
  量级的差异 <1e-3，闸门判定不变。若需与 sr_audit.py 逐位一致，可把 _BICUBIC_USE_TORCH
  开关打开（会 import torch，需要本机 OpenMP 守卫处理）。
- 全程 numpy + scipy，**无需 GPU / 无需 torch**，因此本框架可在任意机器上跑冒烟测试，
  真实外部权重接入时再在 GPU 上产出 SR 即可。

用法：
  # 对真实外部方法输出做审计（每个 .npy 形如 [N,C,H,W] 或 [N,H,W]）
  python code/p2_audit_external.py \
      --lr lr.npy --sr sr_methodX.npy --gt gt.npy \
      --name "MethodX (re-impl)" --task "cross-sensor 6.63x" --split val \
      --out experiments/p2_external_audit.json --report experiments/p2_external_audit.md

  # 合成数据冒烟测试（无需真实权重）：跑 positive / floor-failure / G1-FAIL 三场景
  python code/p2_audit_external.py --demo --out experiments/p2_external_audit.json --report experiments/p2_external_audit.md
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import zoom
from scipy import stats

# 复用 SR-Audit 报告标准（derive 出徽章）。把 code/ 加入路径。
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sr_audit_report import derive  # noqa: E402

# 注意：本脚本刻意**不** import eval_metrics，因为 eval_metrics 顶层会 import torch，
# 在本机沙箱会触发 OpenMP 双运行时 segfault。CC 用下方纯 numpy 实现（逐波段 Pearson，
# 与 eval_metrics.calc_cc 语义一致：逐波段相关系数均值）。
_BICUBIC_USE_TORCH = True  # True：双三次基线用 torch F.interpolate(align_corners=False)，
# 与 sr_audit.bicubic_up / 主文评测口径逐位一致（G1 闸门阈值 0.30 即按此基线标定）。
# 设为 False 则退回 scipy zoom(order=3) 可移植实现，但 bicubic 基线 CC 会低 ~0.003，
# 在 0.30 边界场景（如本 6.63× 跨传感器臂）可能误触发 G1 FAIL。


# ---------------------------------------------------------------- 闸门原语（移植自 sr_audit.py）
def bicubic_up(lr, size):
    """LR (B,C,h,w) -> size=(H,W) 的双三次上采样。

    默认 scipy zoom(order=3)；若 _BICUBIC_USE_TORCH 则改用 torch F.interpolate
    bicubic align_corners=False，与 sr_audit.py 逐位一致。
    """
    if _BICUBIC_USE_TORCH:
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(lr).float()
        return F.interpolate(t, size=size, mode="bicubic",
                             align_corners=False).numpy()
    B, C, h, w = lr.shape
    fx, fy = size[0] / h, size[1] / w
    out = np.empty((B, C, size[0], size[1]), dtype=lr.dtype)
    for b in range(B):
        for c in range(C):
            out[b, c] = zoom(lr[b, c], (fx, fy), order=3)
    return out


def _harmonic_ratio(mag, period):
    """给定周期 p，返回该周期对应频率线能量 / 邻近频率基线（numpy 移植版）。"""
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    fy, fx = int(round(H / period)), int(round(W / period))
    if fy < 2 or fx < 2 or fy >= cy or fx >= cx:
        return float("nan")
    off = max(2, fy // 8)
    rs = []
    for r in (cy + fy, cy - fy):
        if 0 <= r - off and r + off < H:
            base = np.stack([mag[r - off, :].mean(), mag[r + off, :].mean()])
            rs.append(mag[r, :].mean() / (base.mean() + 1e-12))
    for c in (cx + fx, cx - fx):
        if 0 <= c - off and c + off < W:
            base = np.stack([mag[:, c - off].mean(), mag[:, c + off].mean()])
            rs.append(mag[:, c].mean() / (base.mean() + 1e-12))
    if not rs:
        return float("nan")
    return float(np.stack(rs).mean())


def periodicity_scan(img, periods=range(2, 17)):
    """周期伪影扫描（numpy 移植版，阈值与 sr_audit.py 完全一致）。"""
    x = img.astype(np.float64).mean(0)
    mag = np.fft.fftshift(np.abs(np.fft.fft2(x)))
    best, bp = -1.0, None
    for p in periods:
        r = _harmonic_ratio(mag, p)
        if r == r and r > best:      # 跳过 nan
            best, bp = r, p
    return float(best), (int(bp) if bp else -1)


# ---------------------------------------------------------------- 指标辅助
def _to_bchw(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:                 # (H,W) -> (1,1,H,W)
        return x[None, None]
    if x.ndim == 3:                 # (C,H,W) or (N,H,W)
        if x.shape[0] in (1, 3):    # 视为 (C,H,W)
            return x[None]
        return x[:, None]            # (N,H,W) -> (N,1,H,W)
    return x                         # 已是 (N,C,H,W)


def psnr_per_sample(pred, gt):
    """逐样本 PSNR（避免整批池化伪增益）。pred/gt: (N,C,H,W)。"""
    dr = gt.max(axis=(1, 2, 3)) - gt.min(axis=(1, 2, 3))
    dr = np.where(dr <= 0, 1.0, dr)
    mse = ((pred - gt) ** 2).mean(axis=(1, 2, 3))
    out = np.where(mse < 1e-8, 99.0, 10.0 * np.log10((dr ** 2) / mse))
    return out


def cc_pair(pred, gt):
    """逐场景 CC（逐波段 Pearson 相关均值），与 eval_metrics.calc_cc 逐位一致：
    对每个场景，把该场景全部像素展平为一条序列（逐波段），减均值后算单一 Pearson
    相关，再对场景取均值、对波段取均值。这与主文审计 `calc_cc(o, gt)` 逐场景调用的
    语义完全相同（calc_cc 在 (C,H,W) 输入下会 [None] 升维后跨整图求相关）。
    pred/gt: (N,C,H,W)。纯 numpy，避免触发 torch import。
    """
    p = pred.reshape(pred.shape[0], pred.shape[1], -1)
    g = gt.reshape(gt.shape[0], gt.shape[1], -1)
    cs = []
    for b in range(p.shape[1]):
        vals = []
        for i in range(p.shape[0]):
            a = p[i, b].reshape(-1)
            c = g[i, b].reshape(-1)
            a = a - a.mean()
            c = c - c.mean()
            denom = np.sqrt((a ** 2).sum() * (c ** 2).sum())
            vals.append(0.0 if denom < 1e-8 else float((a * c).sum() / denom))
        cs.append(float(np.mean(vals)))
    return float(np.mean(cs))


def std_ratio(pred, gt):
    """std(SR) / std(GT) 逐样本平均（G3 闸门量）。"""
    sp = pred.std(axis=(1, 2, 3))
    sg = gt.std(axis=(1, 2, 3))
    return float(np.mean(sp / np.where(sg < 1e-8, 1e-8, sg)))


# ---------------------------------------------------------------- 核心：审计一张卡
def audit_card(lr, sr, gt, method, task=None, split=None, leak_method_names=()):
    """对 (LR, SR, GT) 复算 SR-Audit 全卡，返回可直接喂 derive() 的 card dict。"""
    lr = _to_bchw(lr)
    sr = _to_bchw(sr)
    gt = _to_bchw(gt)
    B = gt.shape[0]
    size = (gt.shape[2], gt.shape[3])

    # 双三次基线（与 G1/G4 口径一致）
    bic = bicubic_up(lr, size)
    mean_img = gt.mean(axis=(1, 2, 3), keepdims=True) * np.ones_like(gt)

    psnr_bic = psnr_per_sample(bic, gt)
    psnr_sr = psnr_per_sample(sr, gt)
    psnr_mean = psnr_per_sample(mean_img, gt)

    # Δ_floor（数据属性，对全局线性缩放不变）：逐样本再取均值
    delta_floor_per = psnr_bic - psnr_mean
    delta_floor = float(np.mean(delta_floor_per))

    # 配对 model−bicubic 增益
    gain_per = psnr_sr - psnr_bic
    gain = float(np.mean(gain_per))
    sd = float(np.std(gain_per, ddof=1))
    n = len(gain_per)
    if n > 1 and sd > 0:
        t_stat, p = stats.ttest_rel(psnr_sr, psnr_bic)
        if np.isnan(t_stat):
            p = 1.0
        ci = stats.t.interval(0.95, df=n - 1, loc=gain, scale=sd / np.sqrt(n))
        d = gain / sd
    else:
        p = 1.0
        ci = (gain, gain)
        d = 0.0

    # CC 闸门
    cc_bic = cc_pair(bic, gt)
    cc_sr = cc_pair(sr, gt)
    g1 = "PASS" if cc_bic >= 0.30 else "FAIL"
    ratio_cc = cc_sr / max(cc_bic, 1e-6)
    g2 = ("PASS" if ratio_cc >= 0.9 else ("WARN" if ratio_cc >= 0.7 else "FAIL"))

    # G3 输出统计
    g3 = float(std_ratio(sr, gt))

    # G4 周期伪影（取 batch 中位数 period，mean ratio）
    grid = [periodicity_scan(sr[i]) for i in range(B)]
    g4 = float(np.nanmean([r for r, _ in grid]))

    # G5 泄漏风险：PSNR>45（可疑）→ FAIL；或方法名命中已知泄漏档
    leak = (float(np.mean(psnr_sr)) > 45.0) or any(
        nm.lower() in method.lower() for nm in leak_method_names)
    g5 = "FAIL" if leak else "PASS"

    # 是否越过 mean-predictor 技能地板
    floor_gap = gain + delta_floor              # = PSNR(model) - PSNR(mean)
    passes_floor = bool(floor_gap > 0)

    card = {
        "method": method,
        "task": task,
        "n_test": n,
        "split": split,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "g5": g5,
        "delta_floor_db": round(delta_floor, 3),
        "passes_floor": passes_floor,
        "floor_gap_db": round(floor_gap, 3),
        "audit_gain_db": round(gain, 3),
        "audit_p": float(p),
        "audit_ci": [round(ci[0], 3), round(ci[1], 3)],
        "audit_d": round(float(d), 3),
        "_detail": {
            "psnr_bicubic_mean": round(float(np.mean(psnr_bic)), 3),
            "psnr_sr_mean": round(float(np.mean(psnr_sr)), 3),
            "psnr_mean_mean": round(float(np.mean(psnr_mean)), 3),
            "cc_bicubic": round(cc_bic, 4),
            "cc_sr": round(cc_sr, 4),
            "g2_ratio": round(ratio_cc, 3),
            "g3_std_ratio": round(g3, 3),
            "g4_grid_ratio": round(g4, 3),
            "g4_period": int(np.median([pp for _, pp in grid])),
        },
    }
    # 派生徽章
    rec = derive(card)
    card["_derived"] = rec
    return card


# ---------------------------------------------------------------- 合成冒烟数据
def _textured_field(rng, n, c, res):
    """纹理随机场 [N,C,res,res]：低频基 + 多个中高频正弦 + 轻噪声。

    用纹理场而非平滑场，是为了：(a) Δ_floor 落在真实量级（bicubic 丢高频 →
    仅比 mean-predictor 好几个 dB，而非几十 dB）；(b) FFT 频谱铺平 → periodicity
    闸门 G4 不会因单一低频能量而被误判 FAIL。
    """
    base = rng.standard_normal((n, c, res // 8, res // 8))
    out = np.empty((n, c, res, res))
    for i in range(n):
        for k in range(c):
            out[i, k] = zoom(base[i, k], (8, 8), order=3)
    yy, xx = np.mgrid[0:res, 0:res]
    for i in range(n):
        for k in range(c):
            for period in (3, 5, 7, 11):
                amp = rng.uniform(0.05, 0.12)
                out[i, k] += amp * np.sin(2 * np.pi * xx / period + rng.uniform(0, 6.28))
                out[i, k] += amp * np.sin(2 * np.pi * yy / period + rng.uniform(0, 6.28))
    out += rng.standard_normal((n, c, res, res)) * 0.05
    out = (out - out.min()) / (out.max() - out.min() + 1e-12)
    return out


def _synthetic(mode="positive", gain_db=2.0, n=40, seed=0):
    """构造 (lr, sr, gt)。

    - mode='positive'  : 真实配对 LR + SR 比 bicubic 好 gain_db（混合向 GT 逼近）→
                         VERIFIED/positive。
    - mode='saturated' : SR = bicubic + 零均值噪声，增益≈0 且配对 t 检验不显著 →
                         saturated（无判别余量）。
    - mode='g1-fail'   : LR 与 GT 解相关（触发 G1 FAIL → VOID，且 Δ_floor<0 落入
                         floor-failure 语义）→ 演示"配对断裂即作废"。
    """
    rng = np.random.default_rng(seed)
    gt = _textured_field(rng, n, 3, 64)
    if mode == "g1-fail":
        lr = _textured_field(rng, n, 3, 16)          # 与 GT 无关
    else:
        lr = np.empty((n, 3, 16, 16))
        for i in range(n):
            for k in range(3):
                lr[i, k] = zoom(gt[i, k], (0.25, 0.25), order=1)
    bic = bicubic_up(lr, (64, 64))

    if mode == "saturated":
        # SR == bicubic（逐样本完全一致）：增益=0，配对 t 检验不显著 -> saturated。
        # 注意：叠加零均值噪声反而会因 MSE 一致增大而给出显著负增益，故此处取恒等。
        sr = bic.copy()
    else:
        # 线性混合精确实现 PSNR(sr,gt) = PSNR(bic,gt) + gain_db
        alpha = 1.0 - 10 ** (-gain_db / 20.0)
        sr = alpha * gt + (1.0 - alpha) * bic
        sr = sr + rng.standard_normal(gt.shape) * 1e-3   # 去除退化零方差
    return lr, sr, gt


def _run_demo():
    scenarios = [
        ("positive",  dict(mode="positive",  gain_db=2.0, n=40, seed=1),
         "Demo-Positive (gain +2 dB)"),
        ("saturated", dict(mode="saturated", n=40, seed=2),
         "Demo-Saturated (gain~0, not significant)"),
        ("g1-fail",   dict(mode="g1-fail",   n=40, seed=3),
         "Demo-G1FAIL (decorrelated LR)"),
    ]
    cards = []
    for tag, kw, name in scenarios:
        lr, sr, gt = _synthetic(**kw)
        c = audit_card(lr, sr, gt, method=name, task="synthetic smoke",
                       split="demo", leak_method_names=("leak",))
        cards.append(c)
        rec = c["_derived"]
        print("[demo:%s] %s  | Δ_floor=%.2f pf=%s g12345=%s/%s/%s/%s/%s"
              % (tag, rec["headline"], rec["delta_floor_db"], rec["passes_floor"],
                 rec["gates"]["g1"], rec["gates"]["g2"], rec["gates"]["g3"],
                 rec["gates"]["g4"], rec["gates"]["g5"]))
    return cards


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="P2-3: SR-Audit on external method outputs")
    ap.add_argument("--lr", default=None, help="LR .npy [N,C,H,W] or [N,H,W]")
    ap.add_argument("--sr", default=None, help="external SR output .npy")
    ap.add_argument("--gt", default=None, help="ground-truth HR .npy")
    ap.add_argument("--name", default="ExternalMethod", help="method name for the card")
    ap.add_argument("--task", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--leak-names", default="", help="comma-separated substrings that imply leakage")
    ap.add_argument("--demo", action="store_true", help="run synthetic smoke test (3 scenarios)")
    ap.add_argument("--out", default="experiments/p2_external_audit.json")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.demo:
        cards = _run_demo()
    else:
        if not (args.lr and args.sr and args.gt):
            print("[error] provide --lr/--sr/--gt, or use --demo", file=sys.stderr)
            return 2
        lr = np.load(args.lr)
        sr = np.load(args.sr)
        gt = np.load(args.gt)
        leak = [s.strip() for s in args.leak_names.split(",") if s.strip()]
        c = audit_card(lr, sr, gt, method=args.name, task=args.task,
                       split=args.split, leak_method_names=leak)
        rec = c["_derived"]
        print(rec["headline"])
        for r in rec["reasons"]:
            print("  - " + r)
        cards = [c]

    # 落盘：剥离 _derived 里已是 derive() 完整记录，保存原始 card + 派生结果
    out_obj = {"standard": "SR-Audit Report Standard v1 (P2-3 external)",
               "n_cards": len(cards),
               "cards": []}
    for c in cards:
        entry = {k: v for k, v in c.items() if k != "_detail"}
        entry["derived"] = c["_derived"]
        entry["detail"] = c.get("_detail")
        out_obj["cards"].append(entry)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print("[saved] %s" % args.out)

    if args.report:
        L = ["# P2-3 External-Method SR-Audit", "",
             "| method | regime | badge | gain(dB) | p | Δ_floor(dB) | gates |",
             "|---|---|---|---|---|---|---|"]
        for c in cards:
            rec = c["_derived"]
            G = ("g1:%s g2:%s g3:%s g4:%s g5:%s" % (
                rec["gates"]["g1"], rec["gates"]["g2"], rec["gates"]["g3"],
                rec["gates"]["g4"], rec["gates"]["g5"]))
            gain = "n/a" if rec["audit_gain_db"] is None else "%+.2f" % rec["audit_gain_db"]
            df = "n/a" if rec["delta_floor_db"] is None else "%+.2f" % rec["delta_floor_db"]
            p = "n/a" if rec["audit_p"] is None else "%.3g" % rec["audit_p"]
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
                rec["method"], rec["regime"], rec["credibility"], gain, p, df, G))
        L += ["", "---", ""]
        for c in cards:
            rec = c["_derived"]
            L.append("### %s  [%s]" % (rec["method"], rec["credibility"]))
            L.append("- regime: **%s**" % rec["regime"])
            if rec.get("audit_gain_db") is not None:
                L.append("- gain(model-bicubic): **%+.2f dB** (p=%.3g, d=%.2f)"
                         % (rec["audit_gain_db"], rec["audit_p"], rec["audit_d"]))
            if rec.get("delta_floor_db") is not None:
                L.append("- Δ_floor: %+.2f dB" % rec["delta_floor_db"])
            L.append("- gates: g1=%s g2=%s g3=%s g4=%s g5=%s" % (
                rec["gates"]["g1"], rec["gates"]["g2"], rec["gates"]["g3"],
                rec["gates"]["g4"], rec["gates"]["g5"]))
            if rec.get("reasons"):
                L.append("- verdict: " + "; ".join(rec["reasons"]))
            L.append("")
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print("[saved] %s" % args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
