"""
WorldStrat mtxa 三档逐样本配对显著性检验（修后管线专用）。

相对通用 eval_paired_mtxa.py 的修正（见 MEMORY.md "可恢复信息缺口" 与
"PSNR/聚合口径" 铁律）：
  * ckpt 根目录 = experiments/ckpts_worldstrat（external_worldstrat.py 的 save_dir）
  * 数据 norm="per_scene" -> 标准化到 unit-variance，无 clamp，允许负/超 1 值
    -> 评测时【不 clamp pred】（否则 PSNR 失真）；val_dr 取验证集 GT 全局范围
  * scale=7（out_size=424 对应 6.63× 真实 GSD 比）
  * 去掉已知 ndim bug 的 cc 指标，主用 psnr / ssim / sam_deg
  * 新增 bicubic 基线：对 frame0（干净锚点）bicubic 上采样到 GT 尺寸，作"无对齐"参照

配对设计（每个验证样本在完全相同的失配输入下跑三档 + bicubic）：
  (mis_xa − mis_noxa) : cross-attn 自对齐的干净贡献
  (mis_xa − aligned)  : 距"无失配"上界的残差（越小说明 cross-attn 救回越彻底）
  (mis_xa − bicubic)  : 相对"无对齐上采样"的方法增益
判据：配对 t 检验 p<0.05 且 95% CI 不跨 0 -> 差异显著。
"""
import argparse
import json
import math
import os
import sys
import glob
import atexit

# ---- vcomp140.dll guard (mirror of code/train.py) ----
# torch CUDA on this machine segfaults if both MSVC OpenMP (vcomp140.dll) and
# Intel OpenMP (libiomp5md) are loaded in one process. Hide vcomp140 before
# importing torch, restore on exit. Required because the preceding training
# step restores the DLL on clean exit; without this, `import torch` below
# would segfault.
_DLL = r"C:\Windows\System32\vcomp140.dll"
_HIDDEN = _DLL + ".hide"


def _eval_hide_dll():
    if os.path.exists(_HIDDEN) and not os.path.exists(_DLL):
        try:
            os.rename(_HIDDEN, _DLL)   # recover from a crashed run
        except Exception:
            pass
    if os.path.exists(_DLL):
        try:
            os.rename(_DLL, _HIDDEN)
            return True
        except Exception:
            print("[warn] could not hide vcomp140.dll; torch CUDA may segfault.")
    return False


def _eval_restore_dll():
    if os.path.exists(_HIDDEN):
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            print("[warn] could not restore vcomp140.dll; rename manually.")


_eval_hide_dll()
atexit.register(_eval_restore_dll)

import torch
import torch.nn.functional as F  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, os.path.join(BASE, "code", "models"))

from data.dataset import get_dataset  # noqa: E402
from models.sr_model import build_model  # noqa: E402
from eval_metrics import evaluate_all  # noqa: E402

try:
    from scipy import stats as _st
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

CKPT_ROOT = "experiments/ckpts_worldstrat"
STAGES = ["mtxa_aligned", "mtxa_mis_noxa", "mtxa_mis_xa"]
METRICS = ["psnr", "ssim", "sam_deg"]
SCALE = 7


def load_val(stage, device="cuda", subset=None, val_frac=0.9, manifest=None,
             scene_norm=None, eval_all=False, data_range=None):
    d = os.path.join(BASE, CKPT_ROOT, stage)
    fs = sorted(glob.glob(os.path.join(d, "ckpt_ep*.pt")),
                key=lambda p: int(p.split("_ep")[-1].split(".")[0]))
    if not fs:
        raise FileNotFoundError(f"no ckpt in {d}")
    ck = torch.load(fs[-1], map_location="cpu", weights_only=False)
    cfg = dict(ck.get("cfg", {}))
    ds = dict(cfg.get("dataset", {}))
    if subset:
        ds["subset"] = int(subset)
    if manifest:
        # 外部留出清单：整体替换 manifest，并剥掉 subset（否则会再被截断）
        ds["manifest"] = manifest
        ds.pop("subset", None)
    if scene_norm:
        ds["scene_norm"] = scene_norm
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
    if eval_all:
        val_subset = val_ds          # 外部清单已筛选过：整表即评测集
    else:
        n_train = max(1, int(val_frac * len(train_ds)))
        val_subset = torch.utils.data.Subset(val_ds, list(range(n_train, len(val_ds))))
    loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False,
                                         num_workers=0)
    if data_range is not None:
        # 固定动态范围：跳过全表 min/max 扫描（大留出集省一整趟 IO）
        val_dr = float(data_range)
    else:
        mn, mx = float("inf"), float("-inf")
        for vb in loader:
            t = vb["target"].float()
            mn = min(mn, float(t.min()))
            mx = max(mx, float(t.max()))
        val_dr = mx - mn if (mx - mn) > 1e-8 else 1.0
    return ck, cfg, loader, val_dr


def model_scores(stage, device, subset=None, val_frac=0.9, manifest=None,
                 scene_norm=None, eval_all=False, data_range=None):
    ck, cfg, loader, val_dr = load_val(stage, device=device, subset=subset,
                                       val_frac=val_frac, manifest=manifest,
                                       scene_norm=scene_norm, eval_all=eval_all,
                                       data_range=data_range)
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
            # per_scene 标准化：不 clamp（允许负/超1）
            m = evaluate_all(pred, gt, data_range=val_dr, scale=SCALE,
                             with_lpips=False)
            rows.append({k: float(m[k]) for k in METRICS if m.get(k) is not None})
    print(f"  [{stage}] {len(rows)} samples, val_dr={val_dr:.6f}")
    return rows


def bicubic_scores(stage_for_cfg, device, subset=None, val_frac=0.9, manifest=None,
                   scene_norm=None, eval_all=False, data_range=None):
    """bicubic 基线：对 frame0（干净锚点）上采样到 GT 尺寸，作'无对齐'参照。"""
    _, cfg, loader, val_dr = load_val(stage_for_cfg, device=device, subset=subset,
                                      val_frac=val_frac, manifest=manifest,
                                      scene_norm=scene_norm, eval_all=eval_all,
                                      data_range=data_range)
    rows = []
    with torch.no_grad():
        for batch in loader:
            lrs = batch["lrs"].to(device)        # [1,T,C,h,w]
            gt = batch["target"].to(device)      # [1,C,H,W]
            lr0 = lrs[:, 0]                       # [1,C,h,w] 干净锚点
            base = F.interpolate(lr0.float(), size=gt.shape[-2:],
                                 mode="bicubic", align_corners=False)
            m = evaluate_all(base, gt, data_range=val_dr, scale=SCALE,
                             with_lpips=False)
            rows.append({k: float(m[k]) for k in METRICS if m.get(k) is not None})
    print(f"  [bicubic] {len(rows)} samples (frame0 anchor), val_dr={val_dr:.6f}")
    return rows


def mean_scores(stage_for_cfg, device, subset=None, val_frac=0.9, manifest=None,
                scene_norm=None, eval_all=False, data_range=None):
    """mean 基线（金标准地板用）：逐样本 GT 的**空间均值图**。

    Δ_floor = PSNR(bicubic) − PSNR(mean) 即 evaluate_paired_floor 的
    bicubic_vs_mean 口径，且此处与 Δ_model 共用**同一裁剪**，不受跨文件口径影响。
    """
    _, cfg, loader, val_dr = load_val(stage_for_cfg, device=device, subset=subset,
                                      val_frac=val_frac, manifest=manifest,
                                      scene_norm=scene_norm, eval_all=eval_all,
                                      data_range=data_range)
    rows = []
    with torch.no_grad():
        for batch in loader:
            gt = batch["target"].to(device)
            mu = gt.mean(dim=(-2, -1), keepdim=True).expand_as(gt)
            m = evaluate_all(mu, gt, data_range=val_dr, scale=SCALE,
                             with_lpips=False)
            rows.append({k: float(m[k]) for k in METRICS if m.get(k) is not None})
    print(f"  [mean] {len(rows)} samples (per-sample spatial mean), "
          f"val_dr={val_dr:.6f}")
    return rows


def paired_test(a, b):
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
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save", default="experiments/eval_mtxa_worldstrat.json")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--val_frac", type=float, default=0.9)
    ap.add_argument("--no_bicubic", action="store_true")
    # --- 第二数据集站点面板（R2）新增：外部留出清单 + 固定动态范围 + mean 基线 ---
    ap.add_argument("--manifest", default=None,
                    help="外部 manifest（留出景清单），整体替换 cfg 的 manifest")
    ap.add_argument("--scene_norm", default=None,
                    help="与 --manifest 逐位对齐的逐景 (mu,sd) 文件")
    ap.add_argument("--eval_all", action="store_true",
                    help="评测整张（已筛选的）manifest，而非只取后 val_frac 部分")
    ap.add_argument("--data_range", type=float, default=None,
                    help="固定 PSNR 动态范围；给定则跳过 val_dr 全表扫描")
    ap.add_argument("--with_mean", action="store_true",
                    help="额外算逐样本空间均值基线（金标准 Δ_floor 用）")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")
    print(f"[info] device={device}  scipy={_HAS_SCIPY}  ckpt_root={CKPT_ROOT}")
    kw = dict(subset=args.subset, val_frac=args.val_frac, manifest=args.manifest,
              scene_norm=args.scene_norm, eval_all=args.eval_all,
              data_range=args.data_range)

    scores = {}
    for s in args.stages:
        try:
            scores[s] = model_scores(s, device, **kw)
        except FileNotFoundError as e:
            print(f"[skip] {s}: {e}")

    if not args.no_bicubic and scores:
        ref_stage = args.stages[0] if args.stages[0] in scores else next(
            iter(scores))
        scores["bicubic"] = bicubic_scores(ref_stage, device, **kw)
    if args.with_mean and scores:
        ref_stage = args.stages[0] if args.stages[0] in scores else next(
            iter(scores))
        scores["mean"] = mean_scores(ref_stage, device, **kw)

    # 绝对均值（纵览各档落点）
    print("\n===== 各档绝对均值（验证集）=====")
    for s, rows in scores.items():
        if not rows:
            continue
        line = f"  {s:16s}:"
        for k in METRICS:
            vals = [r[k] for r in rows if k in r]
            if vals:
                line += f"  {k}={sum(vals)/len(vals):.4f}"
        print(line)

    print("\n===== 逐样本配对检验 =====")
    pairs = [
        ("mtxa_mis_xa", "mtxa_mis_noxa", "cross-attn 自对齐贡献 (xa − noxa)"),
        ("mtxa_mis_xa", "mtxa_aligned", "距无失配上界 (xa − aligned)"),
        ("mtxa_mis_xa", "bicubic", "相对 bicubic 方法增益 (xa − bicubic)"),
        ("bicubic", "mean", "Δ_floor 金标准口径 (bicubic − mean)"),
    ]
    results = {}
    for a, b, title in pairs:
        if a not in scores or b not in scores:
            print(f"\n-- {title} -- (缺少 {a} 或 {b}，跳过)")
            continue
        print(f"\n-- {title} --")
        r = paired_test(scores[a], scores[b])
        results[f"{a}_vs_{b}"] = r
        for k, v in r.items():
            print(f"  {k}:")
            print(fmt(v))

    if args.save:
        meta = dict(manifest=args.manifest, scene_norm=args.scene_norm,
                    eval_all=bool(args.eval_all), data_range=args.data_range,
                    stages=list(args.stages))
        with open(os.path.join(BASE, args.save), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "per_sample": scores, "paired": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n[saved] {args.save}")


if __name__ == "__main__":
    main()
