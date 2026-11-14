"""
三档 mtxa 消融的统一复核评测（同代码、同 val_dr、同验证集划分）。

动机：2026-09-03 曾因启动时漏传 --cfg，导致 mtxa_mis_xa 静默跑在
configs/default.py 的合成数据配置上（synthetic / feat=64 / epochs=50 / 无 geo_jitter），
白跑 6.5 小时。为避免再次出现"配置不一致却拿数字直接对比"，本脚本不信任外部传入的
配置，而是**从每个 ckpt 内部保存的 cfg 还原该档真实设置**（dataset / geo_jitter /
align_mode / align_ref / feat 等），再在同一套评测代码下重算指标。

要点：
  - 复刻 train.py 的验证集划分（deterministic=True 固定 crop，后 10% 作 val），
    geo_jitter 仅对 misaligned/mtxa_mis 前缀的档生效（与 train.py 一致）。
  - val_dr 在验证集上现算（WorldStrat 应为 ~0.4819，合成数据会是 1.0），
    并打印出来供交叉核对 —— 若出现非 0.4819 说明该档数据配置异常。
  - 输出 PSNR/SSIM/SAM/ERGAS/CC/RMSE，满足投稿对光谱保真指标的要求。

用法：
  python code/eval_compare_mtxa.py                      # 评 ep29 的最终 ckpt
  python code/eval_compare_mtxa.py --epoch 13           # 评指定 epoch
  CUDA_VISIBLE_DEVICES="" python code/eval_compare_mtxa.py   # 强制 CPU（训练占用 GPU 时）
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, os.path.join(BASE, "code", "models"))

import torch  # noqa: E402
from data.dataset import get_dataset  # noqa: E402
from models.sr_model import build_model  # noqa: E402
from eval_metrics import evaluate_all  # noqa: E402

STAGES = ["mtxa_aligned", "mtxa_mis_noxa", "mtxa_mis_xa"]


def pick_ckpt(stage, epoch=None):
    d = os.path.join(BASE, "experiments", "ckpts", stage)
    if not os.path.isdir(d):
        return None
    fs = [f for f in os.listdir(d) if f.startswith("ckpt_ep") and f.endswith(".pt")]
    if not fs:
        return None
    if epoch is None:
        fs.sort(key=lambda f: int(f.split("_ep")[-1].split(".")[0]))
        return os.path.join(d, fs[-1])
    p = os.path.join(d, f"ckpt_ep{epoch}.pt")
    return p if os.path.exists(p) else None


def eval_stage(stage, epoch, device, scale=4, max_batches=None):
    ckpt_path = pick_ckpt(stage, epoch)
    if ckpt_path is None:
        print(f"[skip] {stage}: 无 ckpt")
        return None

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("cfg", {}))
    print(f"\n=== {stage}  ckpt={os.path.basename(ckpt_path)}  epoch={ck.get('epoch')}")

    # --- 配置体检：确认这档确实跑在 WorldStrat 真实数据上 ---
    ds_cfg = cfg.get("dataset", {})
    ds_type = ds_cfg.get("type") if isinstance(ds_cfg, dict) else ds_cfg
    if ds_type != "worldstrat":
        print(f"  [!!] dataset={ds_type} —— 不是 WorldStrat，该档数据配置异常，结果不可比！")

    # 复刻 train.py 的 jitter 判定
    _is_mis = (stage.startswith("misaligned")
               or stage.startswith("xs_misaligned")
               or stage.startswith("mtxa_mis"))
    geo_jitter = cfg.get("geo_jitter") if _is_mis else None
    cfg["geo_jitter_all"] = stage.startswith("xs_misaligned")
    print(f"  dataset={ds_type}  align={cfg.get('align_mode')}/{cfg.get('align_ref')}"
          f"  feat={cfg.get('feat')}  jitter={'yes' if geo_jitter else 'no'}")

    # --- 数据（与 train.py 完全相同的划分） ---
    train_ds = get_dataset(cfg, project_root=BASE, deterministic=False,
                           geo_jitter=geo_jitter)
    val_ds = get_dataset(cfg, project_root=BASE, deterministic=True,
                         geo_jitter=geo_jitter)
    n_train = max(1, int(0.9 * len(train_ds)))
    val_subset = torch.utils.data.Subset(val_ds, list(range(n_train, len(val_ds))))
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=0)

    # --- val_dr：在验证集上现算，作为统一满量程 ---
    mn, mx = float("inf"), float("-inf")
    for vb in val_loader:
        t = vb["target"].float()
        mn = min(mn, float(t.min()))
        mx = max(mx, float(t.max()))
    val_dr = mx - mn if (mx - mn) > 1e-8 else 1.0
    flag = "" if abs(val_dr - 0.4819) < 1e-2 else "  [!!] 非 0.4819，配置存疑"
    print(f"  val_size={len(val_subset)}  val_dr={val_dr:.6f}{flag}")

    # --- 模型 ---
    model = build_model(cfg).to(device)
    missing = model.load_state_dict(ck["model"], strict=False)
    if missing.missing_keys:
        print(f"  [warn] 缺失键 {len(missing.missing_keys)} 个（结构不完全匹配）")
    model.eval()

    use_rsa = not stage.endswith("noxa")
    use_ref = False  # mtxa_* 一律无 HR 参考（无泄漏）

    acc = {}
    n = 0
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if max_batches and bi >= max_batches:
                break
            lrs = batch["lrs"].to(device)
            hr_ref = batch["hr_ref"].to(device)
            gt = batch["target"].to(device)
            pred, _ = model(lrs, hr_ref, return_aux=True,
                            use_rsa=use_rsa, use_ref=use_ref)
            pred = pred.clamp(0, 1)
            m = evaluate_all(pred, gt, data_range=val_dr, scale=scale,
                             with_lpips=False)
            for k, v in m.items():
                if v is None:
                    continue
                acc[k] = acc.get(k, 0.0) + float(v)
            n += 1

    if n == 0:
        print("  [!!] 验证集为空")
        return None
    out = {k: v / n for k, v in acc.items()}
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=None)
    ap.add_argument("--stages", nargs="*", default=STAGES)
    ap.add_argument("--scale", type=float, default=4)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu)
                          else "cpu")
    print(f"[info] device={device}")

    results = {}
    for s in args.stages:
        r = eval_stage(s, args.epoch, device, scale=args.scale,
                       max_batches=args.max_batches)
        if r:
            results[s] = r

    if len(results) >= 2:
        print("\n===== 汇总（PSNR / SSIM）=====")
        for s, r in results.items():
            print(f"  {s:16s} PSNR={r.get('psnr', float('nan')):.2f}"
                  f"  SSIM={r.get('ssim', float('nan')):.4f}")
        if "mtxa_mis_noxa" in results and "mtxa_mis_xa" in results:
            d = results["mtxa_mis_xa"]["psnr"] - results["mtxa_mis_noxa"]["psnr"]
            print(f"\n  cross-attn 自对齐净贡献 = {d:+.2f} dB  (mis_xa − mis_noxa)")
        if "mtxa_aligned" in results and "mtxa_mis_xa" in results:
            g = results["mtxa_aligned"]["psnr"] - results["mtxa_mis_xa"]["psnr"]
            print(f"  距无失配上界还差      = {g:+.2f} dB  (aligned − mis_xa)")


if __name__ == "__main__":
    main()
