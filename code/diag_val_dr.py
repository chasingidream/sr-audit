"""
诊断 mtxa_mis_xa 档的两个反常现象：
  1) val_dr=1.000000（其他所有档都是 0.4819）—— 同一验证集范围不该变
  2) 单次 validation 耗时 1630s（mtxa_aligned 仅 110s）—— 验证集规模疑似异常
做法：完全复刻 train.py 里的数据构建流程，检查
  - len(train_ds) / n_train / len(val_ds)
  - val 上 target 的全局 min/max（即 val_dr）
  - target 是否含 NaN
纯 CPU、num_workers=0，不干扰正在跑的 GPU 训练。
"""
import os
import sys
import math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
sys.path.insert(0, os.path.join(BASE, "code", "models"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 强制不用 GPU

import torch
from data.dataset import get_dataset  # noqa: E402


def load_cfg(stage):
    sys.path.insert(0, os.path.join(BASE, "code", "configs"))
    import importlib
    m = importlib.import_module("worldstrat_ablation")
    cfg = dict(m.CONFIG) if hasattr(m, "CONFIG") else {}
    if not cfg:
        raise SystemExit("configs/worldstrat_ablation.py 里没找到 CONFIG")
    cfg["stage"] = stage
    return cfg


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "mtxa_mis_xa"
    cfg = load_cfg(stage)

    # 复刻 train.py 的 jitter 判定
    _is_mis = stage.startswith("misaligned") or stage.startswith("xs_misaligned") \
        or stage.startswith("mtxa_mis")
    geo_jitter = cfg.get("geo_jitter") if _is_mis else None
    cfg["geo_jitter_all"] = stage.startswith("xs_misaligned")

    print(f"[diag] stage={stage}  geo_jitter={geo_jitter}  jitter_all={cfg['geo_jitter_all']}")

    train_ds = get_dataset(cfg, project_root=BASE, deterministic=False, geo_jitter=geo_jitter)
    val_ds = get_dataset(cfg, project_root=BASE, deterministic=True, geo_jitter=geo_jitter)

    len_train = len(train_ds)
    n_train = max(1, int(0.9 * len_train))
    len_val_full = len(val_ds)
    print(f"[diag] len(train_ds)={len_train}  n_train={n_train}  len(val_ds_full)={len_val_full}")

    val_subset = torch.utils.data.Subset(val_ds, list(range(n_train, len_val_full)))
    print(f"[diag] 实际验证集样本数 = {len(val_subset)}  (期望 ~10% = {len_val_full - n_train})")

    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=cfg.get("batch_size", 4), shuffle=False, num_workers=0)

    mn, mx = float("inf"), float("-inf")
    nan_batches = 0
    n_seen = 0
    max_batches = int(os.environ.get("DIAG_MAX_BATCHES", "200"))
    for bi, vb in enumerate(val_loader):
        if bi >= max_batches:
            break
        t = vb["target"].float()
        if torch.isnan(t).any() or torch.isinf(t).any():
            nan_batches += 1
            t = torch.nan_to_num(t, nan=0.0)
        mn = min(mn, float(t.min()))
        mx = max(mx, float(t.max()))
        n_seen += t.shape[0]

    print(f"[diag] 扫描 {n_seen} 个验证样本（{min(max_batches, bi+1)} 批）")
    print(f"[diag] target min={mn:.6f}  max={mx:.6f}  ->  val_dr={mx - mn:.6f}")
    print(f"[diag] 含 NaN/Inf 的批次数 = {nan_batches}")

    # 对照：期望值
    print(f"[diag] 期望 val_dr（其他档实测）= 0.4819")
    if abs((mx - mn) - 0.4819) > 1e-3:
        print("[diag] >>> 与 0.4819 不一致：本档验证集范围/规模确实异常 <<<")
    else:
        print("[diag] >>> 与 0.4819 一致：val_dr 正常 <<<")


if __name__ == "__main__":
    main()
