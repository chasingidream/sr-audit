"""PROBA-V n≈300 统计功效扩展配置（P3-D：唯一正向轴的规模化复核）。

与 external_probav.py 的唯一差异：
  - val_frac = 0.5  -> 594 对 -> train 297 / val 297
  - 独立的 save_dir / log_dir（避免覆盖原轴的 ckpt）

为什么需要它（P3 可达性复核结论，见 docs/P3_正向复现与三档regime图.md §一）：
  PROBA-V 全量仅 594 对，历史口径 val=后10% 只有 60。为把"唯一正向轴"的统计功效
  拉到计划书要求的 n≥300（本机数据上限 ~297），改用 50/50 划分 —— 因 val 扩大后
  原训练集(前90%)会与 val 重叠，必须【重训】而非仅重评。

⚠ 口径说明：本轴与 external_probav.py（val 后10%，n=60）是两个独立实验，结论需并列报告；
  paired_probav.json(n=60) 仍是历史金标准，本配置产出 paired_probav_n297_s*。

用法：
  python code/train.py --cfg code/configs/external_probav_n300.py --stage mtxa_aligned --seed 42
  python code/eval_paired_floor.py --cfg code/configs/external_probav_n300.py \
      --stage mtxa_aligned --save experiments/paired_probav_n297_s42.json
"""

CONFIG = {
    # --- model（与 external_probav.py 完全一致）---
    "in_ch": 1,
    "feat": 48,
    "scale": 3,
    "out_size": None,
    "T": 5,
    "lr_size": 64,
    "n_groups": 3,
    "n_blocks": 3,
    "rsa_radius": 3,
    "use_hr_ref": False,
    "use_conf": True,
    "align_mode": "cross_attn",
    "align_ref": "frame0",

    # --- data ---
    "dataset": {
        "type": "probav",
        "band": "RED",
        "div": 65535.0,
        "split": "train",
        "norm": "global",
    },

    # --- training ---
    "stage": "mtxa_aligned",
    "batch_size": 4,
    "epochs": 30,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "val_every": 5,
    "resume": None,

    # ★ P3-D 关键差异
    "val_frac": 0.5,

    "loss": {
        "w_l1": 1.0,
        "w_char": 0.0,
        "w_perc": 0.0,
        "w_fourier": 0.1,
        "w_wavelet": 0.1,
        "w_flow": 0.0,
        "w_cyclic": 0.0,
    },

    "save_dir": "experiments/ckpts_probav_n300",
    "log_dir": "experiments/logs_probav_n300",
    "project_root": "..",
}
