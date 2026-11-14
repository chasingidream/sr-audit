"""PROBA-V NIR 波段 —— 正向轴的【独立地理子集】复现（回应评审意见①）。

评审意见（一）的核心要求：
    "至少补充一个独立的正例轴。可以是：同一 PROBA-V 数据集但不同地理区域/季节的子集。"

为什么 NIR 恰好命中这条：
  * PROBA-V 官方 train/RED 与 train/NIR 的 imgset 编号 **完全不相交**
    （RED = imgset0000–0593，594 景；NIR = imgset0594–1159，566 景），
    => NIR 是**同一数据集 / 同一传感器 / 同一 3× 退化**下的**另一批独立地理 ROI**，
       不是 RED 的重新切分，也不是换波段重跑同一批场景。
  * 因此 NIR 轴能分离两个变异源：
      - RED 三 seed  -> 只分离"训练随机性"，场景集相同；
      - NIR 三 seed  -> 同时分离"训练随机性"与"地理/光谱抽样"。

与 external_probav_n300.py 的唯一差异：
  - dataset.band = "NIR"（原 "RED"）
  - 独立的 save_dir / log_dir（避免覆盖 RED 轴 ckpt）

⚠ 口径一致性（与 RED 轴严格可比）：
  * 除 band 外，模型超参 / 训练超参 / loss / val_frac=0.5 / 确定性裁剪 **全部一致**；
  * val_frac = 0.5 -> 566 对 -> train 283 / val 283（从头重训，不复用 RED 权重）；
  * 评测走 code/eval_paired_floor.py 与 code/audit_axis.py，口径与 RED 轴完全相同。

用法：
  python code/train.py --cfg code/configs/external_probav_n300_nir_s42.py \
      --stage mtxa_aligned --seed 42
  python code/eval_paired_floor.py --cfg code/configs/external_probav_n300_nir_s42.py \
      --stage mtxa_aligned --save experiments/paired_probav_nir_s42.json
"""

CONFIG = {
    # --- model（与 external_probav_n300.py 完全一致）---
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
        "band": "NIR",          # ★ 唯一的数据侧差异：独立地理 ROI 子集
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

    "val_frac": 0.5,            # 566 -> train 283 / val 283

    "loss": {
        "w_l1": 1.0,
        "w_char": 0.0,
        "w_perc": 0.0,
        "w_fourier": 0.1,
        "w_wavelet": 0.1,
        "w_flow": 0.0,
        "w_cyclic": 0.0,
    },

    "save_dir": "experiments/ckpts_probav_nir",
    "log_dir": "experiments/logs_probav_nir",
    "project_root": "..",
}
