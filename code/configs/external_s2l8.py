"""S2<->L8 跨传感器 3x 单帧 SR 配置（P1-exp：把唯一正向轴扩到真正的跨传感器轴）。

设计要点：
  - 单帧（T=1）：S2↔L8 每 ROI 仅 1 对 L8(30m) + S2(10m)，无多时相，故 T=1。
  - LR = L8 RGB [3,2,1]，HR = S2 RGB [2,1,0]，均 3 通道（满足模型 in_ch==out_ch）。
  - 原生 3x：L8 64×64 -> S2 192×192（out_size 用整数倍，无需非整数末端插值）。
  - 跨传感器辐射偏置用逐场景标准化（per_scene）消除：LR 的 (mu,sd) 同时作用于 LR/HR。
  - stage=sisr：单帧超分，不开 RSA/HR-ref（T=1 本就无对齐；与 WorldStrat 单图 SOTA
    对照条款一致），干净隔离"cross-sensor 3x 本身"的增益。

⚠ 评测口径：必须用 eval_s2l8.py（全 22 ROI 单确定性 crop，n=22，与主文 S2↔L8 Δ_floor
  臂 n=22 一致），而非按 90/10 切分的 val；最终 Δ_model 以 n=22 配对 t 报出。

用法：
  python code/train.py --cfg code/configs/external_s2l8.py --stage sisr --epochs 80
  python code/eval_s2l8.py --cfg code/configs/external_s2l8.py --epoch 79 \
      --save experiments/paired_s2l8.json
"""

CONFIG = {
    # --- model（单帧 3x SR；与 PROBA-V 同骨架，T=1）---
    # 4 通道：L8 [B2 blue, B3 green, B4 red, B5 NIR] <-> S2 [B2 blue, B3 green, B4 red, B8 NIR]
    # ⚠ L8 导出波段序 = [B2,B3,B4,B5,B6,B7]（无 coastal，从 B2 起；由波段统计证实：
    #   band3 mean=0.30=NIR 亮度）。旧 (3,2,1) 实际选了 [NIR,red,green] 与 S2 [R,G,B]
    #   逐通道错位（NIR->red 负相关），Δfloor 被打到 -1.6~-2.5。改物理配对 4 波段后
    #   Δfloor=+2.26±3.23，与 e1 全波段灰度 +2.24 一致。
    "in_ch": 4,
    "feat": 48,
    "scale": 3,
    "out_size": None,
    "T": 1,
    "lr_size": 64,
    "n_groups": 3,
    "n_blocks": 3,
    "rsa_radius": 3,
    "use_hr_ref": False,
    "use_conf": True,
    "align_mode": "cost_volume",
    "align_ref": "frame0",

    # --- data ---
    "dataset": {
        "type": "s2l8",
        "lr_bands": (0, 1, 2, 3),
        "hr_bands": (0, 1, 2, 3),
        "div": 1.0,
        # house recipe：与 PROBA-V/SEN2VENµS 一致用数据集级 global 归一化（统计量只取
        # LR 侧，同时作用于 LR/HR）。此前 "none" 是因早期 NaN bug 的临时绕行，现已修复。
        # Δ_model 对全局仿射不变，norm 只影响优化条件。
        "norm": "global",
    },

    # --- training ---
    "stage": "sisr",
    "batch_size": 4,
    "epochs": 80,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "val_every": 10,
    "resume": None,

    # 训练用 90/10 切分（仅监控用）；最终评测见 eval_s2l8.py（全 22）
    "val_frac": 0.1,

    "loss": {
        "w_l1": 1.0,
        "w_char": 0.0,
        "w_perc": 0.0,
        "w_fourier": 0.1,
        "w_wavelet": 0.1,
        "w_flow": 0.0,
        "w_cyclic": 0.0,
    },

    "save_dir": "experiments/ckpts_s2l8",
    "log_dir": "experiments/logs_s2l8",
    "project_root": "..",
}
