"""PROBA-V 对照配置（P1 三轴：同传感器 + 多帧 + 3×）。

作用：在【无跨传感器辐射差】的条件下，测量同一架构同一超参的 SR 增益，
与 WorldStrat（跨传感器 + 多帧 + 6.63×）、SEN2VENµS（跨传感器 + 同日 + 2×）
构成对照，用于定位"模型增益消失"到底由哪一维引起。

超参刻意与 configs/worldstrat_ablation.py 对齐（feat/n_blocks/lr/epochs/batch/损失），
只改数据相关项（in_ch / scale / lr_size / T），使跨数据集比较尽量干净。

数据：PROBA-V 300m LR 128×128 -> 100m HR 384×384，单波段（RED 或 NIR），
每场景 9~35 帧（按 quality mask clearance 降序取前 T=9）。
已做 per-scene 归一化（动态范围仅 0.03~0.13，且多时相数据上全局标定有害）。

切换档位：
  python code/train.py --cfg code/configs/external_probav.py --stage mtxa_aligned
  python code/train.py --cfg code/configs/external_probav.py --stage sisr
"""

CONFIG = {
    # --- model（与 worldstrat_ablation 对齐：feat=48 / n_blocks=3）---
    "in_ch": 1,                  # PROBA-V 单波段（RED 或 NIR）
    "feat": 48,
    "scale": 3,                  # 128 -> 384
    "out_size": None,            # 3× 是整数倍，无需末端插值
    # T 与 lr_size 刻意与 worldstrat_ablation 一致（T=5 / LR 64）：
    # CrossAttnAlign 逐帧做注意力且保留 T-1 次激活，显存 ~ (T-1)*(H*W)²，
    # 用原生 T=9 / LR=128 会直接 OOM（实测需 16GB）。
    "T": 5,                      # 取 clearance 最高的 5 帧
    "lr_size": 64,               # 从原生 128 裁到 64（HR 对应 192）
    "n_groups": 3,
    "n_blocks": 3,
    "rsa_radius": 3,
    "use_hr_ref": False,         # 无跨传感器参考；HR 只作监督目标
    "use_conf": True,
    "align_mode": "cross_attn",  # 已证实 cost-volume 在多时相上架构性失效
    "align_ref": "frame0",       # 锚点取 LR 帧，杜绝 GT 泄漏

    # --- data ---
    "dataset": {
        "type": "probav",
        "band": "RED",
        "div": 65535.0,          # 16-bit PNG（14-bit 有效）
        "split": "train",
        "norm": "global",     # 统计量只取 LR 侧
    },

    # --- training（与 worldstrat_ablation 一致）---
    "stage": "mtxa_aligned",
    "batch_size": 4,
    "epochs": 30,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "val_every": 5,
    "resume": None,

    "loss": {
        "w_l1": 1.0,
        "w_char": 0.0,
        "w_perc": 0.0,
        "w_fourier": 0.1,
        "w_wavelet": 0.1,
        "w_flow": 0.0,
        "w_cyclic": 0.0,
    },

    # 独立子目录：避免与世界 WorldStrat 同名 stage（mtxa_aligned）的 ckpt 互相覆盖
    "save_dir": "experiments/ckpts_probav",
    "log_dir": "experiments/logs_probav",
    "project_root": "..",
}
