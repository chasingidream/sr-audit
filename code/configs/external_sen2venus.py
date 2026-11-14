"""SEN2VENµS 对照配置（P1 三轴：跨传感器 + 同日单帧 + 2×）。

作用：在【无时相内容差】的条件下测同一架构的 SR 增益。P1 审计已证明
该数据集配对近乎完美（CC 0.997、双三次比常数图高 +21.96 dB），
本配置用于验证：在这种"干净"条件下，模型能否真的超过双三次。
若"跨传感器"是主因，这里也该失败；若这里成功而 WorldStrat 失败，
则直接把主因排除掉"跨传感器"。

超参与 configs/worldstrat_ablation.py 对齐，只改数据相关项。

数据：Sentinel-2 10m LR 128×128（4 波段 B2/B3/B4/B8）
      -> VENµS 5m HR 256×256，同日获取，单帧（T=1，无多时相）。

注意：T=1 时对齐模块无意义，故只跑 sisr 档。

切换档位：
  python code/train.py --cfg code/configs/external_sen2venus.py --stage sisr
"""

CONFIG = {
    # --- model ---
    "in_ch": 4,                  # B2/B3/B4/B8
    "feat": 48,
    "scale": 2,                  # 128 -> 256
    "out_size": None,
    "T": 1,                      # 同日单帧，无多时相
    "lr_size": 64,               # 与另两个数据集统一（HR 对应 128）
    "n_groups": 3,
    "n_blocks": 3,
    "rsa_radius": 3,
    "use_hr_ref": False,
    "use_conf": True,
    "align_mode": "cross_attn",
    "align_ref": "frame0",

    # --- data ---
    "dataset": {
        "type": "sen2venus",
        "bands": "b2b3b4b8",
        "div": 10000.0,          # int16 DN = 反射率*10000
        "norm": "global",
    },

    # --- training ---
    "stage": "sisr",             # T=1，无对齐无融合无参考
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

    # 独立子目录：避免与 WorldStrat 同名 stage 的 ckpt 互相覆盖
    "save_dir": "experiments/ckpts_s2v",
    "log_dir": "experiments/logs_s2v",
    "project_root": "..",
}
