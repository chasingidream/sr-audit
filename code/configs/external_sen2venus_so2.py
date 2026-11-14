# -*- coding: utf-8 -*-
"""SEN2VENµS **SO2 站点**（ceiling regime 的第二地理站点，模型层）。

目的：把 ceiling regime 从"单一站点（FGMANAUS, n = 13）上的可证伪预测"升级为
"两个地理站点都成立"。此前该轴只有**数据层**跨站点证据（SO2 Δ_floor =
+6.44 dB, n = 74, p = 1.4e-23，见 s2v_crossstation_dfloor.json），**模型层结论
（Δ_model 不显著）仍停留在单站点**，是审稿人最直接的攻击点。

与 external_sen2venus.py 的差异（其余超参逐项一致，保证可比）：
  * dataset.site = "SO2"
  * dataset.root = "data/external/sen2venus_hf"（HF TACO 镜像布局，
    {root}/{SITE}/ 目录下直接是二级 zip；Zenodo 原站因 403 限流不可达）
  * save_dir / log_dir 独立子目录，避免覆盖 FGMANAUS 的 ckpt

数据：Sentinel-2 10 m LR -> VENµS 5 m HR，同日获取，单帧（T = 1），2×。
      SO2 共 738 对，最后 10% = 74 对作验证切片。

用法：
  python code/train.py --cfg code/configs/external_sen2venus_so2.py --stage sisr --seed 42
  python code/audit_axis.py --cfg code/configs/external_sen2venus_so2.py \
      --stage sisr --save experiments/audit_axis_s2v_so2_s42.json
"""

CONFIG = {
    # --- model（与 FGMANAUS 轴逐项一致）---
    "in_ch": 4,                  # B2/B3/B4/B8
    "feat": 48,
    "scale": 2,                  # LR -> HR
    "out_size": None,
    "T": 1,                      # 同日单帧，无多时相
    "lr_size": 64,
    "n_groups": 3,
    "n_blocks": 3,
    "rsa_radius": 3,
    "use_hr_ref": False,
    "use_conf": True,
    "align_mode": "cross_attn",
    "align_ref": "frame0",

    # --- data：仅站点与根目录不同 ---
    "dataset": {
        "type": "sen2venus",
        "site": "SO2",
        "root": "data/external/sen2venus_hf",
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

    # 独立子目录：避免与 FGMANAUS 站的 ckpt 互相覆盖
    "save_dir": "experiments/ckpts_s2v_so2",
    "log_dir": "experiments/logs_s2v_so2",
    "project_root": "..",
}
