# -*- coding: utf-8 -*-
"""SEN2VENµS **ATTO 站点**（ceiling/positive regime 的跨站点泛化，模型层）。

由 code/tools/make_s2v_station_cfgs.py 生成；与 external_sen2venus_so2.py 逐项一致，
仅 dataset.site / root 与 save_dir 不同，以保证与 FGMANAUS / SO2 臂完全可比。
数据源：HF TACO 镜像 tacofoundation/sen2venus，经 code/s2v_extract_direct.py 直读提取，
        data/external/sen2venus_hf/ATTO/ 下为二级 zip。

用法：
  python code/train.py --cfg code/configs/external_sen2venus_atto_s42.py --stage sisr --seed 42
  python code/audit_axis.py --cfg code/configs/external_sen2venus_atto_s42.py \
      --stage sisr --save experiments/audit_axis_s2v_atto_s42.json
"""

CONFIG = {
    # --- model（与 FGMANAUS / SO2 轴逐项一致）---
    "in_ch": 4,
    "feat": 48,
    "scale": 2,
    "out_size": None,
    "T": 1,
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
        "site": "ATTO",
        "root": "data/external/sen2venus_hf",
        "bands": "b2b3b4b8",
        "div": 10000.0,
        "norm": "global",
    },

    # --- training ---
    "stage": "sisr",
    "batch_size": 4,
    "epochs": 30,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "val_every": 5,
    "val_frac": 0.1,
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

    "save_dir": "experiments/ckpts_s2v_atto",
    "log_dir": "experiments/logs_s2v_atto",
    "project_root": "..",
}
