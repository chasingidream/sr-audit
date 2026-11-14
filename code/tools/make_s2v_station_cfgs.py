# -*- coding: utf-8 -*-
"""为新 SEN2VENµS 站点生成模型层配置（与 FGMANAUS / SO2 轴逐项一致）。

产出（每站点 base + 3 个 seed 薄封装）：
  code/configs/external_sen2venus_<key>.py
  code/configs/external_sen2venus_<key>_s{42,123,7}.py

仅改数据相关项（site / root / save_dir / log_dir），其余超参与主协议一致，
以保证与已报道的 FGMANAUS / SO2 臂可比。

用法：python code/tools/make_s2v_station_cfgs.py NARYN ATTO SUDOUE-2
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG_DIR = os.path.join(ROOT, "code", "configs")

BASE = '''# -*- coding: utf-8 -*-
"""SEN2VENµS **{site} 站点**（ceiling/positive regime 的跨站点泛化，模型层）。

由 code/tools/make_s2v_station_cfgs.py 生成；与 external_sen2venus_so2.py 逐项一致，
仅 dataset.site / root 与 save_dir 不同，以保证与 FGMANAUS / SO2 臂完全可比。
数据源：HF TACO 镜像 tacofoundation/sen2venus，经 code/s2v_extract_direct.py 直读提取，
        {root}/{site}/ 下为二级 zip。

用法：
  python code/train.py --cfg code/configs/external_sen2venus_{key}_s42.py --stage sisr --seed 42
  python code/audit_axis.py --cfg code/configs/external_sen2venus_{key}_s42.py \\
      --stage sisr --save experiments/audit_axis_s2v_{key}_s42.json
"""

CONFIG = {{
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
    "dataset": {{
        "type": "sen2venus",
        "site": "{site}",
        "root": "{root}",
        "bands": "b2b3b4b8",
        "div": 10000.0,
        "norm": "global",
    }},

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

    "loss": {{
        "w_l1": 1.0,
        "w_char": 0.0,
        "w_perc": 0.0,
        "w_fourier": 0.1,
        "w_wavelet": 0.1,
        "w_flow": 0.0,
        "w_cyclic": 0.0,
    }},

    "save_dir": "experiments/ckpts_s2v_{key}",
    "log_dir": "experiments/logs_s2v_{key}",
    "project_root": "..",
}}
'''

WRAP = '''"""SEN2VENµS {site} 站点 — seed {seed} 运行槽（薄封装，仅改 save_dir/log_dir）。"""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), "external_sen2venus_{key}.py")
_s = importlib.util.spec_from_file_location("_base_s2v_{key}", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

CONFIG = dict(_m.CONFIG)
CONFIG["save_dir"] = "experiments/ckpts_s2v_{key}_s{seed}"
CONFIG["log_dir"] = "experiments/logs_s2v_{key}_s{seed}"
'''


def main():
    sites = sys.argv[1:] or ["NARYN", "ATTO", "SUDOUE-2"]
    root = "data/external/sen2venus_hf"
    for site in sites:
        key = site.lower().replace("-", "_")
        bp = os.path.join(CFG_DIR, f"external_sen2venus_{key}.py")
        with open(bp, "w", encoding="utf-8") as f:
            f.write(BASE.format(site=site, key=key, root=root))
        print("wrote", os.path.relpath(bp, ROOT))
        for seed in (42, 123, 7):
            wp = os.path.join(CFG_DIR, f"external_sen2venus_{key}_s{seed}.py")
            with open(wp, "w", encoding="utf-8") as f:
                f.write(WRAP.format(site=site, key=key, seed=seed))
            print("  wrote", os.path.relpath(wp, ROOT))


if __name__ == "__main__":
    main()
