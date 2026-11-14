"""PROBA-V n≈300 统计功效扩展配置 — seed 123 运行槽（薄封装，见 s42）。"""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), "external_probav_n300.py")
_s = importlib.util.spec_from_file_location("_base_probav_n300", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

CONFIG = dict(_m.CONFIG)
CONFIG["save_dir"] = "experiments/ckpts_probav_n300_s123"
CONFIG["log_dir"] = "experiments/logs_probav_n300_s123"
