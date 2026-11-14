"""SEN2VENµS SO2 站点模型层 — seed 123 运行槽。

薄封装：继承 external_sen2venus_so2.py，仅改 save_dir/log_dir 以避免 3 个 seed
互相覆盖（train.py 的 ckpt 落盘路径 = save_dir/<stage>/，不含 seed 后缀）。
"""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), "external_sen2venus_so2.py")
_s = importlib.util.spec_from_file_location("_base_s2v_so2", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

CONFIG = dict(_m.CONFIG)
CONFIG["save_dir"] = "experiments/ckpts_s2v_so2_s123"
CONFIG["log_dir"] = "experiments/logs_s2v_so2_s123"
