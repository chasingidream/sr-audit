"""SEN2VENµS NARYN 站点 — seed 7 运行槽（薄封装，仅改 save_dir/log_dir）。"""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), "external_sen2venus_naryn.py")
_s = importlib.util.spec_from_file_location("_base_s2v_naryn", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

CONFIG = dict(_m.CONFIG)
CONFIG["save_dir"] = "experiments/ckpts_s2v_naryn_s7"
CONFIG["log_dir"] = "experiments/logs_s2v_naryn_s7"
