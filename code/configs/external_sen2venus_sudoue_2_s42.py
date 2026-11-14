"""SEN2VENµS SUDOUE-2 站点 — seed 42 运行槽（薄封装，仅改 save_dir/log_dir）。"""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), "external_sen2venus_sudoue_2.py")
_s = importlib.util.spec_from_file_location("_base_s2v_sudoue_2", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

CONFIG = dict(_m.CONFIG)
CONFIG["save_dir"] = "experiments/ckpts_s2v_sudoue_2_s42"
CONFIG["log_dir"] = "experiments/logs_s2v_sudoue_2_s42"
