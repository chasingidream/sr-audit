"""S2<->L8 4-fold CV fold 0 训练槽（held-out: [1, 6, 15, 21, 25, 29]）。

薄封装：继承 external_s2l8.py，仅改 use_ids（剔除 held-out fold）与 save_dir/log_dir。
评测该 fold 时用 eval_s2l8.py --ids [1, 6, 15, 21, 25, 29] --cfg 本文件。
"""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), 'external_s2l8.py')
_s = importlib.util.spec_from_file_location('_base_s2l8', _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

CONFIG = dict(_m.CONFIG)
CONFIG['dataset'] = dict(_m.CONFIG['dataset'])
CONFIG['dataset']['use_ids'] = [2, 3, 5, 10, 13, 14, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30]
CONFIG['save_dir'] = 'experiments/ckpts_s2l8_f0'
CONFIG['log_dir'] = 'experiments/logs_s2l8_f0'
