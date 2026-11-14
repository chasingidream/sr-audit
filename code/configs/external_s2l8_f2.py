"""S2<->L8 4-fold CV fold 2 训练槽（held-out: [3, 13, 19, 23, 27]）。

薄封装：继承 external_s2l8.py，仅改 use_ids（剔除 held-out fold）与 save_dir/log_dir。
评测该 fold 时用 eval_s2l8.py --ids [3, 13, 19, 23, 27] --cfg 本文件。
"""
import importlib.util
import os

_p = os.path.join(os.path.dirname(__file__), 'external_s2l8.py')
_s = importlib.util.spec_from_file_location('_base_s2l8', _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

CONFIG = dict(_m.CONFIG)
CONFIG['dataset'] = dict(_m.CONFIG['dataset'])
CONFIG['dataset']['use_ids'] = [1, 2, 5, 6, 10, 14, 15, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
CONFIG['save_dir'] = 'experiments/ckpts_s2l8_f2'
CONFIG['log_dir'] = 'experiments/logs_s2l8_f2'
