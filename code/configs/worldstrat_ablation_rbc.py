# 方案 A+B 变体（2026-09-04 用户拍板）：
#   A. residual_base=True：输出 = bicubic_up(frame0) + 网络残差
#      （结构上保证起点=bicubic，L1 惩罚残差伪影 -> 抑制周期纹）
#   B. hr_per_lr=7.0 -> hr_size=448 原生裁剪，out_size=None：
#      彻底去掉 448->424 非整数重采样伪影源
#   附带：纯 L1（fourier/wavelet 已证伪有害）。
# 用法：python train.py --stage sisr_rbc_l1 --cfg code/configs/worldstrat_ablation_rbc.py --epochs 90
# 门控：train.py 剥离 "_l1"/"_rbc" 后缀按基础名（sisr）执行；存盘目录独立。
import os as _os

_src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "worldstrat_ablation.py")
with open(_src, encoding="utf-8") as _f:
    exec(compile(_f.read(), _src, "exec"))

CONFIG["residual_base"] = True
CONFIG["dataset"]["hr_per_lr"] = 7.0       # B：448 原生（64*7），不再 424
CONFIG["out_size"] = None                  # 去掉末端插值
CONFIG["loss"]["w_fourier"] = 0.0
CONFIG["loss"]["w_wavelet"] = 0.0
