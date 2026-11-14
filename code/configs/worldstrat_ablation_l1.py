# L1-only 损失变体（诊断用）：关闭 w_fourier / w_wavelet 谱域辅助损失。
# 动机（2026-09-04）：全档重训后 diag_output_stats 显示模型输出 std_ratio≈3.0
# （病态过散/尖峰），怀疑谱域损失把能量推向错误频段并压制 L1 主导。
# 用法：python train.py --stage sisr_l1 --cfg code/configs/worldstrat_ablation_l1.py
# 门控：train.py 自动剥离 "_l1" 后缀按基础名（sisr）执行；存盘/日志独立于 sisr。
import os as _os

_src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "worldstrat_ablation.py")
with open(_src, encoding="utf-8") as _f:
    exec(compile(_f.read(), _src, "exec"))

CONFIG["loss"]["w_fourier"] = 0.0
CONFIG["loss"]["w_wavelet"] = 0.0
