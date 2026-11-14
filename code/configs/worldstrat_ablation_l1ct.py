# L1 + ConvTranspose 上采样器变体（诊断用，2026-09-04）。
# 动机：diag_vis_samples 证实所有档输出有 PixelShuffle(7)+bilinear 448->424 的
# 系统性棋盘/摩尔纹，模型被钉在 bicubic±3dB。本变体：
#   1) upsampler="convtranspose"：单一 ConvTranspose2d(k=7,s=7)，无重叠核无棋盘
#   2) 末端 448->424 插值加 antialias（sr_model.py 内置）
#   3) 纯 L1（排除 fourier/wavelet 干扰）
# 用法：python train.py --stage sisr_ct_l1 --cfg code/configs/worldstrat_ablation_l1ct.py --epochs 90
# 门控：train.py 循环剥离 "_l1"/"_ct" 后缀按基础名（sisr）执行；存盘目录独立。
import os as _os

_src = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "worldstrat_ablation_l1.py")
with open(_src, encoding="utf-8") as _f:
    exec(compile(_f.read(), _src, "exec"))

CONFIG["upsampler"] = "convtranspose"
