# -*- coding: utf-8 -*-
"""
Task #39 支撑图: 尺度信息缺口 (scale information gap) 的坍缩图。

D_floor = PSNR(bicubic(LR), GT) - PSNR(mean(GT), GT)
  - 数据层(零训练): 直接测量 LR 的可恢复性上限
  - 方法层(同模型三轴): 同一协议下逐样本 PSNR 平均
  - 已发表基线: WorldStrat 官方 HighRes-net 的 bicubic 基线 (394 held-out 场景)

输出: experiments/fig_scale_gap.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# --- 中文字体 (Windows) ---
def _setup_cjk():
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            name = fm.FontProperties(fname=p).get_name()
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None

_CJK_NAME = _setup_cjk()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "fig_scale_gap.png")

# ---- 数据层 (零训练审计, docs/跨数据集审计_汇总.md) --------------------------
# (scale, D_floor, label)
data_layer = [
    (2.0,   33.86 - 11.90, "SEN2VENµS\n2× same-day"),
    (3.0,   21.92 - 19.83, "PROBA-V\n3× same-sensor"),
    (6.63,  20.08 - 20.17, "WorldStrat\n6.63× cross-sensor"),
]

# ---- 方法层 (同一模型, eval_paired_floor 口径, 逐样本 PSNR) -------------------
# scale, D_floor, err(± half-CI), label
method_layer = [
    (2.0,   35.4038 - 27.3209, 0.0,  "SEN2VENµS (n=13)"),
    (3.0,   28.2371 - 26.8150, (2.3869 - 0.4573) / 2, "PROBA-V (n=60)"),
    (6.63,  30.8217 - 38.1787, (8.9278 - 5.7862) / 2, "WorldStrat (n=60)"),
]

# ---- 已发表基线互证: WorldStrat 官方 HighRes-net (394 场景) ------------------
hr_scale = 6.63
hr_floor = 24.677 - 31.593   # bicubic - mean_floor

# ---- 模型相对地板的真实技能 (model - floor) ---------------------------------
# 用 HighRes-net: model 32.588 - floor 31.593
hr_model_over_floor = 32.588 - 31.593

fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=170)

xs = np.array([p[0] for p in data_layer])
ys = np.array([p[1] for p in data_layer])
ax.plot(xs, ys, "o-", color="#1f77b4", lw=2.4, ms=10,
        label="数据层：零训练 (bicubic − mean)")
for x, y, lab in data_layer:
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8.5, color="#1f77b4")

mx = np.array([p[0] for p in method_layer])
my = np.array([p[1] for p in method_layer])
me = np.array([p[2] for p in method_layer])
ax.errorbar(mx[1:], my[1:], yerr=me[1:], fmt="s--", color="#d62728", lw=2.2,
            ms=9, capsize=5, label="方法层：同模型三轴 (逐样本, 95%CI)")
ax.scatter(mx[:1], my[:1], marker="s", s=90, facecolors="none",
           edgecolors="#d62728", lw=2.2)
ax.annotate("SEN2VENµS (n=13, CI 未测)", (mx[0], my[0]),
            textcoords="offset points", xytext=(14, -3), ha="left",
            va="center", fontsize=8.5, color="#d62728")

# 已发表实现互证
ax.scatter([hr_scale], [hr_floor], marker="*", s=360, color="#2ca02c",
           zorder=5, label="已发表基线：WorldStrat 官方 HighRes-net\n(其 bicubic 基线, 394 场景)")
ax.annotate("官方 HighRes-net 的 bicubic\n(24.68 vs 地板 31.59 = −6.91 dB)",
            (hr_scale, hr_floor), textcoords="offset points", xytext=(-8, -34),
            ha="right", fontsize=8.5, color="#2ca02c")

ax.axhline(0, color="k", lw=1.4, ls="-")
ax.axhspan(-12, 0, color="#ffd9d9", alpha=0.55, zorder=0)
ax.text(2.03, -11.2, "地板以下：上采样不优于「预测均值」\n→ 任何 model−bicubic 的正值都不能解释为真实技能",
        fontsize=9, color="#a30000", va="bottom")

ax.set_xscale("log")
ax.set_xticks([2.0, 3.0, 6.63])
ax.set_xticklabels(["2×", "3×", "6.63×"])
ax.set_xlim(1.8, 7.6)
ax.set_ylim(-12.5, 25)
ax.set_xlabel("超分倍率 (scale factor, log)", fontsize=11)
ax.set_ylabel(r"尺度信息缺口  $\Delta_{floor}$ = PSNR(bicubic) − PSNR(mean)   [dB]",
              fontsize=10.5)
ax.set_title("尺度信息缺口随倍率单调坍缩：跨传感器 6.63× 处增益跌破技能地板",
             fontsize=12.5, pad=12)
ax.grid(alpha=0.25, ls=":", which="both")
ax.legend(loc="upper right", fontsize=8.6, framealpha=0.95)

fig.tight_layout()
fig.savefig(OUT)
print(f"[ok] 已写出 {OUT}")
print(f"  data-layer  : 2x {ys[0]:+.2f} | 3x {ys[1]:+.2f} | 6.63x {ys[2]:+.2f} dB")
print(f"  method-layer: 2x {my[0]:+.2f} | 3x {my[1]:+.2f} | 6.63x {my[2]:+.2f} dB")
print(f"  HighRes-net official bicubic-floor: {hr_floor:+.2f} dB; "
      f"model-over-floor: {hr_model_over_floor:+.2f} dB")
