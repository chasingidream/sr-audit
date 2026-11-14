#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三档 regime 图（P3-C 交付物）。

同一审计协议（逐样本 PSNR 平均 + 配对 t + 技能地板）套到三条轴，得到三种可区分机制：
  positive / ceiling / floor-failure。

左图（核心）：横轴 Δ_floor = bicubic − mean（"可恢复信息 headroom"），
              纵轴 Δ_model = model − bicubic（方法层增益），含 95%CI 与判定线：
                - 对角线 y = −x  <=> model = mean（技能地板）
                - 竖线 x = 0     <=> bicubic = mean
              区域内：右上=positive，右下=-ceiling，左半=floor-failure。
右图：各轴 Δ_model 的水平条形（含 95%CI），按 regime 着色。

数据源（逐样本配对口径，金标准）：
  experiments/paired_probav.json     (PROBA-V 3×,  n=60)
  experiments/paired_s2v.json        (SEN2VENµS 2×, n=13)
  experiments/paired_worldstrat.json (WorldStrat 6.63×, n=60)

用法：python code/fig_regime.py
输出：experiments/fig_regime.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 中文字体（Windows 微软雅黑）---
_FONT = r"C:\Windows\Fonts\msyh.ttc"
if os.path.exists(_FONT):
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "experiments", "fig_regime.png")

# --- 三轴数据（逐样本配对；x=bicubic−mean, y=model−bicubic, y 的 95%CI）---
# PROBA-V 用 P3-D 的 3 seed × n=297 扩展轴（lo/hi = 3 seed 极差；p 取最差 seed）；
# 历史 n=60 口径为 +2.33 dB（与 3-seed 均值 +2.42 一致），见 docs/P3_正向复现与三档regime图.md §五。
AXES = [
    dict(key="probav", label="PROBA-V 3×（3 seed）\n同传感器·多帧",
         x=1.428, y=2.422, lo=2.295, hi=2.561, p=1.0678e-26, n=297,
         regime="positive", seeds_pts=[2.410, 2.295, 2.561]),
    dict(key="s2v", label="SEN2VENµS 2×\n跨传感器·同日",
         x=7.988, y=0.129, lo=-0.063, hi=0.320, p=0.1685, n=13,
         regime="ceiling"),
    dict(key="ws", label="WorldStrat 6.63×\n跨传感器·多帧",
         x=-7.357, y=-1.120, lo=-2.643, hi=0.402, p=0.1462, n=60,
         regime="floor-failure"),
]

STYLE = {
    "positive":      dict(color="#1a9850", name="positive\n真实方法增益"),
    "ceiling":       dict(color="#f4a300", name="ceiling\n双三次已解决任务"),
    "floor-failure": dict(color="#d73027", name="floor-failure\n地板被击穿"),
}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6),
                              gridspec_kw={"width_ratios": [1.35, 1.0]})

# ============ 左：regime 相图 ============
xs = [a["x"] for a in AXES] + [0.0]
xlim = (min(xs) - 2.6, max(xs) + 2.6)
yabs = max(abs(a["y"]) for a in AXES) + max(abs(a["hi"]) for a in AXES)
ylim = (-max(6.0, yabs + 1.2), max(6.0, yabs + 1.2))

# 地板线 model=mean: y = -x
lx = [xlim[0], xlim[1]]
ax.plot(lx, [-v for v in lx], color="#555555", lw=1.4, ls="--", zorder=1)
ax.fill_between(lx, [-v for v in lx], ylim[1], color="#1a9850", alpha=0.05, zorder=0)
ax.fill_between(lx, ylim[0], [-v for v in lx], color="#d73027", alpha=0.05, zorder=0)
ax.axvline(0.0, color="#888888", lw=1.1, ls=":", zorder=1)
ax.axhline(0.0, color="#888888", lw=1.1, ls=":", zorder=1)

# region 标注
ax.text(xlim[1] - 0.35, ylim[1] - 0.5, "过地板（model>mean）", ha="right", va="top",
        fontsize=9, color="#1a9850")
ax.text(xlim[0] + 0.35, ylim[0] + 0.5, "未过地板（model<mean）", ha="left", va="bottom",
        fontsize=9, color="#d73027")
ax.text(xlim[0] + 0.35, ylim[1] - 0.5, "floor-failure 区\n(bicubic 自身弱于均值)",
        ha="left", va="top", fontsize=9, color="#d73027", alpha=0.85)
ax.text(xlim[1] - 0.35, -3.55, "ceiling 区\n(bicubic 已吃满可恢复信息)",
        ha="right", va="top", fontsize=9, color="#f4a300", alpha=1.0)

for a in AXES:
    st = STYLE[a["regime"]]
    ax.errorbar(a["x"], a["y"], yerr=[[a["y"] - a["lo"]], [a["hi"] - a["y"]]],
                fmt="o", ms=13, color=st["color"], ecolor=st["color"],
                elinewidth=2.2, capsize=5, capthick=2.0, zorder=5,
                markeredgecolor="white", markeredgewidth=1.4)
    # 各 seed 的散点（仅多 seed 轴）：右移一点避免被主标记遮住
    for sy in a.get("seeds_pts", []):
        ax.plot(a["x"] + 0.30, sy, marker="o", ms=7.5, mfc="white",
                mec=st["color"], mew=1.8, zorder=7)
    dy = 0.62 if a["key"] != "s2v" else -0.95
    ax.annotate(a["label"] + f"\nn={a['n']}  p={a['p']:.3g}",
                (a["x"], a["y"]), textcoords="offset points",
                xytext=(12, 10 if dy > 0 else -38),
                fontsize=9.5, color="#222222", ha="left", va="center", zorder=6)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel("Δ_floor = PSNR(bicubic) − PSNR(mean)   [dB]\n可恢复信息 headroom（双三次相对平凡均值预测器）", fontsize=10.5)
ax.set_ylabel("Δ_model = PSNR(model) − PSNR(bicubic)   [dB]\n方法层增益（含 95%CI）", fontsize=10.5)
ax.set_title("三档 regime 相图：同一审计协议，三种可区分机制", fontsize=12.5, pad=10.5)
ax.grid(alpha=0.22, ls="-", lw=0.6)
ax.text(0.985, 0.015, "虚线 y=−x：技能地板 model=mean", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.5, color="#555555")

# ============ 右：Δ_model 条形（含 CI） ============
order = ["positive", "ceiling", "floor-failure"]
axes_sorted = sorted(AXES, key=lambda a: order.index(a["regime"]))
ys = list(range(len(axes_sorted)))[::-1]
for yi, a in zip(ys, axes_sorted):
    st = STYLE[a["regime"]]
    ax2.barh(yi, a["y"], height=0.5, color=st["color"], alpha=0.88,
             edgecolor="white", zorder=3)
    ax2.plot([a["lo"], a["hi"]], [yi, yi], color="#333333", lw=1.8, zorder=4)
    for e in (a["lo"], a["hi"]):
        ax2.plot([e, e], [yi - 0.13, yi + 0.13], color="#333333", lw=1.8, zorder=4)
    txt = f"{a['y']:+.2f}  p={a['p']:.3g}"
    ax2.text(a["hi"] + 0.25, yi, txt, va="center", ha="left", fontsize=9.5, color="#222222")

ax2.axvline(0.0, color="#555555", lw=1.3, ls="--", zorder=2)
ax2.set_yticks(ys)
ax2.set_yticklabels([a["label"].split("\n")[0] + f"（{a['regime']}）" for a in axes_sorted],
                    fontsize=10)
ax2.set_xlabel("Δ_model = PSNR(model) − PSNR(bicubic)   [dB]", fontsize=10.5)
ax2.set_title("方法层增益（含 95%CI）", fontsize=12.5, pad=10.5)
ax2.set_xlim(min(a["lo"] for a in AXES) - 1.6, max(a["hi"] for a in AXES) + 3.4)
ax2.set_ylim(-0.65, len(axes_sorted) - 0.35)
ax2.grid(axis="x", alpha=0.22, ls="-", lw=0.6)
ax2.set_axisbelow(True)

fig.suptitle("跨传感器超分增益的三档机制：positive / ceiling / floor-failure",
             fontsize=13.5, y=1.005)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"[saved] {OUT}")
