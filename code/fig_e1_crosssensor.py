"""Figure: E1 same-scale (3x) arm comparison -- close the scale-vs-decorrelation decomposition.

Three arms at FIXED 3x scale:
  - pure-scale 3x   (SPOT HR self-bicubic, zero decorrelation)            -> +11.22 dB
  - PROBA-V 3x      (same-sensor, multi-temporal, real)                   -> +1.42 dB
  - S2<->L8 3x      (CROSS-sensor, multi-temporal, real, this study)      -> per-ROI spread

Reads experiments/e1_real_crosssensor_3x.json (auto-discovered ROI count).
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTP = "C:/Windows/Fonts/msyh.ttc"
if os.path.exists(FONTP):
    font_manager.fontManager.addfont(FONTP)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

JP = os.path.join(ROOT, "experiments", "e1_real_crosssensor_3x.json")
with open(JP) as f:
    D = json.load(f)
rows = D["rois"]
ref = D["reference_arms"]

gray = np.array([r["delta_floor_gray_db"] for r in rows], float)
b0 = np.array([r["delta_floor_b0_db"] for r in rows], float)
best = np.array([r["delta_floor_bpairs_best_db"] for r in rows], float)
pure = ref["pure_scale_3x_db"]
probav = ref["probav_3x_samesensor_multitemp_db"]

fig, ax = plt.subplots(figsize=(8.6, 5.0))
xs = [0, 1, 2]
labels = ["纯尺度 3×\n(SPOT 自双三次,\n零去相关)",
          "PROBA-V 3×\n(同传感器·多时相)",
          "S2 / L8 3×\n(跨传感器·多时相)"]
cols = ["#4C72B0", "#55A868", "#C44E52"]

# reference arms as labeled points/bars
for x, v, c, lab in [(0, pure, cols[0], "pure"), (1, probav, cols[1], "probav")]:
    ax.plot(x, v, "o", ms=13, color=c, zorder=5)
    ax.annotate("%.2f dB" % v, (x, v), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=10, fontweight="bold", color=c)

# cross-sensor arm: per-ROI scatter (gray) + mean + band-0/best markers
jit = (np.random.RandomState(0).rand(len(gray)) - 0.5) * 0.16
ax.scatter(np.full(len(gray), 2) + jit, gray, s=70, color=cols[2], zorder=4,
           edgecolor="white", linewidth=0.8, label="各 ROI (灰度)")
m = gray.mean()
sd = gray.std(ddof=1) if len(gray) > 1 else 0
ax.errorbar(2, m, yerr=sd, fmt="none", ecolor=cols[2], capsize=6, lw=2, zorder=3)
ax.plot(2, m, "D", ms=11, color="#7A1F23", zorder=6)
ax.annotate("均值 %.2f±%.2f" % (m, sd), (2, m), textcoords="offset points",
            xytext=(0, 14), ha="center", fontsize=10, fontweight="bold", color=cols[2])
# optimistic bound marker (band-0 / best-pair)
if len(b0):
    ax.scatter([2.18]*len(b0), b0, marker="^", s=40, color="#E6A0A6", zorder=4, label="单波段上界")
    ax.annotate("单波段上界 %.2f" % b0.mean(), (2.18, b0.mean()), textcoords="offset points",
                xytext=(8, 0), fontsize=8, color="#9c3b44")

ax.axhline(0.0, color="gray", ls="--", lw=1.2)
ax.annotate("Δ_floor = 0  (LR 不比「预测均值」含更多信息)", (0.02, 0.05),
            xycoords=("axes fraction", "data"), fontsize=8.5, color="gray")

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Δ_floor = PSNR(双三次(LR),GT) − PSNR(均值(GT),GT)  [dB]", fontsize=10)
ax.set_title("E1 同倍率对照 (固定 3×)：尺度 vs 去相关 的分解\n"
             "可恢复信息缺口 = f(尺度, 跨传感器去相关, 时相去相关)", fontsize=11, fontweight="bold")
ax.set_ylim(-6, 14)
ax.grid(axis="y", ls=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax.text(0.5, -0.20, "n=%d ROIs (S2 10m / L8 30m, 原生 3× 配对)" % len(rows),
        transform=ax.transAxes, ha="center", fontsize=8, color="gray")
plt.tight_layout()
OUT = os.path.join(ROOT, "experiments", "fig_e1_crosssensor.png")
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
