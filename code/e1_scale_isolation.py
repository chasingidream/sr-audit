# -*- coding: utf-8 -*-
"""
Task #39 / E1: 同倍率对照 —— 把"倍率"从"时相/传感器去相关"里分离出来。

方法 (scale-isolation decomposition):
  真实三轴臂的 Δ_floor 同时被 "尺度" 与 "去相关(跨传感器 + 时相错位)" 两个因素污染。
  为隔离"尺度", 这里构造 **零去相关对照组**: 直接用 WorldStrat 的 SPOT HR 自身做 bicubic
  降采样到 2×/3×/6.63×, 再 bicubic 上采样回 HR。此时 LR 与 HR 同传感器、同日、零噪声
  (仅尺度变化), 故 Δ_floor(pure) = **纯尺度缺口**。

  于是分解:
    Δ_floor(real) − Δ_floor(pure@同倍率) = 去相关贡献
    - 2× : pure(2×) vs SEN2VENµS(2× 跨传感器同日)       → 传感器去相关
    - 3× : pure(3×) vs PROBA-V(3× 同传感器多时相)        → 时相去相关
    - 6.63×: pure(6.63×) vs WorldStrat(6.63× 跨传感器多时相) → 传感器+时相去相关

  注: 真实"跨传感器 3× 多时相"臂需要 S2↔L8(30m→10m=3×) 等外部数据, 本机暂无, 此处未做, 见文末说明。

输出:
  experiments/e1_scale_isolation.json
  experiments/fig_e1_decomp.png

依赖: numpy / tifffile / scipy (CPU, 无需 torch, 规避 vcomp140 segfault)。
"""
import os, json, math
import numpy as np
from scipy.ndimage import zoom
import tifffile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "processed", "worldstrat_manifest.json")
OUT_JSON = os.path.join(ROOT, "experiments", "e1_scale_isolation.json")
OUT_PNG = os.path.join(ROOT, "experiments", "fig_e1_decomp.png")

SCALES = [2.0, 3.0, 6.63]
N_SCENES = 60            # 确定性子集 (manifest 末 60, 与三轴评测量级一致)
CROP = 640              # 中心裁剪, 降计算
HR_DIV = 10000.0

# ---- 真实三轴臂 Δ_floor (来自 prior 实验结果, 同口径) ------------------------
# 数据层零训练 (docs/跨数据集审计_汇总.md)
REAL_DATALAYER = {2.0: 33.86 - 11.90, 3.0: 21.92 - 19.83, 6.63: 20.08 - 20.17}
REAL_LABEL = {2.0: "SEN2VENµS 2× 跨传感器同日",
              3.0: "PROBA-V 3× 同传感器多时相",
              6.63: "WorldStrat 6.63× 跨传感器多时相"}

# 方法层同模型三轴 —— **必须取"逐样本 PSNR 平均"口径**（全文规则，禁批池化）。
# ⚠ 2026-09-23 修 (自查 F4)：此前 2× 硬编码 35.4038-27.3209 = 8.0829，那是
#   group_mean_batchpooled 的值；逐样本口径为 35.3374-27.3491 = **7.9882**（差 0.09 dB，
#   不改变结论但违反自定口径）。现改为直接读 paired_*.json 的 group_mean_per_sample，
#   与主文 Table 1 / E3 协议 A 同源、可追溯。三轴统一用 n=60 历史划分，保持同量级可比。
PAIRED_SOURCES = {2.0: "paired_s2v.json",
                  3.0: "paired_probav.json",
                  6.63: "paired_worldstrat.json"}


def _per_sample_delta_floor(fname):
    """Δ_floor = PSNR(bicubic) − PSNR(mean)，逐样本平均（读 paired json）。"""
    d = json.load(open(os.path.join(ROOT, "experiments", fname)))
    return float(d["group_mean_per_sample"]["bicubic"]["psnr"]) - \
        float(d["group_mean_per_sample"]["mean"]["psnr"])


REAL_METHODLAYER = {s: _per_sample_delta_floor(f) for s, f in PAIRED_SOURCES.items()}


def psnr_db(a, b, data_range=1.0):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10((data_range ** 2) / mse))


def load_hr_rgb(path, div=HR_DIV):
    arr = tifffile.imread(path).astype(np.float64)   # (H,W,4) RGBA
    if arr.ndim == 2:
        arr = arr[..., None]
    rgb = arr[..., :3]
    rgb = rgb / div
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def center_crop(x, size=CROP):
    h, w = x.shape[:2]
    y0 = max(0, (h - size) // 2); x0 = max(0, (w - size) // 2)
    return x[y0:y0 + size, x0:x0 + size]


def main():
    m = json.load(open(MANIFEST))
    subset = m[-N_SCENES:]
    print(f"[info] 子集: {len(subset)} 场景, 中心裁剪 {CROP}x{CROP}")

    per_scale = {s: {"bicubic": [], "mean": [], "delta": []} for s in SCALES}
    for i, e in enumerate(subset):
        hr = load_hr_rgb(e["hr"])
        hr = center_crop(hr)
        for s in SCALES:
            # LR = HR 经 bicubic 降采样 (零去相关)
            zoom_out = 1.0 / s
            lr = zoom(hr, (zoom_out, zoom_out, 1.0), order=3)
            # 上采样回 HR 尺寸
            lr_up = zoom(lr, (hr.shape[0] / lr.shape[0], hr.shape[1] / lr.shape[1], 1.0), order=3)
            # 预测均值 (逐图空间均值, 常数图)
            mean_pred = hr.mean(axis=(0, 1), keepdims=True)
            p_bic = psnr_db(lr_up, hr)
            p_mean = psnr_db(mean_pred, hr)
            per_scale[s]["bicubic"].append(p_bic)
            per_scale[s]["mean"].append(p_mean)
            per_scale[s]["delta"].append(p_bic - p_mean)
        if (i + 1) % 10 == 0:
            print(f"  scene {i+1}/{len(subset)} done")

    results = {"n_scenes": N_SCENES, "crop": CROP, "scales": {}, "decomposition": {}}
    for s in SCALES:
        bic = np.array(per_scale[s]["bicubic"]); mn = np.array(per_scale[s]["mean"])
        df = np.array(per_scale[s]["delta"])
        results["scales"][s] = {
            "psnr_bicubic_mean": float(bic.mean()), "psnr_bicubic_std": float(bic.std()),
            "psnr_meanpred_mean": float(mn.mean()), "psnr_meanpred_std": float(mn.std()),
            "delta_floor_pure_mean": float(df.mean()), "delta_floor_pure_std": float(df.std()),
        }
        # 分解: 去相关贡献 = 真实 Δ_floor − 纯尺度 Δ_floor
        dl_real = REAL_DATALAYER[s]; df_real = REAL_METHODLAYER[s]
        results["decomposition"][s] = {
            "real_label": REAL_LABEL[s],
            "delta_floor_real_datalayer": dl_real,
            "delta_floor_real_methodlayer": df_real,
            "delta_floor_pure": float(df.mean()),
            "decorr_datalayer": dl_real - float(df.mean()),
            "decorr_methodlayer": df_real - float(df.mean()),
        }
    json.dump(results, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    print(f"[ok] 已写出 {OUT_JSON}")

    # 终端摘要
    print("\n=== E1 尺度隔离分解 (Δ_floor, dB) ===")
    print(f"{'scale':>6} | {'pure(同传感器同日)':>20} | {'real(数据层)':>14} | {'去相关(数据层)':>14} | {'real(方法层)':>14} | {'去相关(方法层)':>14}")
    for s in SCALES:
        r = results["decomposition"][s]
        print(f"{s:>6} | {r['delta_floor_pure']:>20.2f} | {r['delta_floor_real_datalayer']:>14.2f} | "
              f"{r['decorr_datalayer']:>+14.2f} | {r['delta_floor_real_methodlayer']:>14.2f} | {r['decorr_methodlayer']:>+14.2f}")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager as fm
        for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
            if os.path.exists(p):
                fm.fontManager.addfont(p)
                plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=p).get_name(), "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                break
        xs = SCALES
        pure = [results["scales"][s]["delta_floor_pure_mean"] for s in xs]
        rd = [REAL_DATALAYER[s] for s in xs]
        rm = [REAL_METHODLAYER[s] for s in xs]
        fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=170)
        ax.plot(xs, pure, "o-", color="#2ca02c", lw=2.4, ms=10,
                label="纯尺度缺口 (同传感器·同日·零去相关, 本实验)")
        ax.plot(xs, rd, "s--", color="#1f77b4", lw=2.2, ms=9,
                label="真实三轴 Δ_floor (数据层)")
        ax.plot(xs, rm, "^:", color="#d62728", lw=2.0, ms=9,
                label="真实三轴 Δ_floor (方法层)")
        # 标注去相关贡献 (方法层, 与纯尺度同口径、scale-invariant, 可直接相减)
        for s, pr in zip(xs, pure):
            dc = results["decomposition"][s]["decorr_methodlayer"]  # 负数=真实臂低于纯尺度
            ax.annotate(f"去相关 {dc:+.1f}", (s, pr), textcoords="offset points",
                        xytext=(8, -16), fontsize=8.5, color="#555555")
        ax.axhline(0, color="k", lw=1.2)
        ax.axhspan(-12, 0, color="#ffd9d9", alpha=0.5, zorder=0)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["2×", "3×", "6.63×"])
        ax.minorticks_off()
        ax.set_xlim(1.8, 7.6); ax.set_ylim(-12.5, 30)
        ax.set_xlabel("超分倍率 (scale factor, log)")
        ax.set_ylabel(r"尺度信息缺口 $\Delta_{floor}$ = PSNR(bicubic)−PSNR(mean)  [dB]")
        ax.set_title("E1 尺度隔离: 纯尺度缺口 vs 真实臂 (差额=去相关贡献)", fontsize=12.5, pad=12)
        ax.grid(alpha=0.25, ls=":")
        ax.legend(loc="upper right", fontsize=8.6, framealpha=0.95)
        fig.tight_layout(); fig.savefig(OUT_PNG)
        print(f"[ok] 已写出 {OUT_PNG}")
    except Exception as ex:
        print(f"[warn] 画图失败: {ex}")


if __name__ == "__main__":
    main()
