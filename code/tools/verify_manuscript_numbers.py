"""
主文数字回链核查（独立事实核查）。

每条 = (claim_id, 主文写法, 期望值, 源文件, 取值路径, 容差)。
输出 PASS / MISMATCH / NO-SOURCE / NOTE 四态。

用法（必须用项目解释器）：
    python code/tools/verify_manuscript_numbers.py
"""
import json
import os

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 开发仓用 experiments/，发布仓用 results/；两处都找，先命中先用
_DIRS = [os.path.join(_BASE, "experiments"), os.path.join(_BASE, "results")]
EXP = _DIRS[0]
_cache = {}


def load(fn):
    if fn in _cache:
        return _cache[fn]
    p = next((os.path.join(d, fn) for d in _DIRS
              if os.path.exists(os.path.join(d, fn))), None)
    if p is None:
        _cache[fn] = None
        return None
    try:
        _cache[fn] = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  [ERR load {fn}: {e}]")
        _cache[fn] = None
    return _cache[fn]


def get(d, path, default="<MISSING>"):
    if d is None:
        return default
    cur = d
    for k in path.split("/"):
        if isinstance(cur, dict):
            if k not in cur:
                return default
            cur = cur[k]
        elif isinstance(cur, list):
            try:
                cur = cur[int(k)]
            except Exception:
                return default
        else:
            return default
    return cur


RESULTS = []


def check(cid, text, expected, src, path, tol=5e-3):
    """expected/src/path 任一是列表时，逐项按位置比对（长度需一致）。"""
    if isinstance(path, (list, tuple)):
        d = load(src)
        vals = [get(d, p) for p in path]
        exps = expected if isinstance(expected, (list, tuple)) else [expected] * len(vals)
        actual = vals
        ok = all(isinstance(v, (int, float)) and abs(v - e) <= tol for v, e in zip(vals, exps))
    elif isinstance(src, (list, tuple)):
        vals = [get(load(s), path) for s in src]
        actual = vals
        exps = expected if isinstance(expected, (list, tuple)) else [expected] * len(vals)
        ok = all(isinstance(v, (int, float)) and abs(v - e) <= tol for v, e in zip(vals, exps))
    else:
        actual = get(load(src), path)
        ok = isinstance(actual, (int, float)) and abs(actual - expected) <= tol
    RESULTS.append({"id": cid, "manuscript": text, "expected": expected, "actual": actual,
                    "src": src, "path": path,
                    "status": "PASS" if ok else ("NO-SOURCE" if actual == "<MISSING>" else "MISMATCH")})


def note(cid, text, actual):
    RESULTS.append({"id": cid, "manuscript": text, "expected": None, "actual": actual,
                    "src": "-", "path": "-", "status": "NOTE"})


def raw(fn):
    p = next((os.path.join(d, fn) for d in _DIRS
              if os.path.exists(os.path.join(d, fn))), None)
    return open(p, encoding="utf-8").read() if p else ""


# ================= 数据层 Δ_floor（金标准口径） =================
check("D01", "PROBA-V RED 3× Δ_floor +1.43 (n=297)",
      1.43, "paired_probav_n297_s42.json", "paired_per_sample/bicubic_vs_mean/psnr/mean_diff", 0.02)
check("D02", "PROBA-V RED Δ_floor n=297 vs n=60: +1.428 / +1.422",
      [1.42781, 1.42208],
      ["paired_probav_n297_s42.json", "paired_probav.json"],
      "paired_per_sample/bicubic_vs_mean/psnr/mean_diff", 0.005)
check("D03", "PROBA-V NIR Δ_floor +1.64",
      1.6447, "paired_probav_nir_s42.json", "paired_per_sample/bicubic_vs_mean/psnr/mean_diff", 0.01)
check("D04", "FGMANAUS 2× Δ_floor +7.99",
      7.9882, "paired_s2v_s42.json", "paired_per_sample/bicubic_vs_mean/psnr/mean_diff", 0.01)
check("D05", "SO2 Δ_floor +6.443",
      6.4426, "paired_s2v_so2_s42.json", "paired_per_sample/bicubic_vs_mean/psnr/mean_diff", 0.01)
check("D06", "WorldStrat 6.63× Δ_floor −7.36 (protocol A)",
      -7.357, "paired_worldstrat.json", "paired_per_sample/bicubic_vs_mean/psnr/mean_diff", 0.01)
check("D07", "WorldStrat 6.63× Δ_floor CI 上界 −5.79",
      -5.7862, "paired_worldstrat.json", "paired_per_sample/bicubic_vs_mean/psnr/ci95/1", 0.01)
check("D08", "WorldStrat Δ_floor CI 下界 (单元素)",
      -8.93, "paired_worldstrat.json", "paired_per_sample/bicubic_vs_mean/psnr/ci95/0", 0.01)
check("D09", "S2↔L8 3× Δ_floor +2.33 ± 2.78",
      [2.3313, 2.7799], "paired_s2l8.json",
      ["paired_bicubic_vs_mean_gray/psnr/mean_diff", "paired_bicubic_vs_mean_gray/psnr/sd"], 0.01)
check("D10", "S2↔L8 held-out Δ_floor +2.15",
      2.1457, "paired_s2l8_cv.json", "delta_floor_psnr/mean_diff", 0.01)
# 数据层 p 值（主文引 0.0046 / 7.7e-11 / 2.8e-13 —— 均由 paired JSON 直接携带）
check("D11", "3× Δ_floor p = 0.0046 (n=60)",
      0.004557, "paired_probav.json", "paired_per_sample/bicubic_vs_mean/psnr/p", 0.0005)
check("D12", "2× Δ_floor p = 7.7e-11",
      7.6763e-11, "paired_s2v_s42.json", "paired_per_sample/bicubic_vs_mean/psnr/p", 1e-11)
check("D13", "6.63× Δ_floor p = 2.8e-13",
      2.7593e-13, "paired_worldstrat.json", "paired_per_sample/bicubic_vs_mean/psnr/p", 1e-13)

# ================= 方法层 Δ_model =================
check("M01", "PROBA-V 3× +2.42 ± 0.13 (三 seed 均值，p1 汇总)",
      2.42193, "p1_probav_subgroup.json", "seed_aggregate/mean_db", 0.005)
check("M02", "PROBA-V 3× 每 seed +2.41/+2.30/+2.56",
      [2.4099, 2.2945, 2.5613],
      ["paired_probav_n297_s42.json", "paired_probav_n297_s123.json", "paired_probav_n297_s7.json"],
      "paired_per_sample/model_vs_bicubic/psnr/mean_diff", 0.005)
check("M03", "PROBA-V 3× mean/sd = 2.4219 / 0.1338",
      0.13381, "p1_probav_subgroup.json", "seed_aggregate/sd_db", 0.005)
check("M04", "PROBA-V NIR 三 seed +2.439/+2.532/+2.537（均值 2.5026 ± 0.0556）",
      [2.4385, 2.5322, 2.5371],
      ["paired_probav_nir_s42.json", "paired_probav_nir_s123.json", "paired_probav_nir_s7.json"],
      "paired_per_sample/model_vs_bicubic/psnr/mean_diff", 0.005)
check("M05", "FGMANAUS +0.13 (paired s42)",
      0.12992, "paired_s2v_s42.json", "paired_per_sample/model_vs_bicubic/psnr/mean_diff", 0.005)
check("M06", "FGMANAUS 三 seed +0.130/+0.111/+0.088",
      [0.12992, 0.111, 0.0878], ["paired_s2v_s42.json", "paired_s2v_s123.json", "paired_s2v_s7.json"],
      "paired_per_sample/model_vs_bicubic/psnr/mean_diff", 0.005)
check("M07", "SO2 +1.82",
      1.8627, "paired_s2v_so2_s42.json", "paired_per_sample/model_vs_bicubic/psnr/mean_diff", 0.01)
check("M08", "SO2 三 seed +1.754/+1.863/+1.844",
      [1.7543, 1.8627, 1.8444], ["paired_s2v_so2_s7.json", "paired_s2v_so2_s42.json", "paired_s2v_so2_s123.json"],
      "paired_per_sample/model_vs_bicubic/psnr/mean_diff", 0.005)
check("M09", "NARYN +0.24",
      0.2428, "s2v_station_panel.json", "rows/2/dm", 0.01)
check("M10", "ATTO +1.20",
      1.1985, "s2v_station_panel.json", "rows/0/dm", 0.01)
check("M11", "SUDOUE-2 +0.90",
      0.8981, "s2v_station_panel.json", "rows/4/dm", 0.01)
check("M12", "S2↔L8 +0.74 (same-set)",
      0.7422, "paired_s2l8.json", "paired_model_vs_bicubic_gray/psnr/mean_diff", 0.01)
check("M13", "S2↔L8 held-out +0.33 (p=0.22)",
      [0.3274, 0.2183], "paired_s2l8_cv.json",
      ["delta_model_psnr/mean_diff", "delta_model_psnr/p"], 0.01)
check("M14", "WorldStrat 模型增益 −0.88, p=0.210",
      -0.8792, "paired_worldstrat.json", "paired_per_sample/model_vs_bicubic/psnr/mean_diff", 0.01)
check("M15", "WorldStrat 模型−均值 −8.24",
      -8.2362, "paired_worldstrat.json", "paired_per_sample/model_vs_mean/psnr/mean_diff", 0.01)
check("M16", "WorldStrat 批池化 +0.79, p=0.174",
      0.7856, "paired_worldstrat.json", "paired_batchpooled/model_vs_bicubic/psnr/mean_diff", 0.005)
check("M17", "WorldStrat 批池化口径 bicubic/mean PSNR (差 = −7.38)",
      [22.8486, 30.2276], "paired_worldstrat.json",
      ["group_mean_batchpooled/bicubic/psnr", "group_mean_batchpooled/mean/psnr"], 0.005)
check("M18", "WorldStrat 三 seed 极差 1.81 dB (s42 −1.276 / s123 −1.482 / s7 +0.330)",
      [0.329889, -1.482200], ["paired_ws_s7.json", "paired_ws_s123.json"],
      "paired_per_sample/model_vs_bicubic/psnr/mean_diff", 0.005)
check("M19", "WorldStrat s7 单 seed 报 +0.33 纯噪声",
      0.3299, "paired_ws_s7.json", "paired_per_sample/model_vs_bicubic/psnr/mean_diff", 0.005)

# ================= n 值 =================
check("N01", "FGMANAUS n_val 13", 13, "paired_s2v_s42.json", "n_sample", 0)
check("N02", "SO2 n_val 74", 74, "paired_s2v_so2_s42.json", "n_sample", 0)
check("N03", "NARYN n_val 31", 31, "s2v_station_panel.json", "rows/2/n_val", 0)
check("N04", "ATTO / SUDOUE-2 n_val 各 30",
      [30, 30], "s2v_station_panel.json", ["rows/0/n_val", "rows/4/n_val"], 0)
check("N05", "PROBA-V n 297", 297, "p1_probav_subgroup.json", "n_scenes", 0)
check("N06", "PROBA-V NIR n 283", 283, "paired_probav_nir_s42.json", "n_sample", 0)
check("N07", "S2↔L8 n 22", 22, "paired_s2l8.json", "n", 0)

# ================= 五站点面板 Δ_floor（金标准） =================
for st, exp in (("FGMANAUS", 7.99), ("SO2", 6.44), ("NARYN", 13.02), ("ATTO", 4.46), ("SUDOUE-2", 10.78)):
    check(f"X-{st}", f"{st} Δ_floor(gold) ~{exp}",
          exp, "s2v_crossstation_dfloor.json", f"sites/{st}/dfloor/psnr/mean_diff", 0.02)

# ================= 尺度隔离 =================
for f, exp in ((2.0, 13.20), (3.0, 8.64), (4.0, 6.79), (5.0, 5.75), (6.63, 4.70)):
    check(f"E1-{f}", f"纯尺度 {f}× Δ_floor {exp}",
          exp, "e1_matched_aa_5pt.json", f"scales/{f}/delta_floor_matched_mean", 0.05)

# ================= 对齐三臂 =================
_al = load("robustness_mtxa_worldstrat.json")
for cid, key, exp in (("A1", "xa_minus_noxa", -0.2928), ("A2", "xa_minus_aligned", -0.5649),
                      ("A3", "xa_minus_bicubic", -1.4282)):
    v = get(_al, f"views/A_full/{key}/mean")
    ok = isinstance(v, (int, float)) and abs(v - exp) < 0.01
    RESULTS.append({"id": cid, "manuscript": f"对齐臂 {key} 均值 {exp}",
                    "expected": exp, "actual": v, "src": "robustness_mtxa_worldstrat.json",
                    "path": f"views/A_full/{key}/mean",
                    "status": "PASS" if ok else ("NO-SOURCE" if v == "<MISSING>" else "MISMATCH")})

# ================= Table 2 外部方法 =================
check("T2-HR", "HighRes-net +7.91 / Δ_floor −6.916",
      -6.916, "audit_cards.json", "cards/2/delta_floor_db", 0.005)
check("T2-HR2", "HighRes-net audited gain +7.912",
      7.912, "audit_cards.json", "cards/2/audit_gain_db", 0.005)
check("T2-EDSR", "EDSR −2.428 (p=0.0298, d=−0.287)",
      -2.428, "audit_cards.json", "cards/1/audit_gain_db", 0.005)
check("T2-WDSR", "WDSR-a −3.021 (p=0.006694)",
      -3.021, "audit_cards.json", "cards/4/audit_gain_db", 0.005)
check("T2-RAMS", "RAMS +2.886 (CI 1.893..3.878, d=0.736)",
      2.886, "audit_cards.json", "cards/0/audit_gain_db", 0.005)
check("T2-RCAN", "RCAN −2.524 (p=0.023)",
      -2.524, "sota_review/rcan.json", "audit_gain_db", 0.005)
check("T2-SWIN", "SwinIR-lite −0.767 (p=0.300)",
      -0.767, "sota_review/swinir.json", "audit_gain_db", 0.005)
check("T2-EDNAT", "EDiffSR 原生 ×4 n=12: −7.66 (p=8.0e-11)",
      -7.6606, "sota_review/ediffsr_native_x4_multiscene.json", "paired/mean_diff", 0.005)
check("T2-EDNAT2", "EDiffSR 原生 ×4 CC(bic,GT)=0.944",
      0.94396, "sota_review/ediffsr_raw.json", "A_synthetic_x4/cc_bicubic", 0.005)

# ================= SEN2SR 原生复核 =================
check("SN1", "SEN2NAIPv2 n=120, +0.034, p=0.068",
      0.034, "sota_review/sen2sr_native.json", "audit_gain_db", 0.003)
check("SN2", "Δ_floor 14.64 / 0.23% (β)",
      14.64, "sota_review/sen2sr_native.json", "detail/delta_floor_db", 0.01)
check("SN3", "G4 1.387 @ period 4",
      1.387, "sota_review/sen2sr_native.json", "detail/grid_ratio_model", 0.005)
check("SN4", "corr(lr, box4(hr)) = 0.967 (0.928–0.995), 斜率 1.06, 子样本 0.906 vs 全体 0.907",
      [0.9671, 0.9281, 0.9951, 1.06], "sota_review/sen2sr_native.json",
      ["detail/genuine_cross_sensor_check/corr_lr_box4hr_mean",
       "detail/genuine_cross_sensor_check/min",
       "detail/genuine_cross_sensor_check/max",
       "detail/genuine_cross_sensor_check/linfit_slope"], 0.005)
check("SN5", "分布外 WorldStrat ×4 (n=60): −0.051, p=1.74e-6, G4 5.498",
      [-0.051, 5.498], "sota_review/sen2sr_native.json",
      ["detail/out_of_distribution_contrast/delta_model_db",
       "detail/out_of_distribution_contrast/grid_ratio_model"], 0.005)

# ================= 分层 / 回归 / 气候 =================
check("S01", "全球均值 −3.69, 92.6% 为负, n=3927",
      -3.68557, "geoscience_independent_validation.json", "global_mean_df", 0.005)
check("S02", "绿化 −4.488 vs 建成 −2.327（差 2.16 dB, p=4e-45, d=0.54）",
      -4.48838, "stratified_floor_summary.json", "veg_vs_built/veg_mean", 0.005)
check("S02b", "建成均值 −2.327", -2.32695, "stratified_floor_summary.json",
      "veg_vs_built/built_mean", 0.005)
check("S02c", "植被/建成 Mann-Whitney p=3.596e-45, d=0.5397",
      0.53972, "stratified_floor_summary.json", "veg_vs_built/cohen_d", 0.005)
check("S03", "IPCC 森林 −6.19", -6.19414, "stratified_floor_summary.json",
      "by_IPCC/rows/2/2", 0.01)
check("S04", "IPCC 建成 −2.33", -2.32695, "stratified_floor_summary.json",
      "by_IPCC/rows/7/2", 0.01)
check("S05", "IPCC η²=0.25", 0.250385, "stratified_floor_summary.json", "by_IPCC/eta2", 0.001)
check("S06", "SMOD η²=0.06", 0.061486, "stratified_floor_summary.json", "by_SMOD/eta2", 0.001)
check("S07", "时滞 η²=0.07 (>180d −7.81 / ≤30d −3.23)",
      -7.80516, "stratified_floor_summary.json", "by_lag/rows/0/2", 0.01)
check("S08", "云 ≥30% −18.7", -18.6515, "stratified_floor_summary.json",
      "by_cloud/rows/0/2", 0.05)
check("S09", "五物理量 R²=0.33, IPCC R²=0.11, β(bright_hr)=−2.53, p=5e-191",
      0.32845, "floor_predictors_regression.json", "r2", 0.001)
check("S10", "气候增量 0.25 -> 0.30 (空间 CV 30×60)",
      0.2974, "geoscience_independent_validation.json", "spatial_cv/30x60/combined/cv_r2", 0.001)
check("S11", "无影像气候集 0.15 (5×5) -> 0.08 (30×60)",
      0.0816, "geoscience_independent_validation.json", "spatial_cv/30x60/independent/cv_r2", 0.001)
check("S12", "干燥 vs 湿润差 1.7 dB, p=2e-5, d=0.43",
      -1.68696, "geoscience_independent_validation.json", "arid_vs_humid/diff", 0.01)
check("S13", "气候带 η²=0.007, p=3e-6",
      0.007102, "geoscience_independent_validation.json", "aridity_kw/eta2", 0.0005)

# ================= 图集精度层 =================
check("AT1", "1° 格网 K 折 CV R²=0.728",
      0.72826, "pathc_cellcv.json", "grids/1.0/kfold_by_cell_r2", 0.001)
check("AT2", "空间块 CV 30×60 R²=0.684",
      0.68435, "pathc_regime_acc.json", "tiers/spatial_30x60/r2", 0.001)
check("AT3", "30° 缓冲 R²=0.631",
      0.631, "pathc_atlas.json", "tier_A/r2_buffered30", 0.001)
check("AT4", "留战役 0.508 / 留国家 0.505",
      0.508, "pathc_atlas.json", "tier_B/r2_leave_campaign", 0.001)
check("AT5", "留国家 0.505（91 单元）",
      0.505, "pathc_atlas.json", "tier_B/r2_leave_country", 0.001)
check("AT6", "regime 平衡正确率 0.530 / AUC 0.774 / 符号 0.926 (1° K 折)",
      0.53026, "pathc_regime_acc.json", "tiers/cell_1deg/balanced_accuracy", 0.001)
check("AT7", "留战役 平衡正确率 0.536 / AUC 0.747",
      0.53602, "pathc_regime_acc.json", "tiers/leave_campaign/balanced_accuracy", 0.001)
check("AT8", "留战役 符号正确率 = 0.9223（≠ 基线 0.9264）",
      0.92233, "pathc_regime_acc.json", "tiers/leave_campaign/sign_accuracy", 0.001)

# ================= 自查披露 =================
check("G4-1", "早期档位 G4 = 6.28 @ 7px",
      6.2778, "audit_mtxa_mis_xa_rbc_zi.json", "grid_ratio_model", 0.005)
check("G4-2", "早期档位 G4 = 7.21 @ 7px",
      7.2137, "audit_sisr_rbc_zi.json", "grid_ratio_model", 0.005)
check("SA1", "自查三臂模型 PSNR 29.94 / 29.67 / 29.38",
      29.942, "audit_ws2_mtxa_aligned.json", "predictors/model/psnr", 0.005)
check("SA2", "自查 no-anchor 29.67", 29.67, "audit_ws2_mtxa_mis_noxa.json", "predictors/model/psnr", 0.02)
check("SA3", "自查 trainable 29.378", 29.378, "audit_ws2_mtxa_mis_xa.json", "predictors/model/psnr", 0.005)
check("SA4", "WorldStrat G1 (bicubic CC) 0.3028",
      0.30277, "paired_worldstrat.json", "group_mean_per_sample/bicubic/cc", 0.002)

# ================= 反事实 =================
check("CF1", "real −2.35 / radiometric +8.41 / rereg +9.28 / pure-scale +10.38",
      -2.35, "e1_paired_counterfactual.json", "stages/real/mean", 0.01)
check("CF2", "radiometric +8.409", 8.409, "e1_paired_counterfactual.json", "stages/radiometric/mean", 0.01)
check("CF3", "reregistered +9.276", 9.276, "e1_paired_counterfactual.json", "stages/reregistered/mean", 0.01)
check("CF4", "pure-scale +10.382", 10.382, "e1_paired_counterfactual.json", "stages/pure_scale/mean", 0.01)

# ============ R16：跨行派生量 / 聚合和（补 2026-10-07 复评 §四、§九） ============
# R16-A 捕获比 β = Δ_model / (Δ_floor + Δ_model)，由两个分量重算（跨行派生量）
def check_beta(cid, label, exp_pct, df_path, dm_path, tol=0.7):
    df = get(load(df_path[0]), df_path[1])
    dm = get(load(dm_path[0]), dm_path[1])
    if not (isinstance(df, (int, float)) and isinstance(dm, (int, float))):
        RESULTS.append({"id": cid, "manuscript": label, "expected": exp_pct, "actual": "<MISSING>",
                        "src": f"{df_path[0]}|{dm_path[0]}", "path": f"{df_path[1]} + {dm_path[1]}",
                        "status": "NO-SOURCE"})
        return None
    b = dm / (df + dm) * 100.0
    ok = abs(b - exp_pct) <= tol
    RESULTS.append({"id": cid, "manuscript": label, "expected": exp_pct, "actual": round(b, 3),
                    "src": f"{df_path[0]}|{dm_path[0]}", "path": f"{df_path[1]} + {dm_path[1]}",
                    "status": "PASS" if ok else "MISMATCH"})
    return b


BETAS = {}
BETAS["PROBA-V RED"] = check_beta("B01", "β PROBA-V RED 63%", 63,
    ("paired_probav_n297_s42.json", "paired_per_sample/bicubic_vs_mean/psnr/mean_diff"),
    ("p1_probav_subgroup.json", "seed_aggregate/mean_db"))
BETAS["PROBA-V NIR"] = check_beta("B02", "β PROBA-V NIR 60%", 60,
    ("paired_probav_nir_s42.json", "paired_per_sample/bicubic_vs_mean/psnr/mean_diff"),
    ("p1_probav_subgroup_nir.json", "seed_aggregate/mean_db"))
BETAS["RAMS"] = check_beta("B03", "β RAMS 67%（同 PROBA-V 轴）", 67,
    ("paired_probav_n297_s42.json", "paired_per_sample/bicubic_vs_mean/psnr/mean_diff"),
    ("audit_cards.json", "cards/0/audit_gain_db"))
BETAS["SO2"] = check_beta("B04", "β SO2 22%", 22,
    ("s2v_crossstation_dfloor.json", "sites/SO2/dfloor/psnr/mean_diff"),
    ("paired_s2v_so2_s42.json", "paired_per_sample/model_vs_bicubic/psnr/mean_diff"))
BETAS["ATTO"] = check_beta("B05", "β ATTO 21%", 21,
    ("s2v_crossstation_dfloor.json", "sites/ATTO/dfloor/psnr/mean_diff"),
    ("s2v_station_panel.json", "rows/0/dm"))
BETAS["SUDOUE-2"] = check_beta("B06", "β SUDOUE-2 7.7%", 7.7,
    ("s2v_crossstation_dfloor.json", "sites/SUDOUE-2/dfloor/psnr/mean_diff"),
    ("s2v_station_panel.json", "rows/4/dm"))
BETAS["S2L8-heldout"] = check_beta("B07", "β S2↔L8 留出 13%", 13,
    ("paired_s2l8_cv.json", "delta_floor_psnr/mean_diff"),
    ("paired_s2l8_cv.json", "delta_model_psnr/mean_diff"))
BETAS["NARYN"] = check_beta("B08", "β NARYN 1.8%（ceiling）", 1.8,
    ("s2v_crossstation_dfloor.json", "sites/NARYN/dfloor/psnr/mean_diff"),
    ("s2v_station_panel.json", "rows/2/dm"))
BETAS["FGMANAUS"] = check_beta("B09", "β FGMANAUS 1.6%（ceiling）", 1.6,
    ("paired_s2v_s42.json", "paired_per_sample/bicubic_vs_mean/psnr/mean_diff"),
    ("paired_s2v_s42.json", "paired_per_sample/model_vs_bicubic/psnr/mean_diff"))
BETAS["SEN2NAIPv2"] = check_beta("B10", "β SEN2NAIPv2 0.23%（ceiling）", 0.23,
    ("sota_review/sen2sr_native.json", "detail/delta_floor_db"),
    ("sota_review/sen2sr_native.json", "audit_gain_db"))

# R16-B β 空隙：ceiling 最大 < 正臂最小（无臂落在两者之间）
_ceiling = [BETAS[k] for k in ("NARYN", "FGMANAUS", "SEN2NAIPv2") if isinstance(BETAS.get(k), (int, float))]
_floorok = [BETAS[k] for k in ("PROBA-V RED", "PROBA-V NIR", "RAMS", "SO2", "ATTO", "SUDOUE-2", "S2L8-heldout")
            if isinstance(BETAS.get(k), (int, float))]
if _ceiling and _floorok:
    hi_c, lo_f = max(_ceiling), min(_floorok)
    RESULTS.append({"id": "B11", "manuscript": f"β 空隙：ceiling 最大 < 正臂最小（{hi_c:.2f}% < {lo_f:.2f}%）",
                    "expected": "ceiling<positive", "actual": f"{hi_c:.2f}% < {lo_f:.2f}%",
                    "src": "(derived)", "path": "-", "status": "PASS" if hi_c < lo_f else "MISMATCH"})

# R16-C 证据基数聚合 ≈14,900 对（分项相加）
_comp = {"WorldStrat": 3927, "SEN2NAIPv2": 8000, "PROBA-V RED": 594,
         "PROBA-V NIR": 566, "SEN2VENµS×5": 1773, "S2↔L8": 22}
_s = sum(_comp.values())
note("B12", "≈14,900 对聚合分项", " + ".join(f"{k} {v}" for k, v in _comp.items()) + f" = {_s}")
RESULTS.append({"id": "B13", "manuscript": "聚合和 ≈14,900（容差 100）", "expected": 14900, "actual": _s,
                "src": "(derived)", "path": "-", "status": "PASS" if abs(14900 - _s) <= 100 else "MISMATCH"})

# R16-D 91 国
check("B14", "91 国（S87_country 站点数）", 91, "pathc_stations.json", "S87_country/n_stations", 0)

# R16-E 协议间 Δ_floor 跨度 5.7 dB
_a = get(load("paired_worldstrat.json"), "paired_per_sample/bicubic_vs_mean/psnr/mean_diff")
_b = get(load("e3_datalayer_paired_ci.json"), "2/delta_floor_paired/mean")
if isinstance(_a, (int, float)) and isinstance(_b, (int, float)):
    _sp = abs(_a - _b)
    RESULTS.append({"id": "B15", "manuscript": "6.63× 协议间 Δ_floor 跨度 5.7 dB", "expected": 5.7,
                    "actual": round(_sp, 3), "src": "paired_worldstrat.json|e3_datalayer_paired_ci.json",
                    "path": "-", "status": "PASS" if abs(_sp - 5.7) <= 0.1 else "MISMATCH"})

# R16-F 负样本占比范围 13–23%（RED∪NIR 三 seed 并集）
_fr = []
for _f in ("p1_probav_subgroup.json", "p1_probav_subgroup_nir.json"):
    for _sd in ("42", "123", "7"):
        _v = get(load(_f), f"per_seed/{_sd}/robustness/frac_negative")
        if isinstance(_v, (int, float)):
            _fr.append(_v)
if _fr:
    _lo, _hi = min(_fr), max(_fr)
    _ok = round(_lo * 100) == 13 and round(_hi * 100) == 23
    RESULTS.append({"id": "B16", "manuscript": "负样本占比 13–23%（RED∪NIR 三 seed 并集）",
                    "expected": "13–23%", "actual": f"{_lo*100:.1f}%–{_hi*100:.1f}%",
                    "src": "p1_probav_subgroup{,_nir}.json", "path": "per_seed/*/robustness/frac_negative",
                    "status": "PASS" if _ok else "MISMATCH"})

# ------------------- R2 面板：第二数据集（WorldStrat）站点 B17–B21 -------------------
_pan = load("ws_station_panel.json")
_rows = (_pan or {}).get("rows") or []
_st = [r for r in _rows if r.get("label") != "ALL(pooled)"]
_al = next((r for r in _rows if r.get("label") == "ALL(pooled)"), None)


def _r(v, n=2):
    return None if v is None else round(v, n)


if _st and _al:
    _df = [r["dfloor"] for r in _st]
    _bf = [r["below_floor"] for r in _st]
    RESULTS.append({"id": "B17", "manuscript": "WorldStrat 站间 Δ_floor −4.7 … −3.1 dB",
                    "expected": "−4.7…−3.1", "actual": f"{min(_df):.2f}…{max(_df):.2f}",
                    "src": "ws_station_panel.json", "path": "rows[*].dfloor",
                    "status": "PASS" if _r(min(_df), 1) == -4.7 and _r(max(_df), 1) == -3.1
                    else "MISMATCH"})
    _spr = max(_df) - min(_df)
    RESULTS.append({"id": "B18", "manuscript": "站间地板极差 1.6 dB", "expected": 1.6,
                    "actual": _r(_spr, 2), "src": "ws_station_panel.json", "path": "rows[*].dfloor",
                    "status": "PASS" if _r(_spr, 1) == 1.6 else "MISMATCH"})
    _ok19 = (_r(max(_bf), 1) == -1.3 and _r(min(_bf), 1) == -3.0 and all(v < 0 for v in _bf))
    RESULTS.append({"id": "B19", "manuscript": "model−常数 −1.3 … −3.0 dB（5 站全 <0，无站过地板）",
                    "expected": "−1.3…−3.0 且全 <0", "actual": f"{max(_bf):.2f}…{min(_bf):.2f}",
                    "src": "ws_station_panel.json", "path": "rows[*].below_floor",
                    "status": "PASS" if _ok19 else "MISMATCH"})
    _cells = [s for r in _st for s in r["seeds"]]
    _dm = [s["dm"] for s in _cells]
    _ps = [s["p"] for s in _cells]
    _ok20 = (len(_cells) == 20 and max(_ps) <= 1e-5
             and _r(min(_dm), 2) == 1.28 and _r(max(_dm), 2) == 2.50)
    RESULTS.append({"id": "B20", "manuscript": "20 个 station×seed 格全部显著，增益 +1.28 … +2.50 dB",
                    "expected": "20 格 / maxp ≤ 1e-5", 
                    "actual": f"{len(_cells)} 格 maxp={max(_ps):.1e} "
                              f"range={min(_dm):.2f}…{max(_dm):.2f}",
                    "src": "ws_station_panel.json", "path": "rows[*].seeds[*]",
                    "status": "PASS" if _ok20 else "MISMATCH"})
    _pdm = [s["dm"] for s in _al["seeds"]]
    _pps = [s["p"] for s in _al["seeds"]]
    _mean = sum(_pdm) / len(_pdm)
    _ok21 = abs(_mean - 1.83) <= 0.02 and max(_pps) <= 1.5e-45
    RESULTS.append({"id": "B21", "manuscript": "pooled +1.83 dB（4 seed 均值），p ≤ 1.4e-45",
                    "expected": "1.83 / maxp ≤ 1.4e-45",
                    "actual": f"{_mean:.2f} / maxp={max(_pps):.1e}",
                    "src": "ws_station_panel.json", "path": "rows[ALL(pooled)].seeds[*]",
                    "status": "PASS" if _ok21 else "MISMATCH"})

# ---------------------------------------------------------------- 输出
w = max(len(r["id"]) for r in RESULTS) + 2
npass = sum(1 for r in RESULTS if r["status"] == "PASS")
nmis = sum(1 for r in RESULTS if r["status"] == "MISMATCH")
nno = sum(1 for r in RESULTS if r["status"] == "NO-SOURCE")
print("=" * 110)
for r in RESULTS:
    if r["status"] == "NOTE":
        print(f"  [{r['status']:<9}] {r['id']:<{w}} {r['manuscript'][:62]:<64} {str(r['actual'])[:80]}")
    else:
        exp = r["expected"]
        print(f"  [{r['status']:<9}] {r['id']:<{w}} {r['manuscript'][:62]:<64} exp={exp} got={r['actual']}")
print("=" * 110)
print(f"PASS={npass}  MISMATCH={nmis}  NO-SOURCE={nno}  合计={len(RESULTS)}")

# 畸形 JSON 扫描
import glob
bad = []
for f in sorted(f2 for d in _DIRS for f2 in glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
    try:
        json.load(open(f, encoding="utf-8"))
    except Exception as e:
        bad.append((os.path.relpath(f, EXP), str(e)[:80]))
print(f"\n不可解析 JSON：{len(bad)}")
for f, e in bad:
    print(f"  ✗ {f} -> {e}")
