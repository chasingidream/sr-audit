#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 v2：定位 crop 管线 bicubic PSNR (19.3) 与 e1 全图 zoom (26.8) 的 7 dB 差距。

v1 教训：A 变体把 LR/HR 传反（作废）；填充方式无关（C=D 逐位相同）；且裁切区内
n=36864=192² 全有效，NaN 根本不进crop。
v2 假设：
  H1 几何错位：整 3× 裁切假设 vs e1 真实比例 zoom -> 亚像素错位掉 PSNR
  H2 灰度定义：RGB 3 波段灰度 vs e1 全波段灰度 -> 单波段对 Δfloor 本就为负
判别设计（每 ROI 同时给 CC 与 PSNR，CC 高=对齐好）：
  F  e1 全图 zoom（修正参数顺序：lr=L8, hr=S2），全波段灰度   —— 应复现 e1 json
  E  crop 整 3×，全波段灰度（其余同 F）                        —— 若 E≈F 则几何无损
  G  crop 整 3×，RGB 灰度                                     —— v1 的 C，作对照
  H  crop 真实比例裁切（HR 框按 fx,fy 取整对应），全波段灰度     —— 直接消错位
"""
import os, glob, numpy as np
import tifffile
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(ROOT, ".l8s2naip")


def load_raw(path):
    with tifffile.TiffFile(path) as t:
        return t.series[0].asarray().astype(np.float64)


def psnr(a, b, dr):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse <= 0 else 10.0 * np.log10(dr * dr / mse)


def metrics(pred, gt):
    dr = float(gt.max() - gt.min()) or 1.0
    p = psnr(pred, gt, dr)
    cc = float(np.corrcoef(pred.flatten(), gt.flatten())[0, 1]) \
        if pred.std() > 1e-8 and gt.std() > 1e-8 else float("nan")
    return p, cc


def full_zoom(lr_g, hr_g):
    """e1 df_core 口径：全图 nanmean 填充 + 真实比例 zoom，全图 common。"""
    lr_nan = np.isnan(lr_g); hr_nan = np.isnan(hr_g)
    lr_fill = np.where(lr_nan, np.nanmean(lr_g), lr_g)
    fx = hr_g.shape[0] / lr_g.shape[0]; fy = hr_g.shape[1] / lr_g.shape[1]
    lr_up = zoom(lr_fill, (fx, fy), order=3)
    mlr = zoom((~lr_nan).astype(np.float64), (fx, fy), order=0) > 0.5
    common = (~hr_nan) & mlr
    gt = hr_g[common]
    p_bic, cc = metrics(lr_up[common], gt)
    p_mean = psnr(np.full_like(gt, gt.mean()), gt, dr_of(gt))
    return p_bic, cc, p_mean, p_bic - p_mean


def dr_of(g):
    d = float(g.max() - g.min())
    return d if d > 0 else 1.0


def crop_eval(l_chw, s_chw, band_sel_l, band_sel_s, exact_ratio=True):
    """crop 口径。band_sel_*: 波段索引列表（None=全波段）。
    exact_ratio=True 用整 3×；False 用真实 fx/fy 取整对应 HR 框。"""
    l = l_chw[list(band_sel_l)] if band_sel_l is not None else l_chw
    s = s_chw[list(band_sel_s)] if band_sel_s is not None else s_chw
    l_g = np.clip(l.mean(axis=0), 0, 1)          # 灰度
    s_g = np.clip(s.mean(axis=0), 0, 1)
    Hl, Wl = l_g.shape; Hs, Ws = s_g.shape
    lr_size = 64
    if exact_ratio:
        sx = sy = 3.0
        out_h = out_w = lr_size * 3
    else:
        sx = Hs / Hl; sy = Ws / Wl
        out_h = int(round(lr_size * sx)); out_w = int(round(lr_size * sy))
    max_i = min(Hl - lr_size, int((Hs - out_h) / sx))
    max_j = min(Wl - lr_size, int((Ws - out_w) / sy))
    if max_i < 0 or max_j < 0:
        return (np.nan,) * 4
    rng = np.random.RandomState(13)
    i = rng.randint(0, max_i + 1); j = rng.randint(0, max_j + 1)
    lc = l_g[i:i + lr_size, j:j + lr_size]
    lc = np.where(np.isnan(lc), np.nanmean(lc) if (~np.isnan(lc)).any() else 0.0, lc)
    if exact_ratio:
        sc = s_g[i * 3:(i + lr_size) * 3, j * 3:(j + lr_size) * 3]
    else:
        r0, r1 = int(round(i * sx)), int(round((i + lr_size) * sx))
        c0, c1 = int(round(j * sy)), int(round((j + lr_size) * sy))
        sc = s_g[r0:r1, c0:c1]
    lr_up = zoom(lc, (sc.shape[0] / lc.shape[0], sc.shape[1] / lc.shape[1]), order=3)
    p_bic, cc = metrics(lr_up, sc)
    p_mean = psnr(np.full_like(sc, sc.mean()), sc, dr_of(sc))
    return p_bic, cc, p_mean, p_bic - p_mean


def main():
    s2_files = sorted(glob.glob(os.path.join(DATADIR, "sentinel2_export_roi_*.tif")),
                      key=lambda p: int(os.path.basename(p)[len("sentinel2_export_roi_"):-4]))
    rows = []
    for k, sp in enumerate(s2_files):
        rid = os.path.basename(sp)[len("sentinel2_export_roi_"):-4]
        lp = os.path.join(DATADIR, "landsat8_export_roi_%s.tif" % rid)
        if not (os.path.exists(lp) and os.path.getsize(lp) > 50000 and os.path.getsize(sp) > 50000):
            continue
        s2 = load_raw(sp); l8 = load_raw(lp)
        lg = np.clip(l8.mean(axis=2), 0, 1)      # e1 全波段灰度
        sg = np.clip(s2.mean(axis=2), 0, 1)
        f_bic, f_cc, f_mean, f_df = full_zoom(lg, sg)          # F: e1 复现
        l_chw = np.transpose(l8, (2, 0, 1))
        s_chw = np.transpose(s2, (2, 0, 1))
        e_bic, e_cc, e_mean, e_df = crop_eval(l_chw, s_chw, None, None, True)     # E: crop 全波段 整3x
        g_bic, g_cc, g_mean, g_df = crop_eval(l_chw, s_chw, (3, 2, 1), (2, 1, 0), True)  # G: crop RGB
        h_bic, h_cc, h_mean, h_df = crop_eval(l_chw, s_chw, None, None, False)    # H: crop 全波段 真实比例
        rows.append(dict(roi=rid, scale=round(s2.shape[0] / l8.shape[0], 3),
                         F_bic=f_bic, F_cc=f_cc, F_df=f_df,
                         E_bic=e_bic, E_cc=e_cc, E_df=e_df,
                         G_df=g_df, H_bic=h_bic, H_cc=h_cc, H_df=h_df))
        print("ROI %3s scale=%.3f | F(e1) bic=%.2f cc=%.3f df=%+.2f | E(crop,all,3x) bic=%.2f cc=%.3f df=%+.2f | "
              "G(crop,rgb) df=%+.2f | H(crop,all,exact) bic=%.2f cc=%.3f df=%+.2f"
              % (rid, rows[-1]["scale"], f_bic, f_cc, f_df, e_bic, e_cc, e_df, g_df, h_bic, h_cc, h_df))

    def col(key):
        v = np.array([r[key] for r in rows], float)
        return v[~np.isnan(v)]

    print("\n=== 汇总（均值 ± sd, n=%d）===" % len(col("F_df")))
    for key, label in [("F_bic", "F(e1全图) bicubic"), ("F_cc", "F CC"),
                       ("F_df", "F Δfloor"),
                       ("E_bic", "E(crop全波段) bicubic"), ("E_cc", "E CC"), ("E_df", "E Δfloor"),
                       ("G_df", "G(crop RGB) Δfloor"),
                       ("H_bic", "H(crop真实比例) bicubic"), ("H_cc", "H CC"), ("H_df", "H Δfloor")]:
        v = col(key)
        print("  %-22s = %+.2f ± %.2f" % (label, v.mean(), v.std(ddof=1)))


if __name__ == "__main__":
    main()
