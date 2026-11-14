"""E1 real cross-sensor 3x arm: S2(10m) <-> L8(30m) native-resolution pairs.

Goal: close the E1 same-scale comparison by adding a REAL cross-sensor 3x arm.
At fixed 3x scale we now have three arms:
  - pure-scale 3x   (SPOT HR self-bicubic, same-sensor/same-day/zero decorr)  -> +11.22 dB  (E1)
  - PROBA-V 3x      (same-sensor, multi-temporal, real)                       -> +1.42 dB  (method-layer real)
  - S2<->L8 3x      (CROSS-sensor, multi-temporal, real)                      -> THIS script

For each ROI we compute the skill-floor gap (recoverable-information gap):
    D_floor = PSNR(bicubic(L8_30m up 3x), S2_10m) - PSNR(mean(S2_10m), S2_10m)
which is data_range-invariant (the DR term cancels in the difference), so it is
directly comparable to the E1 / PROBA-V D_floor numbers that used the same formula.

Band handling (cross-sensor spectral mismatch is part of the decorrelation we want to
measure, but we avoid *inflating* it via naive grayscale averaging of unequal band sets):
  - grayscale   : mean over all bands of each sensor (simple, conservative)
  - band-0      : first band of each sensor (~blue) — single comparable channel
  - band-pairs  : every (L8 band i, S2 band j) pair; report MEAN and BEST(max) across
                  the 6x4 cross-product. Order-agnostic; BEST is an optimistic upper
                  bound (least-decorrelated spectral pair), MEAN is the average budget.

L8/S2 carry ~4% no-data NaN at borders; we fill LR NaN before the cubic zoom and
evaluate MSE only on the S2-valid ∩ L8-valid overlap.
"""
import os, glob, json, numpy as np
import tifffile
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(ROOT, ".l8s2naip")

def load_raw(path):
    with tifffile.TiffFile(path) as t:
        return t.series[0].asarray().astype(np.float64)

def gray(arr):
    g = arr.mean(axis=2) if arr.ndim == 3 else arr
    return np.clip(g, 0.0, 1.0)

def psnr(a, b, dr):
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(dr * dr / mse)

def df_core(lr_chan, hr_chan):
    """NaN-safe D_floor for two single-channel arrays."""
    lr_nan = np.isnan(lr_chan)
    hr_nan = np.isnan(hr_chan)
    if lr_nan.all() or hr_nan.all():
        return float("nan"), 0.0
    lr_fill = np.where(lr_nan, np.nanmean(lr_chan), lr_chan)
    fx = hr_chan.shape[0] / lr_chan.shape[0]
    fy = hr_chan.shape[1] / lr_chan.shape[1]
    lr_up = zoom(lr_fill, (fx, fy), order=3)
    mlr = zoom((~lr_nan).astype(np.float64), (fx, fy), order=0) > 0.5
    common = (~hr_nan) & mlr
    n = int(common.sum())
    if n < 100:
        return float("nan"), 0.0
    gt = hr_chan[common]
    pred = lr_up[common]
    dr = float(gt.max() - gt.min())
    if dr <= 0:
        dr = 1.0
    mp = float(gt.mean())
    p_bic = psnr(pred, gt, dr)
    p_mean = psnr(np.full_like(gt, mp), gt, dr)
    return p_bic - p_mean, n / float(hr_chan.size)

def roi_ids():
    s2 = glob.glob(os.path.join(DATADIR, "sentinel2_export_roi_*.tif"))
    ids = []
    for s in s2:
        rid = os.path.basename(s)[len("sentinel2_export_roi_"):-4]
        l8 = os.path.join(DATADIR, "landsat8_export_roi_%s.tif" % rid)
        if os.path.exists(l8) and os.path.getsize(l8) > 50000 and os.path.getsize(s) > 50000:
            ids.append(rid)
    return sorted(ids, key=lambda x: int(x))

def main():
    ids = roi_ids()
    print("ROIs available: %s" % ids)
    rows = []
    for rid in ids:
        s2p = os.path.join(DATADIR, "sentinel2_export_roi_%s.tif" % rid)
        l8p = os.path.join(DATADIR, "landsat8_export_roi_%s.tif" % rid)
        s2 = load_raw(s2p)           # (Hs, Ws, Cs)
        l8 = load_raw(l8p)           # (Hl, Wl, Cl)
        sg, lg = gray(s2), gray(l8)
        df_g, vf = df_core(lg, sg)
        s2b0 = np.clip(s2[..., 0], 0, 1)
        l8b0 = np.clip(l8[..., 0], 0, 1)
        df_b0, _ = df_core(l8b0, s2b0)
        # band-pair cross product (order-agnostic)
        pair_dfs = []
        n_cl, n_cs = l8.shape[2], s2.shape[2]
        for i in range(n_cl):
            for j in range(n_cs):
                d, _ = df_core(l8[..., i], s2[..., j])
                if np.isfinite(d):
                    pair_dfs.append(d)
        pair_dfs = np.array(pair_dfs)
        df_pair_mean = float(pair_dfs.mean()) if pair_dfs.size else float("nan")
        df_pair_best = float(pair_dfs.max()) if pair_dfs.size else float("nan")
        # psnr absolutes (recompute on valid overlap)
        from scipy.ndimage import zoom as _z
        lr_fill = np.where(np.isnan(lg), np.nanmean(lg), lg)
        fx = sg.shape[0]/lg.shape[0]; fy = sg.shape[1]/lg.shape[1]
        lr_up = _z(lr_fill, (fx, fy), order=3)
        cm = (~np.isnan(sg)) & (_z((~np.isnan(lg)).astype(float),(fx,fy),order=0)>0.5)
        gt = sg[cm]; pr = lr_up[cm]; dr = float(gt.max()-gt.min()) or 1.0
        p_bic = psnr(pr, gt, dr); p_mean = psnr(np.full_like(gt, gt.mean()), gt, dr)
        rows.append({
            "roi": rid,
            "s2_shape": list(s2.shape[:2]), "l8_shape": list(l8.shape[:2]),
            "bands": [n_cl, n_cs],
            "scale_h": round(s2.shape[0]/l8.shape[0], 4),
            "scale_w": round(s2.shape[1]/l8.shape[1], 4),
            "valid_frac": round(vf, 3),
            "psnr_bicubic_db": round(p_bic, 3),
            "psnr_mean_db": round(p_mean, 3),
            "delta_floor_gray_db": round(df_g, 3),
            "delta_floor_b0_db": round(df_b0, 3),
            "delta_floor_bpairs_mean_db": round(df_pair_mean, 3),
            "delta_floor_bpairs_best_db": round(df_pair_best, 3),
        })
        print("ROI %3s  scale=%.3fx%.3f  Dfloor gray=%.2f  b0=%.2f  pairs(mean=%.2f,best=%.2f)  valid=%.1f%%"
              % (rid, rows[-1]["scale_h"], rows[-1]["scale_w"], df_g, df_b0, df_pair_mean, df_pair_best, 100*vf))

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)
    out = {
        "dataset": "namelesscoder/L8-S2-NAIP (native S2 10m <-> L8 30m, 3x cross-sensor)",
        "n_rois": len(rows),
        "rois": rows,
        "summary": {
            "delta_floor_gray_db_mean": round(float(col("delta_floor_gray_db").mean()), 3),
            "delta_floor_gray_db_std": round(float(col("delta_floor_gray_db").std(ddof=1)), 3),
            "delta_floor_gray_db_min": round(float(col("delta_floor_gray_db").min()), 3),
            "delta_floor_gray_db_max": round(float(col("delta_floor_gray_db").max()), 3),
            "delta_floor_b0_db_mean": round(float(col("delta_floor_b0_db").mean()), 3),
            "delta_floor_bpairs_mean_db_mean": round(float(col("delta_floor_bpairs_mean_db").mean()), 3),
            "delta_floor_bpairs_best_db_mean": round(float(col("delta_floor_bpairs_best_db").mean()), 3),
            "psnr_bicubic_db_mean": round(float(col("psnr_bicubic_db").mean()), 3),
            "psnr_mean_db_mean": round(float(col("psnr_mean_db").mean()), 3),
        },
        "reference_arms": {
            "pure_scale_3x_db": 8.549,    # E1 场景匹配 PRIMARY (matched 60 景 + antialias; 旧场景不匹配值 11.221 已废弃)
            "probav_3x_samesensor_multitemp_db": 1.42,  # PROBA-V method-layer real
        },
    }
    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    op = os.path.join(ROOT, "experiments", "e1_real_crosssensor_3x.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    s = out["summary"]
    print("\n=== SUMMARY (n=%d) ===" % len(rows))
    print("  gray  D_floor = %.2f +/- %.2f dB (min %.2f, max %.2f)"
          % (s["delta_floor_gray_db_mean"], s["delta_floor_gray_db_std"],
             s["delta_floor_gray_db_min"], s["delta_floor_gray_db_max"]))
    print("  band0 D_floor = %.2f dB" % s["delta_floor_b0_db_mean"])
    print("  pairs D_floor = mean %.2f / best %.2f dB" % (s["delta_floor_bpairs_mean_db_mean"], s["delta_floor_bpairs_best_db_mean"]))
    print("  reference: pure-scale 3x = +11.22 dB ; PROBA-V 3x same-sensor = +1.42 dB")
    print("  wrote %s" % op)

if __name__ == "__main__":
    main()
