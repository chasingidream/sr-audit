# -*- coding: utf-8 -*-
"""SEN2VENµS 跨站点零训练 Δ_floor 验证（回应三轮审稿攻击点1）。

目的：把 ceiling regime 从"单一站点（FGMANAUS, n=13）上的可证伪预测"升级为
"多站点验证的模式"。审稿人指出：零训练验证只需 bicubic vs mean，成本低。

做法（口径与 eval_paired_floor.py 逐位一致，保证与已报道 +7.99 dB 可比）：
  * get_dataset(type=sen2venus, site=<SITE>)，最后 10% 作为"验证"切片（与主文一致）；
  * val_dr = 该切片 target 的全局动态范围（protocol A 约定）；
  * bicubic = F.interpolate(lrs[:,0], scale 2×, bicubic, align_corners=False)；
  * mean   = GT 的空间均值常数图；
  * 逐样本 evaluate_all（PSNR/SSIM/CC/SAM），paired_test 给 bicubic − mean 的
    配对 t / CI / Cohen's d。
  * FGMANAUS 先跑作为管线自检（应复现 +7.99 dB 附近），SO2 为跨站点新测量。

用法：
  python code/s2v_crossstation_dfloor.py --sites FGMANAUS,SO2
输出：experiments/s2v_crossstation_dfloor.json
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code"))

OUT = os.path.join(REPO, "experiments", "s2v_crossstation_dfloor.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="FGMANAUS,SO2")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--root", default=None,
                    help="站点数据根目录（含 {SITE}.zip 或 {SITE}/ 子目录）；"
                         "缺省用 data/external/sen2venus（Zenodo 布局）")
    args = ap.parse_args()

    import _dll_guard
    _dll_guard.hide()
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Subset

        from data.dataset import get_dataset  # noqa: E402
        from eval_paired_floor import paired_test, evaluate_all, METRICS  # noqa: E402

        results = {}
        for site in [s.strip() for s in args.sites.split(",") if s.strip()]:
            ds_cfg = {"type": "sen2venus", "site": site,
                      "bands": "b2b3b4b8", "div": 10000.0, "norm": "global"}
            if args.root:
                ds_cfg["root"] = args.root
            cfg = {
                "T": 1, "scale": 2, "lr_size": 64,
                "dataset": ds_cfg,
            }
            ds = get_dataset(cfg, project_root=REPO, deterministic=True)
            n = len(ds)
            n_train = max(1, min(n - 1, int((1.0 - args.val_frac) * n)))
            val = Subset(ds, list(range(n_train, n)))
            print(f"[{site}] 总样本={n}  验证切片={len(val)}", flush=True)
            loader = DataLoader(val, batch_size=4, shuffle=False, num_workers=0)

            mn, mx = float("inf"), float("-inf")
            for vb in loader:
                t = vb["target"].float()
                mn, mx = min(mn, float(t.min())), max(mx, float(t.max()))
            val_dr = (mx - mn) if mx - mn > 1e-8 else 1.0
            print(f"[{site}] val_dr={val_dr:.6f}", flush=True)

            rows_bic, rows_mean = [], []
            with torch.no_grad():
                for batch in loader:
                    lrs = batch["lrs"].float()
                    gt = batch["target"].float()
                    mean_map = gt.mean(dim=(2, 3), keepdim=True).expand_as(gt)
                    bic = F.interpolate(lrs[:, 0], size=gt.shape[-2:],
                                        mode="bicubic", align_corners=False)
                    for i in range(gt.shape[0]):
                        g = gt[i]; b = bic[i]; m = mean_map[i]
                        rows_bic.append({k: float(evaluate_all(b, g, data_range=val_dr)[k])
                                         for k in METRICS})
                        rows_mean.append({k: float(evaluate_all(m, g, data_range=val_dr)[k])
                                          for k in METRICS})

            paired = paired_test(rows_bic, rows_mean)   # bicubic − mean
            df = paired.get("psnr", {})
            diffs = np.array([r["psnr"] - m["psnr"]
                              for r, m in zip(rows_bic, rows_mean)])
            frac_pos = float((diffs > 0).mean())
            print(f"[{site}] Δ_floor = {df.get('mean_diff'):+.4f} dB  "
                  f"CI={df.get('ci95')}  p={df.get('p'):.3e}  n={len(diffs)}  "
                  f"frac>0={frac_pos:.3f}", flush=True)
            results[site] = dict(
                n_total=n, n_val=len(val), val_dr=val_dr,
                dfloor=paired, frac_positive=frac_pos,
                mean_bicubic_psnr=float(np.mean([r["psnr"] for r in rows_bic])),
                mean_psnr=float(np.mean([r["psnr"] for r in rows_mean])),
            )

        out_path = args.out
        prev = {}
        if os.path.exists(out_path):
            try:
                prev = json.load(open(out_path, encoding="utf-8"))
            except Exception:  # noqa: BLE001
                prev = {}
        sites_all = dict(prev.get("sites", {}))
        sites_all.update(results)          # 只更新本次站点，保留既有站点
        out = dict(
            purpose="SEN2VENµS 跨站点零训练 Δ_floor 验证（ceiling regime 在多站点上的泛化）。"
                    "FGMANAUS = Zenodo v2 原始数据；SO2 与新站点经 HF TACO 镜像直读提取。",
            convention="protocol A：per-sample PSNR, shared data range = val 分片动态范围, "
                       "bicubic align_corners=False, 无裁剪（与 Methods 标准约定一致）",
            val_frac=args.val_frac, sites=sites_all,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_path}  ({len(sites_all)} sites: {sorted(sites_all)})", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        _dll_guard.restore()


if __name__ == "__main__":
    main()
