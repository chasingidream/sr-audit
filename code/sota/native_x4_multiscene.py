# -*- coding: utf-8 -*-
"""EDiffSR 原生 x4 regime 多场景健康探针（回应二轮审稿攻击点4：n=2 -> n=12）。

背景：二轮审稿指出 "n=2 的 probe 不足以支撑 'same weights lose 7.4 dB in their
own native regime'"，建议扩到 10–20 scene（inference-only，无训练成本）。

本脚本只跑 health_check_ediffsr.py 的 T2 部分（原生合成 x4：bicubic 下/上采样
vs EDiffSR 精化），在 WorldStrat manifest 上均匀抽 12 个场景：
  * 每场景：自然化 GT -> 合成 x4 条件 -> 双三次 PSNR + EDiffSR 一次 SDE 采样 PSNR
  * 汇总：逐场景差值 + 配对 t 检验（eval_paired_floor.paired_test 同口径）
  * 用途定位不变：这是"排除 adapter 完全坏掉"的健康探针，不作为 quantitative claim

用法：
  python code/sota/native_x4_multiscene.py --n 12 --seed 0
输出：experiments/sota_review/ediffsr_native_x4_multiscene.json
"""
import argparse
import json
import os
import sys

REPO = r""
CODES = os.path.join(REPO, r"third_party\_partial_extract\EDiffSR-main\codes")
SISR = os.path.join(CODES, r"config\sisr")
WEIGHTS = os.path.join(REPO, r"third_party\ediffsr\180000_G.pth")
MANIFEST = os.path.join(REPO, r"data\processed\worldstrat_manifest.json")
OUT = os.path.join(REPO, r"experiments\sota_review\ediffsr_native_x4_multiscene.json")

DLL = r"C:\Windows\System32\vcomp140.dll"
HID = DLL + ".hide"
_hidden = False


def _hide():
    global _hidden
    if os.path.exists(DLL):
        try:
            os.rename(DLL, HID)
            _hidden = True
            print("[guard] vcomp140.dll 已隐藏", flush=True)
        except Exception as e:
            print("[guard] 隐藏失败:", e, flush=True)


def _restore():
    global _hidden
    if _hidden and os.path.exists(HID):
        try:
            os.rename(HID, DLL)
            _hidden = False
            print("[guard] vcomp140.dll 已还原", flush=True)
        except Exception as e:
            print("[guard] 还原失败:", e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _hide()
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        sys.path.insert(0, CODES)
        sys.path.insert(0, SISR)
        from utils.sde_utils import IRSDE
        from models.modules.DenoisingNAFNet_arch import ConditionalNAFNet

        sys.path.insert(0, os.path.join(REPO, "code"))
        from data.dataset import WorldStratDataset
        from eval_metrics import calc_psnr

        import data.dataset as dsmod
        _orig = dsmod.read_tif
        dsmod.read_tif = lambda p, b=None, div=None: _orig(p, b, div if div is not None else 10000.0)

        net = ConditionalNAFNet(img_channel=3, width=64, enc_blk_nums=[14, 1, 1, 1],
                                middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1], upscale=1)
        sd = torch.load(WEIGHTS, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = net.load_state_dict(sd, strict=False)
        assert not unexpected, "ckpt 有多余 key: %r" % sorted(unexpected)[:5]
        assert set(missing) <= {"enhance.rcab.%d.%s" % (i, k)
                                for i in (0, 2, 3)
                                for k in ("weight", "bias")} | {
            "enhance.rcab.3.attention.1.weight", "enhance.rcab.3.attention.1.bias",
            "enhance.rcab.3.attention.3.weight", "enhance.rcab.3.attention.3.bias"}, \
            "缺失集超出 enhance: %r" % sorted(missing)
        print("[ckpt] 载入 OK, 缺失仅 enhance(%d key) -> 全零替代" % len(missing), flush=True)

        class _Zero(nn.Module):
            def forward(self, x):
                return 0.0 * x
        net.enhance = _Zero()
        net = net.cuda().eval()

        manifest = json.load(open(MANIFEST, encoding="utf-8"))
        ds = WorldStratDataset(manifest, lr_size=64, T=1, deterministic=True,
                               hr_per_lr=6.63, hr_div=10000.0, norm="none")
        n_total = len(ds)
        idxs = np.linspace(0, n_total - 1, args.n).astype(int).tolist()
        print(f"[data] manifest 场景总数={n_total}, 抽样 idx={idxs}", flush=True)

        rows = []
        for k, idx in enumerate(idxs):
            s = ds[int(idx)]
            gt_raw = s["target"].cuda()
            _, H, W = gt_raw.shape
            gmin = gt_raw.amin(dim=(1, 2), keepdim=True)
            gmax = gt_raw.amax(dim=(1, 2), keepdim=True)
            rng = (gmax - gmin).clamp(min=1e-6)
            gt_nat = ((gt_raw - gmin) / rng).clamp(0.0, 1.0)

            down = F.interpolate(gt_nat[None], size=(H // 4, W // 4),
                                 mode="bicubic", align_corners=True)
            cond4 = F.interpolate(down, size=(H, W), mode="bicubic",
                                  align_corners=True).clamp(0, 1)
            p_bic = calc_psnr(cond4.detach().cpu().numpy().astype(np.float64),
                              gt_nat[None].detach().cpu().numpy().astype(np.float64),
                              data_range=1.0)

            torch.manual_seed(args.seed)
            sde = IRSDE(max_sigma=50, T=100, schedule="cosine", eps=0.005, device="cuda")
            sde.set_model(net)
            with torch.no_grad():
                sde.set_mu(cond4)
                st = sde.noise_state(cond4)
                out = sde.reverse_sde(st, save_states=False).clamp(0, 1)
            p_sr = calc_psnr(out.detach().cpu().numpy().astype(np.float64),
                             gt_nat[None].detach().cpu().numpy().astype(np.float64),
                             data_range=1.0)
            rows.append(dict(idx=int(idx), psnr_bicubic_x4=round(p_bic, 4),
                             psnr_ediffsr_x4=round(p_sr, 4),
                             diff_sr_minus_bic=round(p_sr - p_bic, 4)))
            print(f"[{k+1}/{len(idxs)}] idx={idx}  bicubic={p_bic:.2f}  "
                  f"ediffsr={p_sr:.2f}  diff={p_sr-p_bic:+.2f} dB", flush=True)

        # ---- 配对统计（与 eval_paired_floor 同口径）----
        diffs = np.array([r["diff_sr_minus_bic"] for r in rows])
        from scipy import stats as sps
        n = len(diffs)
        md = float(diffs.mean()); sd = float(diffs.std(ddof=1)); se = sd / np.sqrt(n)
        t = md / se
        p = float(2 * sps.t.sf(abs(t), n - 1))
        ci = [round(md - 1.96 * se, 4), round(md + 1.96 * se, 4)]
        print(f"\n[pair] n={n}  mean_diff={md:+.4f} dB  sd={sd:.4f}  "
              f"CI95={ci}  t={t:+.2f}  p={p:.3e}", flush=True)

        out = dict(
            purpose="回应二轮攻击点4：EDiffSR 原生 x4 regime 健康探针扩到 n=%d（原 n=2）" % n,
            role="diagnostic health probe（排除 adapter 完全坏掉），非 quantitative claim",
            weights=WEIGHTS, seed=args.seed, n=n, indices=idxs, per_scene=rows,
            paired=dict(mean_diff=round(md, 4), sd=round(sd, 4), ci95=ci,
                        t=round(t, 2), p=p),
            conclusion=("同一权重在其原生合成 x4 regime 中 %s 双三次（均值 %+.2f dB）"
                        % ("仍落后于" if md < 0 else "优于", md)),
        )
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[saved] {OUT}", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        _restore()


if __name__ == "__main__":
    main()
