# -*- coding: utf-8 -*-
"""
Task #34 续: EDiffSR 采样链 oracle 对照 (决定"适配层 vs 模型本身"的最终归因)

原理
----
IRSDE.reverse_sde 的每步只用 score = -noise_pred/sigma_bar(t)。
把模型的 noise_pred 换成解析 oracle(利用真值 x0 的后验均值):
    get_real_score(x_t, x0, t) = -(x_t - mu_bar(x0,t)) / sigma_bar(t)^2
再用同一个 reverse_sde 采样。这条轨迹是后验均值轨迹, 理论上应高保真收敛到 x0。

  A) oracle-score 采样 PSNR 高(>40)  -> 采样链/时间步/我的用法全对,
                                        弱的是训好的模型 -> "不迁移"为合法结论
  B) oracle-score 采样也不行          -> 我的 SDE 用法有 bug, 数字继续封存

附带 C): 单次前向, 检查模型预测噪声与真噪声的相关性(直觉证据)。
"""
import argparse
import os
import sys

REPO = r""
CODES = os.path.join(REPO, r"third_party\_partial_extract\EDiffSR-main\codes")
SISR = os.path.join(CODES, r"config\sisr")
WEIGHTS = os.path.join(REPO, r"third_party\ediffsr\180000_G.pth")
MANIFEST = os.path.join(REPO, r"data\processed\worldstrat_manifest.json")

DLL = r"C:\Windows\System32\vcomp140.dll"
HID = DLL + ".hide"
_hidden = False


def _hide():
    global _hidden
    if os.path.exists(DLL):
        try:
            os.rename(DLL, HID)
            _hidden = True
        except Exception as e:
            print("[guard] 隐藏失败:", e)


def _restore():
    global _hidden
    if _hidden and os.path.exists(HID):
        try:
            os.rename(HID, DLL)
            print("[guard] vcomp140.dll 已还原")
        except Exception as e:
            print("[guard] 还原失败(需手动!):", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _hide()
    try:
        import math
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
        net.load_state_dict(sd, strict=False)

        class _Zero(nn.Module):
            def forward(self, x):
                return 0.0 * x
        net.enhance = _Zero()
        net = net.cuda().eval()

        import json
        manifest = json.load(open(MANIFEST, encoding="utf-8"))
        ds = WorldStratDataset(manifest, lr_size=64, T=1, deterministic=True,
                               hr_per_lr=6.63, hr_div=10000.0, norm="none")
        s = ds[args.idx]
        gt_raw = s["target"].cuda()
        _, H, W = gt_raw.shape
        gmin = gt_raw.amin(dim=(1, 2), keepdim=True)
        gmax = gt_raw.amax(dim=(1, 2), keepdim=True)
        gt_nat = ((gt_raw - gmin) / (gmax - gmin).clamp(min=1e-6)).clamp(0, 1)

        x0 = gt_nat[None]                                   # 真值
        down = F.interpolate(x0, size=(H // 4, W // 4), mode="bicubic", align_corners=True)
        mu = F.interpolate(down, size=(H, W), mode="bicubic", align_corners=True).clamp(0, 1)
        p_bic = calc_psnr(mu.cpu().numpy().astype(np.float64),
                          x0.cpu().numpy().astype(np.float64), data_range=1.0)
        print("[数据] idx=%d 自然化域  双三次x4 基准=%.2f dB" % (args.idx, p_bic))

        # ---------- oracle-score 版 IRSDE ----------
        class OracleIRSDE(IRSDE):
            x0_ref = None

            def score_fn(self, x, t, **kwargs):
                return self.get_real_score(x, self.x0_ref, t)

        def sample(sde_obj, tag):
            torch.manual_seed(args.seed)
            with torch.no_grad():
                sde_obj.set_mu(mu)
                st = sde_obj.noise_state(mu)
                out = sde_obj.reverse_sde(st, save_states=False).clamp(0, 1)
            p = calc_psnr(out.cpu().numpy().astype(np.float64),
                          x0.cpu().numpy().astype(np.float64), data_range=1.0)
            print("  [%s] PSNR=%.2f dB" % (tag, p))
            return p

        print("\n========== A) oracle-score 采样(解析后验均值) ==========")
        sde_o = OracleIRSDE(max_sigma=50, T=100, schedule="cosine", eps=0.005,
                            device="cuda")
        sde_o.x0_ref = x0
        p_oracle = sample(sde_o, "oracle score")

        print("\n========== B) model-score 采样(官方流程) ==========")
        sde_m = IRSDE(max_sigma=50, T=100, schedule="cosine", eps=0.005, device="cuda")
        sde_m.set_model(net)
        p_model = sample(sde_m, "model score")

        print("\n========== C) 模型预测噪声 vs 真噪声(单次前向) ==========")
        torch.manual_seed(args.seed)
        with torch.no_grad():
            n = torch.randn_like(mu)
            t_mid = 50
            A_t = math.exp(-float(sde_m.thetas_cumsum[t_mid]) * float(sde_m.dt))
            sb_t = float(sde_m.sigma_bar(t_mid))
            x_t = mu + (x0 - mu) * A_t + sb_t * n
            pred = sde_m.model(x_t, mu, t_mid)
            # 去掉 batch 维后逐像素相关
            a = pred.flatten().double().cpu().numpy()
            b = n.flatten().double().cpu().numpy()
            corr = float(np.corrcoef(a, b)[0, 1])
            print("  t=%d: sigma_bar=%.4f  corr(pred_noise, true_noise)=%.3f  "
                  "std(pred)=%.3f std(true)=%.3f" % (t_mid, sb_t, corr,
                                                     float(pred.std()), 1.0))

        print("\n================ 判定 ================")
        print("oracle=%.2f dB | model=%.2f dB | bicubic=%.2f dB" % (p_oracle, p_model, p_bic))
        if p_oracle >= 40.0:
            print("-> oracle 采样高保真: 采样链/时间步/用法全部正确。")
            print("   弱的是训好的模型本身 -> EDiffSR 在 WorldStrat 内容上不迁移是合法结论,")
            print("   但必须在论文里注明 oracle 对照已排除实现问题。")
        elif p_oracle >= 33.0:
            print("-> oracle 采样仅部分还原: 采样链大致对, 但 SDE 离散化/参数可能欠佳, 需复核。")
        else:
            print("-> oracle 采样也不行: 我的 SDE 用法有 bug, 数字继续封存。")

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        _restore()


if __name__ == "__main__":
    main()
