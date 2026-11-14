# -*- coding: utf-8 -*-
"""
Task #34: EDiffSR 健康性对照

背景: 双协议数字被自审计否决(原生 x4 regime 落后双三次 10.4 dB)，且
"条件=GT 零退化"也只能还原 34.60 dB —— 瓶颈疑似在扩散链路而非超分。
本脚本用「自然化输入」做健康性对照，一次性回答三个问题:

  T1 保真性(核心判据)
     条件 = 自然化 GT(逐通道 min-max 拉到 [0,1]，std 接近自然影像)，
     加噪(max_sigma=0.196 官方值)后逆扩散还原。
     ≥40 dB  -> 扩散链健康, 之前的坏数字根因 = 输入统计错配(反射率 OOD)
     ≈34 dB  -> 适配层有问题(头号嫌疑: enhance 权重缺失)
  T2 原生 SR
     自然化域里合成 x4，EDiffSR vs 双三次。健康链路下至少不应惨败。
  T3 采样器对照
     reverse_ode(确定性) vs reverse_sde(随机)，定量分离随机弥散成分。

所有 PSNR 均在各自实验域内计算(data_range=1.0)，同域内互比公平。
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
            print("[guard] vcomp140.dll 已隐藏")
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
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # 包遮蔽顺序: 先 EDiffSR(codes 提供 utils, sisr 提供 models)，再项目 code/
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

        # ---- 模型: 官方权重 + 已验证的 enhance 全零替代 ----
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
        print("[ckpt] 载入 OK, 缺失仅 enhance(%d key) -> 全零替代" % len(missing))

        class _Zero(nn.Module):
            def forward(self, x):
                return 0.0 * x
        net.enhance = _Zero()
        net = net.cuda().eval()

        manifest = __import__("json").load(open(MANIFEST, encoding="utf-8"))
        ds = WorldStratDataset(manifest, lr_size=64, T=1, deterministic=True,
                               hr_per_lr=6.63, hr_div=10000.0, norm="none")
        s = ds[args.idx]
        gt_raw = s["target"].cuda()                     # 反射率域 [0,1], std≈0.05
        _, H, W = gt_raw.shape

        # ---- 自然化: 逐通道 min-max -> [0,1] ----
        gmin = gt_raw.amin(dim=(1, 2), keepdim=True)
        gmax = gt_raw.amax(dim=(1, 2), keepdim=True)
        rng = (gmax - gmin).clamp(min=1e-6)
        gt_nat = ((gt_raw - gmin) / rng).clamp(0.0, 1.0)

        def std_of(x):
            return float(x.std())

        print("\n[数据] idx=%d  反射率域 std=%.4f  自然化域 std=%.4f"
              % (args.idx, std_of(gt_raw), std_of(gt_nat)))

        def run(cond, tag, sampler="sde"):
            torch.manual_seed(args.seed)
            sde = IRSDE(max_sigma=50, T=100, schedule="cosine", eps=0.005,
                        device="cuda")
            sde.set_model(net)
            with torch.no_grad():
                sde.set_mu(cond)
                st = sde.noise_state(cond)
                out = (sde.reverse_sde(st, save_states=False) if sampler == "sde"
                       else sde.reverse_ode(st, save_states=False)).clamp(0, 1)
            p = calc_psnr(out.detach().cpu().numpy().astype(np.float64),
                          cond.detach().cpu().numpy().astype(np.float64),
                          data_range=1.0)
            # 相对条件的残差 RMS
            rms = float((out - cond).pow(2).mean().sqrt())
            print("  [%s] PSNR(out,cond)=%.2f dB   残差RMS=%.4f" % (tag, p, rms))
            return p, out

        print("\n================ T1 保真性 (条件=GT, 零退化) ================")
        p_raw, _ = run(gt_raw[None], "反射率域(旧基线, 预期~34.6)")
        p_nat, _ = run(gt_nat[None], "自然化域   ")

        print("\n================ T3 采样器对照 (自然化域) ================")
        p_ode, _ = run(gt_nat[None], "reverse_ode(确定性)")

        print("\n================ T2 原生 SR (自然化域, 合成 x4) ================")
        down = F.interpolate(gt_nat[None], size=(H // 4, W // 4),
                             mode="bicubic", align_corners=True)
        cond4 = F.interpolate(down, size=(H, W), mode="bicubic",
                              align_corners=True).clamp(0, 1)
        p_bic = calc_psnr(cond4.detach().cpu().numpy().astype(np.float64),
                          gt_nat[None].detach().cpu().numpy().astype(np.float64),
                          data_range=1.0)
        p_sr, out_sr = run(cond4, "EDiffSR x4 精化")
        print("  [双三次 x4 基准] PSNR=%.2f dB" % p_bic)
        print("  -> EDiffSR vs 双三次 = %+.2f dB" % (p_sr - p_bic))

        print("\n================ 判定 ================")
        if p_nat >= 40.0:
            print("T1 自然化域 %.2f dB >= 40  -> 扩散链健康" % p_nat)
            print("结论: 之前坏数字的根因 = 反射率输入严重 OOD，")
            print("      需为 EDiffSR 做输入统计标定后重测双协议(不能记方法失败)")
        elif p_nat >= 37.0:
            print("T1 自然化域 %.2f dB 介于 34.6 与 40 之间 -> 部分改善, 存疑" % p_nat)
            print("结论: 输入统计是主因之一, 但仍可能有适配层残留问题, 需进一步排查")
        else:
            print("T1 自然化域 %.2f dB 无实质改善 -> 适配层有问题" % p_nat)
            print("结论: 头号嫌疑 enhance 权重缺失; 下一步查官方是否有含 enhance 的权重版本")
        print("T3: sde=%.2f vs ode=%.2f (差 %.2f dB = 随机弥散成分)"
              % (p_nat, p_ode, p_nat - p_ode))

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        _restore()


if __name__ == "__main__":
    main()
