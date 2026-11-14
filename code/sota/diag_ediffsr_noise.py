# -*- coding: utf-8 -*-
"""
决定性诊断: EDiffSR 的误差是「残余噪声」还是「结构性幻觉」?

判据
----
扩散采样是随机的。若误差主要是**随机残余噪声**, 多次独立采样取平均会把噪声 ~1/sqrt(K)
压下去, 平均图的 PSNR 应显著高于单次采样。
若平均后 PSNR 几乎不动, 则误差是**确定性的结构偏差/幻觉**, 属于模型本身的输出。

这决定了 Task #32 的结论该怎么下:
  * 若是噪声 -> 说明我的采样/去噪流程没跑满(T 或 reverse_sde 参数需修), 结果不可用
  * 若是结构 -> 说明 EDiffSR 确实在该数据上产生了确定性退化, 可以作为审计证据
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        except Exception as e:
            print("[guard] 还原失败(需手动!):", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=3600)
    ap.add_argument("--K", type=int, default=4, help="独立采样次数")
    ap.add_argument("--T", type=int, default=100, help="reverse SDE 步数")
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
        net.load_state_dict(sd, strict=False)

        class _Zero(nn.Module):
            def forward(self, x):
                return 0.0 * x
        net.enhance = _Zero()
        net = net.cuda().eval()

        manifest = json_load(MANIFEST)
        ds = WorldStratDataset(manifest, lr_size=64, T=1, deterministic=True,
                               hr_per_lr=6.63, hr_div=10000.0, norm="none")
        s = ds[args.idx]
        gt = s["target"].cuda()
        _, H, W = gt.shape
        down = F.interpolate(gt[None], size=(H // 4, W // 4), mode="bicubic", align_corners=True)
        cond = F.interpolate(down, size=(H, W), mode="bicubic", align_corners=True).clamp(0, 1)

        sde = IRSDE(max_sigma=50, T=args.T, schedule="cosine", eps=0.005, device="cuda")
        sde.set_model(net)

        gt_np = gt[None].detach().cpu().numpy().astype(np.float64)
        bic_np = cond.detach().cpu().numpy().astype(np.float64)

        # ---- 单次 ----
        torch.manual_seed(0)
        with torch.no_grad():
            sde.set_mu(cond)
            st = sde.noise_state(cond)
            single = sde.reverse_sde(st, save_states=False).clamp(0, 1)

        # ---- K 次平均 ----
        acc = torch.zeros_like(cond)
        for k in range(args.K):
            torch.manual_seed(1000 + k)
            with torch.no_grad():
                sde.set_mu(cond)
                st = sde.noise_state(cond)
                acc = acc + sde.reverse_sde(st, save_states=False).clamp(0, 1)
        avg = (acc / args.K).clamp(0, 1)

        single_np = single.detach().cpu().numpy().astype(np.float64)
        avg_np = avg.detach().cpu().numpy().astype(np.float64)

        p_single = calc_psnr(single_np, gt_np, data_range=1.0)
        p_avg = calc_psnr(avg_np, gt_np, data_range=1.0)
        p_bic = calc_psnr(bic_np, gt_np, data_range=1.0)
        # 单次 vs 平均 之间的差异量 = 随机成分
        p_sv_avg = calc_psnr(single_np, avg_np, data_range=1.0)

        print("\n================ 诊断结果 ================")
        print("T(步数)=%d   K(采样次数)=%d" % (args.T, args.K))
        print("bicubic(基准)      = %.2f dB" % p_bic)
        print("EDiffSR 单次采样   = %.2f dB" % p_single)
        print("EDiffSR %d次平均    = %.2f dB" % (args.K, p_avg))
        print("单次 vs 平均之差   = %.2f dB  <-- 随机噪声成分大小" % p_sv_avg)
        # ---- 保真性测试: 条件 = GT 本身(无退化) ----
        # 忠实的扩散去噪器应当高度还原输入。若连这一步都只有 ~30dB,
        # 说明适配层(而非 EDiffSR 本身)有问题。
        torch.manual_seed(0)
        with torch.no_grad():
            sde.set_mu(cond)
            st_self = sde.noise_state(cond)
            selfout = sde.reverse_sde(st_self, save_states=False).clamp(0, 1)
        p_self = calc_psnr(selfout.detach().cpu().numpy().astype(np.float64),
                           cond.detach().cpu().numpy().astype(np.float64),
                           data_range=1.0)
        # 网络输入本身加噪 T=100 后能否还原 -> 这是纯粹的"去噪保真度"
        print("\n[保真性测试] 条件=GT, 加噪后逆扩散还原 vs 条件: %.2f dB" % p_self)
        print("            (>35dB 视为去噪链健康; <=32dB 强烈提示适配层有问题)")

        gain = p_avg - p_single
        print("\n平均相对单次改善 = %+.2f dB" % gain)
        if gain > 2.0:
            print("判定: 误差以【随机残余噪声】为主 -> 采样未收敛, 需修 SDE 步数/参数, 结果暂不可用")
        else:
            print("判定: 误差以【确定性结构偏差】为主 -> 非采样噪声问题")
        print("==========================================")

    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        _restore()


def json_load(p):
    import json
    return json.load(open(p, encoding="utf-8"))


if __name__ == "__main__":
    main()
