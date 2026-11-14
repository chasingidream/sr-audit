# -*- coding: utf-8 -*-
"""
EDiffSR 探针 (Task #32 前置验证)
目的: 在动刀改适配层之前，先确定三件事能不能成
  1. vcomp140.dll 守卫是否生效 (本 Win 机跑 torch CUDA 的必要条件)
  2. 唯一可用解释器能否 import torch 且拿到 GPU
  3. 官方权重 180000_G.pth 能否被 ConditionalNAFNet 完整载入
只要任一环节失败，Task #32 就该判 NO-GO，不必浪费 GPU 时间。
"""
import os
import sys
import traceback

DLL = r"C:\Windows\System32\vcomp140.dll"
HID = DLL + ".hide"

WEIGHTS = r"third_party\ediffsr\180000_G.pth"
CODES = r"third_party\_partial_extract\EDiffSR-main\codes\config\sisr"

hidden = False

def hide_dll():
    """把 OpenMP 冲突源 vcomp140.dll 移走，跑完必须还原。"""
    global hidden
    if os.path.exists(DLL):
        try:
            os.rename(DLL, HID)
            hidden = True
            print("[guard] vcomp140.dll -> .hide  (已隐藏)")
        except Exception as e:
            print("[guard] 隐藏失败:", e)
    elif os.path.exists(HID):
        print("[guard] 已是隐藏态")
    else:
        print("[guard] 未找到 vcomp140.dll")


def restore_dll():
    global hidden
    if hidden and os.path.exists(HID):
        try:
            os.rename(HID, DLL)
            print("[guard] vcomp140.dll 已还原")
        except Exception as e:
            print("[guard] 还原失败(需手动):", e)


def main():
    hide_dll()
    try:
        import torch
        print("[torch] version =", torch.__version__)
        print("[torch] cuda available =", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("[torch] device =", torch.cuda.get_device_name(0))
            prop = torch.cuda.get_device_properties(0)
            print("[torch] VRAM = %.1f GB" % (prop.total_memory / 1024 ** 3))

        # ---- 权重文件本体 ----
        sd = torch.load(WEIGHTS, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        print("[ckpt] loaded OK, tensors =", len(sd))
        ks = list(sd.keys())[:6]
        print("[ckpt] first keys:", ks)

        # ---- 实例化 ConditionalNAFNet 并严格载入 ----
        sys.path.insert(0, CODES)
        from models.modules.DenoisingNAFNet_arch import ConditionalNAFNet
        net = ConditionalNAFNet(width=64, enc_blk_nums=[14, 1, 1, 1],
                                middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
        missing, unexpected = net.load_state_dict(sd, strict=True)
        print("[model] ConditionalNAFNet strict=True 载入成功")
        print("[model] missing =", len(missing), " unexpected =", len(unexpected))
        n_param = sum(p.numel() for p in net.parameters())
        print("[model] params = %.2f M" % (n_param / 1e6))

        print("\n==== PROBE VERDICT: GO ====")
    except Exception:
        traceback.print_exc()
        print("\n==== PROBE VERDICT: NO-GO (见上方 traceback) ====")
    finally:
        restore_dll()


if __name__ == "__main__":
    main()
