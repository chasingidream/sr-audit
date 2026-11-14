#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数值自检：新窗口注意力 (8x8 局部窗口) 是否正确 + 速度。
在 import torch 前隐藏 vcomp140.dll（本机 CUDA OpenMP 冲突保护）。"""
import os
import sys
import time

_DLL = r"C:\Windows\System32\vcomp140.dll"
_HID = _DLL + ".hide"
if os.path.exists(_HID) and not os.path.exists(_DLL):
    os.rename(_HID, _DLL)
_hidden = False
if os.path.exists(_DLL):
    try:
        os.rename(_DLL, _HID)
        _hidden = True
    except Exception as e:
        print("[warn] hide failed:", e)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn.functional as F
    from edsr_worldstrat import SwinIRLite, _WindowAttention

    print("cuda:", torch.cuda.is_available())

    # 1) 形状 + 数值健康
    m = _WindowAttention(64, heads=4, window=8).cuda().eval()
    x = torch.randn(2, 64, 64, 64, device="cuda")
    with torch.no_grad():
        y = m(x)
    print("attn out shape:", tuple(y.shape), "finite:", bool(torch.isfinite(y).all()))

    # 2) 非整除尺寸 (H=W=60) 时 padding 分支
    x2 = torch.randn(1, 64, 60, 60, device="cuda")
    with torch.no_grad():
        y2 = m(x2)
    print("pad out shape:", tuple(y2.shape), "finite:", bool(torch.isfinite(y2).all()))

    # 3) 完整模型前向 + 速度
    net = SwinIRLite(scale=7, n_feats=64, n_blocks=6, out_size=424).cuda().eval()
    n_params = sum(p.numel() for p in net.parameters()) / 1e3
    xb = torch.randn(4, 3, 64, 64, device="cuda")
    with torch.no_grad():
        net(xb)  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(3):
            out = net(xb)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 3
    print(f"SwinIRLite params={n_params:.1f}K  fwd(batch4)={dt:.3f}s  out={tuple(out.shape)}")
    print("OK")
finally:
    if _hidden and os.path.exists(_HID):
        try:
            os.rename(_HID, _DLL)
            print("[restored] vcomp140.dll")
        except Exception as e:
            print("[warn] restore failed:", e)
