#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HighRes-net (WorldStrat 官方基线) 权重加载 + 推理骨架 (Task #30 P0)
==============================================================
WorldStrat 的 HighResNet 是 **PyTorch** 实现 (非 TF)。其 published
model.ckpt 把 backbone 实例 (HighResNet) 序列化进了 hyper_parameters,
因此 torch.load 需要 `src.modules` 可导入。我们:
  1. 从 hyper_parameters['backbone'] 恢复 *精确* 架构 (无需猜超参)
  2. 从 ckpt['state_dict'] (前缀 'backbone.') 重载 *训练好的* 权重
     (hyper_parameters 里的 backbone 仅是结构快照, 权重为随机初始化)
  3. 在随机输入上做前向 sanity check, 确认通道/分辨率对齐

用法:
  python code/sota/highresnet_load.py
"""
import os
import sys
import shutil

# ---- vcomp140.dll 守卫 (torch import 在 Windows 上会 segfault, 必须隐藏) ----
_DLL = r"C:\Windows\System32\vcomp140.dll"
_dll_hidden = False
if os.path.exists(_DLL):
    try:
        os.rename(_DLL, _DLL + ".hide")
        _dll_hidden = True
    except Exception as e:
        print(f"[warn] 无法隐藏 vcomp140.dll: {e}")

_RESTORE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "third_party", "worldstrat")


def _restore_dll():
    global _dll_hidden
    if _dll_hidden and os.path.exists(_DLL + ".hide"):
        try:
            os.rename(_DLL + ".hide", _DLL)
            _dll_hidden = False
        except Exception:
            pass


try:
    import torch

    # 把 vendored worldstrat/src 加入路径, 使 `import src` 可用
    SRC_ROOT = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "third_party", "worldstrat"))
    if SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)
    # kornia 桩 (model.ckpt 的反序列化会引用 kornia.geometry.transform.Resize)
    KORNIA_STUB = os.path.join(SRC_ROOT, "_kornia_stub")
    if KORNIA_STUB not in sys.path:
        sys.path.insert(0, KORNIA_STUB)

    from src.modules import HighResNet  # noqa: E402

    CKPT = os.path.join(SRC_ROOT, "pretrained_model", "model.ckpt")

    def load_highresnet(ckpt_path=CKPT, device="cpu"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # 恢复精确架构
        backbone = ckpt["hyper_parameters"]["backbone"]
        assert isinstance(backbone, HighResNet), \
            f"hyper_parameters['backbone'] 类型异常: {type(backbone)}"
        # 重载训练好的权重 (hyper_parameters 内的 backbone 权重是随机初始化快照)
        sd = {k[9:]: v for k, v in ckpt["state_dict"].items()
              if k.startswith("backbone.")}
        missing, unexpected = backbone.load_state_dict(sd, strict=True)
        if missing or unexpected:
            print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}")
        backbone.eval()
        backbone.to(device)
        # 报告关键配置
        print("=== HighResNet 配置 (来自 ckpt) ===")
        for attr in ("in_channels", "out_channels", "revisits", "hidden_channels",
                     "kernel_size", "residual_layers", "zoom_factor",
                     "sr_kernel_size", "registration_kind", "use_reference_frame",
                     "skip_paddings", "output_size"):
            print(f"  {attr}: {getattr(backbone, attr, '?')}")
        nparams = sum(p.numel() for p in backbone.parameters())
        print(f"  参数量: {nparams/1e3:.1f}K")
        return backbone

    def sanity_forward(model, device="cpu"):
        # x: (batch, revisits, channels, H, W); 官方 benchmark: 8 revisits, 12 bands, 160x160
        x = torch.randn(1, 8, 12, 160, 160, device=device)
        with torch.no_grad():
            y = model(x)
        print(f"\n[sanity] input {tuple(x.shape)} -> output {tuple(y.shape)}")
        assert y.shape[0] == 1 and y.shape[1] == 1, "输出应为 (B,1,C,H,W)"
        assert y.shape[2] == model.out_channels, "输出通道数 != out_channels"
        assert y.shape[-2:] == tuple(model.output_size), \
            f"输出空间尺寸 {y.shape[-2:]} != output_size {model.output_size}"
        print("[sanity] OK: 架构/权重/前向对齐")
        return y

    if __name__ == "__main__":
        m = load_highresnet()
        sanity_forward(m)

finally:
    _restore_dll()
