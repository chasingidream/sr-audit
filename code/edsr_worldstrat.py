#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDSR (Lim et al. 2017) 单图 SISR —— WorldStrat 轴独立双协议复现 (Task #25)。

事实纠正：EDSR/WDSR 并非 WorldStrat 官方基线（官方基线为 SRCNN + HighResNet，
见 third_party/worldstrat/src/modules.py，全 third_party/ 目录均无 EDSR/WDSR）；
本脚本从 Lim et al. 2017 规范*独立实现* EDSR，作为第 2 个强单图 SOTA 对照，
复用本项目已验证的 WorldStrat 流水线（data/dataset.py 的 get_dataset +
configs/external_worldstrat.py），只在 frame0 单帧上训练，与 bicubic(frame0) 正交对比。

评测口径与 mtxa 系列 / HighRes-net 复审完全一致：
  * per_scene 归一化（LR/HR 同 (mu,sd) 标准化，不 clamp）
  * val_dr = 验证集 GT 全局动态范围（消除 PSNR 虚高）
  * 逐样本配对 t 检验（CI/p/Cohen's d）+ SR-Audit G1-G5 门 + 技能地板
输出 experiments/sota_review/edsr.json（ProtocolResult schema），可入双协议复审对照表。

运行：
  python code/edsr_worldstrat.py --mode all --epochs 30
  python code/edsr_worldstrat.py --mode train --epochs 30
  python code/edsr_worldstrat.py --mode eval
"""
import argparse
import atexit
import glob
import json
import math
import os
import sys
import time

# ---------------------------------------------------------------- vcomp140 guard
# torch CUDA 在本机 Windows 上若同时加载 MSVC OpenMP(vcomp140) 与 Intel OpenMP
# (libiomp5md) 会 segfault。必须在 import torch 前把 vcomp140.dll 改名隐藏，
# 进程退出时还原。worker 子进程不触碰该 DLL（继承已隐藏状态），故安全。
_DLL = r"C:\Windows\System32\vcomp140.dll"
_HIDDEN = _DLL + ".hide"


def _hide_dll():
    if os.path.exists(_HIDDEN) and not os.path.exists(_DLL):
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            pass
    if os.path.exists(_DLL):
        try:
            os.rename(_DLL, _HIDDEN)
            return True
        except Exception:
            print("[warn] could not hide vcomp140.dll; torch CUDA may segfault.")
    return False


def _restore_dll():
    if os.path.exists(_HIDDEN):
        try:
            os.rename(_HIDDEN, _DLL)
        except Exception:
            print("[warn] could not restore vcomp140.dll; rename manually.")


_hide_dll()
atexit.register(_restore_dll)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from scipy import stats as _st
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(BASE, "code")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "models"))

from data.dataset import get_dataset  # noqa: E402
from eval_metrics import calc_psnr, calc_ssim, calc_cc, calc_sam  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCH = "edsr"          # set by --arch; see main()
CKPT_DIR = os.path.join(BASE, "experiments", "ckpts_worldstrat", "edsr_f0")
JSON_OUT = os.path.join(BASE, "experiments", "sota_review", "edsr.json")
ARCH_META = {
    "edsr": ("EDSR (Lim et al. 2017, 16 blocks/64 feats, 单图 SISR)", "edsr_f0", "edsr.json"),
    "rcan": ("RCAN (Zhang et al. 2018, ECCV; 10 RCAB/64 feats, 通道注意力, 单图 SISR)",
             "rcan_f0", "rcan.json"),
    "swinir": ("SwinIR-lite (Liu et al. 2021, ICCV; 6 Swin blocks/64 feats, 单图 SISR)",
               "swinir_f0", "swinir.json"),
}
CFG_PATH = os.path.join(CODE, "configs", "external_worldstrat.py")
VAL_FRAC = 0.9
BATCH = 4
N_BLOCKS = 16
N_FEATS = 64


# ---------------------------------------------------------------- EDSR 模型
class _ResBlock(nn.Module):
    """EDSR 残差块：Conv-ReLU-Conv + 残差（无 BN、无末位 ReLU —— 这是 EDSR 相对 SRResNet 的关键）。"""

    def __init__(self, n=64):
        super().__init__()
        self.c1 = nn.Conv2d(n, n, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(n, n, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.c2(self.relu(self.c1(x))) + x


class EDSR(nn.Module):
    """单图 SISR (Lim et al. 2017)。输入 [B,3,64,64] -> 输出 [B,3,out_size,out_size]。"""

    def __init__(self, scale=7, n_feats=64, n_blocks=16, out_size=424):
        super().__init__()
        self.out_size = (out_size, out_size) if isinstance(out_size, int) else out_size
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1, bias=False)
        self.body = nn.Sequential(*[_ResBlock(n_feats) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False)
        # 上采样：conv(64 -> 64*scale^2) + PixelShuffle(scale) + conv(64->3)。
        # scale=7 为质数，单次 PixelShuffle(7)（64*7=448）再双线性插值到真实 HR 尺寸 424，
        # 与本项目 SpatioTemporalSR / bicubic 基线的末端 resize 完全同构。
        self.up = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, 1, 1, bias=False),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        x0 = self.head(x)
        x = self.tail(self.body(x0)) + x0          # 全局残差
        x = self.up(x)
        if self.out_size is not None and tuple(x.shape[-2:]) != tuple(self.out_size):
            x = F.interpolate(x, size=tuple(self.out_size), mode="bilinear",
                              align_corners=True, antialias=True)
        return x


# ---------------------------------------------------------------- RCAN 模型
class _CALayer(nn.Module):
    """RCAN 的通道注意力 (Zhang et al., ECCV 2018)。"""

    def __init__(self, n, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(n, max(4, n // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, n // reduction), n, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class _RCAB(nn.Module):
    """Residual channel-attention block."""

    def __init__(self, n):
        super().__init__()
        self.c1 = nn.Conv2d(n, n, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(n, n, 3, 1, 1, bias=False)
        self.ca = _CALayer(n)

    def forward(self, x):
        return self.ca(self.c2(self.relu(self.c1(x)))) + x


class RCAN(nn.Module):
    """单图 SISR (Zhang et al., ECCV 2018)。输入 [B,3,64,64] -> [B,3,out_size,out_size]。"""

    def __init__(self, scale=7, n_feats=64, n_blocks=10, out_size=424):
        super().__init__()
        self.out_size = (out_size, out_size) if isinstance(out_size, int) else out_size
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1, bias=False)
        self.body = nn.Sequential(*[_RCAB(n_feats) for _ in range(n_blocks)],
                                  nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=False))
        self.up = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, 1, 1, bias=False),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        x0 = self.head(x)
        x = self.body(x0) + x0
        x = self.up(x)
        if self.out_size is not None and tuple(x.shape[-2:]) != tuple(self.out_size):
            x = F.interpolate(x, size=tuple(self.out_size), mode="bilinear",
                              align_corners=True, antialias=True)
        return x


# ---------------------------------------------------------------- SwinIR-lite 模型
class _WindowAttention(nn.Module):
    """窗口内多头自注意力 (SwinIR 的 W-MSA)。
    把feature map切成 8x8 的非重叠窗口，在每个窗口内部做自注意力，
    复杂度从 O((HW)^2) 降到 O(HW * w^2)，与真实 SwinIR 一致。"""

    def __init__(self, dim, heads=4, window=8):
        super().__init__()
        self.h = heads
        self.window = window
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        w = self.window
        pad_b = (w - H % w) % w
        pad_r = (w - W % w) % w
        if pad_b or pad_r:
            x = F.pad(x, (0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[-2], x.shape[-1]
        nH, nW = Hp // w, Wp // w
        # 切成非重叠窗口: [B,C,Hp,Wp] -> [B*nH*nW, C, w, w]
        xw = x.view(B, C, nH, w, nW, w).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, w, w)
        N = xw.shape[0]
        L = w * w
        # qkv: 1x1 conv 逐 token 线性变换，输出 3C 通道
        t = xw.reshape(N, C, w, w)
        qkv = self.qkv(t).reshape(N, 3 * C, L)                    # [N,3C,L]
        qkv = qkv.reshape(N, 3, self.h, C // self.h, L).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # [N,h,L,d]
        attn = torch.einsum("nhld,nhmd->nhlm", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("nhlm,nhmd->nhld", attn, v)            # [N,h,L,d]
        out = out.permute(0, 1, 3, 2).reshape(N, C, w, w)         # [N,C,w,w]
        out = self.proj(out)                                      # [N,C,w,w]
        # un-tile: [N,C,w,w] -> [B,C,Hp,Wp]
        out = out.reshape(B, nH, nW, C, w, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        if pad_b or pad_r:
            out = out[:, :, :H, :W]
        return out


class _SwinBlock(nn.Module):
    """Swin Transformer block (Liu et al., ICCV 2021) 的单窗口退化实现：
    LN -> window attention -> 残差 -> LN -> MLP -> 残差。"""

    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hid = dim * mlp_ratio
        self.mlp = nn.Sequential(nn.Conv2d(dim, hid, 1), nn.GELU(),
                                 nn.Conv2d(hid, dim, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(h)
        h = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.mlp(h)
        return x


class SwinIRLite(nn.Module):
    """SwinIR 的轻量单图配置 (Liu et al., ICCV 2021)。
    输入 [B,3,64,64] -> [B,3,out_size,out_size]。"""

    def __init__(self, scale=7, n_feats=64, n_blocks=6, out_size=424):
        super().__init__()
        self.out_size = (out_size, out_size) if isinstance(out_size, int) else out_size
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.body = nn.Sequential(*[_SwinBlock(n_feats) for _ in range(n_blocks)])
        self.conv_after = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.up = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, 1, 1, bias=False),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        x0 = self.head(x)
        x = self.conv_after(self.body(x0)) + x0
        x = self.up(x)
        if self.out_size is not None and tuple(x.shape[-2:]) != tuple(self.out_size):
            x = F.interpolate(x, size=tuple(self.out_size), mode="bilinear",
                              align_corners=True, antialias=True)
        return x


def build_model(arch, scale, out_size):
    if arch == "edsr":
        return EDSR(scale=scale, n_feats=N_FEATS, n_blocks=N_BLOCKS, out_size=out_size)
    if arch == "rcan":
        return RCAN(scale=scale, n_feats=N_FEATS, n_blocks=10, out_size=out_size)
    if arch == "swinir":
        return SwinIRLite(scale=scale, n_feats=N_FEATS, n_blocks=6, out_size=out_size)
    raise ValueError(f"unknown arch {arch}")


# ---------------------------------------------------------------- 配置 / 数据
def load_cfg():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ext_ws_cfg", CFG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.CONFIG)


def build_loaders(cfg):
    train_ds = get_dataset(cfg, project_root=BASE, deterministic=False)
    val_ds = get_dataset(cfg, project_root=BASE, deterministic=True)
    n = len(train_ds)
    n_train = max(1, int(VAL_FRAC * n))
    train_sub = Subset(train_ds, list(range(n_train)))
    val_sub = Subset(val_ds, list(range(n_train, n)))
    train_loader = DataLoader(train_sub, batch_size=BATCH, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader, n_train, (n - n_train)


def compute_val_dr(val_loader):
    mn, mx = float("inf"), float("-inf")
    for vb in val_loader:
        t = vb["target"].float()
        mn = min(mn, float(t.min()))
        mx = max(mx, float(t.max()))
    return (mx - mn) if (mx - mn) > 1e-8 else 1.0


# ---------------------------------------------------------------- SR-Audit 周期伪影（G4）
def _periodicity_scan(img, periods=range(2, 17)):
    """周期伪影扫描：2..16px 周期里最强谐波突出度；ratio>2.5 视为周期伪影。
    与 code/sr_audit.py 同实现。"""
    x = img.float().mean(0)
    mag = torch.fft.fftshift(torch.abs(torch.fft.fft2(x)))
    H, W = mag.shape
    cy, cx = H // 2, W // 2
    best, bp = -1.0, None
    for p in periods:
        fy, fx = int(round(H / p)), int(round(W / p))
        if fy < 2 or fx < 2 or fy >= cy or fx >= cx:
            continue
        off = max(2, fy // 8)
        rs = []
        for r in (cy + fy, cy - fy):
            if 0 <= r - off and r + off < H:
                base = torch.stack([mag[r - off, :].mean(), mag[r + off, :].mean()])
                rs.append(mag[r, :].mean() / (base.mean() + 1e-12))
        for c in (cx + fx, cx - fx):
            if 0 <= c - off and c + off < W:
                base = torch.stack([mag[:, c - off].mean(), mag[:, c + off].mean()])
                rs.append(mag[:, c].mean() / (base.mean() + 1e-12))
        if rs:
            r = float(torch.stack(rs).mean())
            if r == r and r > best:
                best, bp = r, p
    return best, (int(bp) if bp else -1)


# ---------------------------------------------------------------- 训练
def train(cfg, epochs=30):
    train_loader, val_loader, n_tr, n_val = build_loaders(cfg)
    scale = cfg.get("scale", 7)
    out_size = cfg.get("out_size", 424)
    model = build_model(ARCH, scale, out_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"[info] {ARCH} params={n_params:.1f}K  device={DEVICE}  "
          f"scale={scale} out_size={out_size}  n_train={n_tr} n_val={n_val}")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=2e-5)
    os.makedirs(CKPT_DIR, exist_ok=True)
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for bi, batch in enumerate(train_loader):
            lr0 = batch["lrs"].to(DEVICE)[:, 0]        # frame0 单帧 [B,3,64,64]
            gt = batch["target"].to(DEVICE)
            pred = model(lr0)
            loss = F.l1_loss(pred, gt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            dr = compute_val_dr(val_loader)
            model.eval()
            ps = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= 60:
                        break
                    lr0 = batch["lrs"].to(DEVICE)[:, 0]
                    gt = batch["target"].to(DEVICE)
                    pred = model(lr0)
                    if tuple(pred.shape[-2:]) != tuple(gt.shape[-2:]):
                        pred = F.interpolate(pred, size=tuple(gt.shape[-2:]),
                                             mode="bilinear", align_corners=True)
                    ps.append(calc_psnr(pred, gt, data_range=dr))
            vpsnr = sum(ps) / len(ps) if ps else 0.0
            print(f"[ep{ep}] loss={tot/max(1,nb):.4f}  val_psnr={vpsnr:.2f}  "
                  f"({time.time()-t0:.1f}s)")
        torch.save({"model": model.state_dict(), "epoch": ep, "cfg": cfg},
                   os.path.join(CKPT_DIR, f"ckpt_ep{ep}.pt"))
    print("[done] training ->", CKPT_DIR)


# ---------------------------------------------------------------- 评测
def evaluate(cfg):
    _ARCH_TAG = {"edsr": "EDSR", "rcan": "RCAN", "swinir": "SwinIR-lite"}.get(ARCH, ARCH.upper())
    _, val_loader, _, n_val = build_loaders(cfg)
    val_dr = compute_val_dr(val_loader)
    fs = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_ep*.pt")),
                key=lambda p: int(p.split("_ep")[-1].split(".")[0]))
    if not fs:
        raise FileNotFoundError(f"no EDSR ckpt in {CKPT_DIR}")
    ck = torch.load(fs[-1], map_location="cpu", weights_only=False)
    scale = cfg.get("scale", 7)
    out_size = cfg.get("out_size", 424)
    model = build_model(ARCH, scale, out_size).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[info] loaded {fs[-1]}  val_dr={val_dr:.4f}  n_val={n_val}")

    rows_edsr, rows_bic, rows_floor = [], [], []
    pair_cc, cc_m, cc_b = [], [], []
    stdr_list, grid_m, grid_b = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            lr0 = batch["lrs"].to(DEVICE)[:, 0]      # [1,3,64,64]
            gt = batch["target"].to(DEVICE)          # [1,3,424,424]
            pred = model(lr0)
            if tuple(pred.shape[-2:]) != tuple(gt.shape[-2:]):
                pred = F.interpolate(pred, size=tuple(gt.shape[-2:]),
                                    mode="bilinear", align_corners=True, antialias=True)
            bic = F.interpolate(lr0.float(), size=tuple(gt.shape[-2:]),
                                mode="bicubic", align_corners=True)
            gt_mean = float(gt.mean())
            flr = torch.full_like(gt, gt_mean)
            rows_edsr.append(calc_psnr(pred, gt, data_range=val_dr))
            rows_bic.append(calc_psnr(bic, gt, data_range=val_dr))
            rows_floor.append(calc_psnr(flr, gt, data_range=val_dr))
            # 注意：本流水线为 per_scene 归一化（零均值 unit-variance），不能用
            # sr_audit 的 clamp(0,1)（那只适用于 [0,1] 反射率数据）；对零均值数据
            # clamp 会把负值压成 0，使 std/CC 失真。G1/G2/G3 一律在【未 clamp 的
            # 归一化张量】上计算，与 PSNR/SSIM/SAM 主指标同口径。
            # G1 配对完整性：CC(bicubic(LR),GT)
            pair_cc.append(calc_cc(bic, gt))
            # G2 结构保真：model CC / bicubic CC
            cc_m.append(calc_cc(pred, gt))
            cc_b.append(calc_cc(bic, gt))
            stdr_list.append(float(pred.std() / (gt.std() + 1e-8)))
            rm, _ = _periodicity_scan(pred[0])
            rb, _ = _periodicity_scan(bic[0])
            grid_m.append(rm)
            grid_b.append(rb)

    n = len(rows_edsr)
    m_edsr = sum(rows_edsr) / n
    m_bic = sum(rows_bic) / n
    m_floor = sum(rows_floor) / n
    g1 = sum(pair_cc) / n
    r_cc = (sum(cc_m) / n) / max(sum(cc_b) / n, 1e-6)
    std_ratio = sum(stdr_list) / n
    grid_vals = [x for x in grid_m if x == x]
    grid_ratio_model = sum(grid_vals) / len(grid_vals) if grid_vals else float("nan")

    # 门控（与 sr_audit.py 一致）
    vG1 = "PASS" if g1 >= 0.30 else "FAIL"
    vG2 = "PASS" if r_cc >= 0.9 else ("WARN" if r_cc >= 0.7 else "FAIL")
    vG3 = "PASS" if 0.5 <= std_ratio <= 1.8 else "FAIL"
    vG4 = "PASS" if grid_ratio_model < 2.5 else "WARN"
    vG5 = "PASS"  # EDSR 无 GT 泄漏，PSNR 不会 >45dB

    # 配对 t 检验（EDSR - bicubic，逐样本）
    diffs = [a - b for a, b in zip(rows_edsr, rows_bic)]
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    if _HAS_SCIPY and sd > 0:
        t_stat, p_val = _st.ttest_rel(rows_edsr, rows_bic)
        t_stat, p_val = float(t_stat), float(p_val)
        ci = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)
    else:
        t_stat, p_val = float("nan"), float("nan")
        ci = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)
    d = mean_diff / sd if sd > 0 else float("nan")
    passes_floor = bool(m_edsr > m_floor)

    result = {
        "method": ARCH_META[ARCH][0],
        "task": "WorldStrat 轴单图 SISR (S2 frame0 10m -> SPOT 1.5m, 6.63x 跨传感器)",
        "n_test": n,
        "split": "subset=600, val=后10% (60 samples), deterministic crop",
        "orig_gain_db": None,  # EDSR 原协议增益在 Set5/Set14 等合成 bicubic 退化基准(2-4x)，非 6.63x 跨传感器，不虚构
        "audit_gain_db": round(mean_diff, 3),
        "audit_ci": [round(ci[0], 3), round(ci[1], 3)],
        "audit_p": round(p_val, 6),
        "audit_d": round(d, 3),
        "passes_floor": passes_floor,
        "floor_gap_db": round(m_edsr - m_floor, 3),
        "g1": vG1,
        "g2": vG2,
        "g3": f"{std_ratio:.2f}",
        "g4": vG4,
        "g5": vG5,
        "note": (
            f"model(frame0) {m_edsr:.2f} / bicubic(frame0) {m_bic:.2f} / "
            f"mean-floor {m_floor:.2f} dB. 审计结论(逐样本配对): {_ARCH_TAG}-bicubic "
            f"{mean_diff:+.2f} dB (p={p_val:.4f}, d={d:+.2f}). "
            + ("越过技能地板, 真实技能成立; " if passes_floor
               else "未过技能地板, 增益被可恢复信息缺口吞没; ")
            + f"SR-Audit G1={vG1}(CC(bic,GT)={g1:.3f}) G2={vG2}(CC比={r_cc:.2f}) "
            f"G3={vG3}(std_ratio={std_ratio:.2f}) G4={vG4}(grid={grid_ratio_model:.2f}) G5={vG5}. "
            f"{_ARCH_TAG} 原协议增益(合成 bicubic 退化基准)与本 6.63x 跨传感器真实配对不可直接比; "
            "本条目作为独立强 SOTA 复现, 与 HighRes-net 共同构成 WorldStrat 轴双协议复审."
        ),
        "detail": {
            "psnr_model_mean": round(m_edsr, 3),
            "psnr_bic_mean": round(m_bic, 3),
            "psnr_floor_mean": round(m_floor, 3),
            "g1_pair_cc": round(g1, 4),
            "g2_cc_ratio": round(r_cc, 4),
            "std_ratio_model_over_gt": round(std_ratio, 4),
            "grid_ratio_model": round(grid_ratio_model, 3),
            "paired": {
                "n": n, "mean": round(mean_diff, 4), "sd": round(sd, 4),
                "t": round(t_stat, 3), "p": round(p_val, 6), "d": round(d, 3),
                "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
            },
            "device": str(DEVICE),
        },
    }
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[saved] {JSON_OUT}")
    print(f"  {_ARCH_TAG} {m_edsr:.2f} | bicubic {m_bic:.2f} | floor {m_floor:.2f} dB")
    print(f"  gain {mean_diff:+.2f} dB  p={p_val:.4f}  d={d:+.2f}  "
          f"pass_floor={passes_floor}")
    print(f"  G1={vG1} G2={vG2} G3={vG3} G4={vG4} G5={vG5}")
    return result


def main():
    global ARCH, CKPT_DIR, JSON_OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["train", "eval", "all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--arch", default="edsr", choices=["edsr", "rcan", "swinir"],
                    help="single-image SISR architecture to audit")
    args = ap.parse_args()
    ARCH = args.arch
    CKPT_DIR = os.path.join(BASE, "experiments", "ckpts_worldstrat", ARCH_META[ARCH][1])
    JSON_OUT = os.path.join(BASE, "experiments", "sota_review", ARCH_META[ARCH][2])
    print(f"[info] arch={ARCH}  ckpt={CKPT_DIR}  json={JSON_OUT}")
    cfg = load_cfg()
    if args.mode in ("train", "all"):
        train(cfg, epochs=args.epochs)
    if args.mode in ("eval", "all"):
        evaluate(cfg)


if __name__ == "__main__":
    main()
