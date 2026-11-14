"""遥感超分完整评测指标（面向一区投稿）。

现有 train.py 只用 PSNR/SSIM，不足以支撑 RSE / ISPRS J / TGRS 投稿：
  * 审稿常要求**光谱保真**（不能只好看，不能偏色/偏谱）；
  * 标准遥感融合指标：SAM（光谱角）、ERGAS（相对全局无量纲误差）、CC（相关系数）、RMSE。

本模块自包含（不 import train.py 相关模块），可在训练进行中安全使用。

指标说明：
  PSNR  : 峰值信噪比（dB），越高越好。data_range 用 GT 真实动态范围，否则虚高。
  SSIM  : 结构相似度，越高越好。
  SAM   : 光谱角映射（度），逐像素光谱向量夹角均值，越低越好（0 = 完全一致）。
  ERGAS : 100/scale * sqrt(mean_b(RMSE_b^2 / mean_b^2))，越低越好。
          scale = 分辨率比（LR像素尺寸/HR像素尺寸），用于归一化，使不同 scale 可比。
  CC    : 逐波段相关系数均值，越高越好（1 = 完全线性相关）。
  RMSE  : 逐波段均方根误差均值，越低越好。
  LPIPS : 感知相似度（越低越好）。需要 alexnet 权重，本机下载慢 —— 默认关闭，
          权重不可用时优雅跳过（返回 None），不阻断评测。
"""
import math

import numpy as np

try:                                     # 允许无 torch 环境用（纯 numpy 路径）
    import torch
    _HAS_TORCH = True
except Exception:                        # pragma: no cover
    torch = None
    _HAS_TORCH = False


def _to_np(x):
    """接受 torch.Tensor / np.ndarray，统一转 float64 numpy。"""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def calc_rmse(pred, gt):
    """逐波段 RMSE 均值。输入 [C,H,W] 或 [B,C,H,W]。"""
    p, g = _to_np(pred), _to_np(gt)
    if p.ndim == 3:
        p, g = p[None], g[None]
    return float(np.sqrt(((p - g) ** 2).mean(axis=(0, 2, 3))).mean())


def calc_sam(pred, gt, eps=1e-8):
    """SAM（光谱角，度）：逐像素比较光谱向量夹角，越低越好。

    输入 [C,H,W] 或 [B,C,H,W]，C 为波段数。
    """
    p, g = _to_np(pred), _to_np(gt)
    if p.ndim == 3:
        p, g = p[None], g[None]
    # 展平为 [N, C]，N = B*H*W
    p = p.transpose(0, 2, 3, 1).reshape(-1, p.shape[1])
    g = g.transpose(0, 2, 3, 1).reshape(-1, g.shape[1])
    dot = (p * g).sum(axis=1)
    np_ = np.linalg.norm(p, axis=1) * np.linalg.norm(g, axis=1)
    cos = np.clip(dot / np.maximum(np_, eps), -1.0, 1.0)
    ang = np.arccos(cos)                      # 弧度
    return float(np.degrees(ang).mean())


def calc_ergas(pred, gt, scale=4, eps=1e-8):
    """ERGAS：越低越好。scale = 分辨率比（LR尺寸/HR尺寸），用于跨尺度归一化。

    ERGAS = (100 / scale) * sqrt( mean_b( RMSE_b^2 / mean_b^2 ) )
    """
    p, g = _to_np(pred), _to_np(gt)
    if p.ndim == 3:
        p, g = p[None], g[None]
    B, C = p.shape[0], p.shape[1]
    mse_b = ((p - g) ** 2).mean(axis=(0, 2, 3))          # [C]
    mean_b = g.mean(axis=(0, 2, 3))                      # [C]
    rel = mse_b / np.maximum(mean_b ** 2, eps)           # [C]
    return float(100.0 / scale * math.sqrt(rel.mean()))


def calc_cc(pred, gt, eps=1e-8):
    """CC：逐波段相关系数均值，越高越好。"""
    p, g = _to_np(pred), _to_np(gt)
    if p.ndim == 3:
        p, g = p[None], g[None]
    B, C = p.shape[0], p.shape[1]
    ccs = []
    for b in range(C):
        a = p[:, b].reshape(-1)
        c = g[:, b].reshape(-1)
        a = a - a.mean()
        c = c - c.mean()
        denom = math.sqrt((a ** 2).sum() * (c ** 2).sum())
        ccs.append(0.0 if denom < eps else float((a * c).sum() / denom))
    return float(np.mean(ccs))


def calc_psnr(pred, gt, data_range=None, eps=1e-8):
    """PSNR(dB)。data_range 必须给 GT 真实动态范围，否则会虚高（本项目已知坑）。"""
    p, g = _to_np(pred), _to_np(gt)
    dr = float(g.max() - g.min()) if data_range is None else float(data_range)
    mse = float(((p - g) ** 2).mean())
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((dr ** 2) / mse)


def calc_ssim(pred, gt, data_range=None, c1=0.01 ** 2, c2=0.03 ** 2):
    """简化 SSIM（高斯窗省略，用均匀 11x11 窗）。够用且与常用实现同量级。
    依赖 skimage 时优先用 skimage 的实现（更标准）。"""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        p, g = _to_np(pred), _to_np(gt)
        if p.ndim == 3:
            p, g = p[None], g[None]
        dr = float(g.max() - g.min()) if data_range is None else float(data_range)
        vals = []
        for i in range(p.shape[0]):
            img = np.moveaxis(p[i], 0, -1)
            ref = np.moveaxis(g[i], 0, -1)
            vals.append(ssim_fn(img, ref, data_range=dr, channel_axis=-1))
        return float(np.mean(vals))
    except Exception:
        # 纯 numpy 回退（单波段/整图近似）
        p, g = _to_np(pred), _to_np(gt)
        if p.ndim == 3:
            p, g = p[None], g[None]
        dr = float(g.max() - g.min()) if data_range is None else float(data_range)
        C1, C2 = (c1 * dr) ** 2, (c2 * dr) ** 2
        mu_p, mu_g = p.mean(), g.mean()
        sig_p, sig_g = p.var(), g.var()
        sig_pg = ((p - mu_p) * (g - mu_g)).mean()
        return float(((2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)) /
                     ((mu_p ** 2 + mu_g ** 2 + C1) * (sig_p + sig_g + C2)))


def evaluate_all(pred, gt, data_range=None, scale=4, with_lpips=False):
    """一次性算全套指标，返回 dict。pred/gt: [C,H,W] 或 [B,C,H,W]，值域任意但需一致。

    data_range: GT 真实动态范围（本项目统一用验证集 GT 全局范围 val_dr，否则 PSNR 虚高）。
    scale: 分辨率比，用于 ERGAS 归一化（本项目标称 4，但 Sentinel-2→SPOT 真实约 6.7，
           投稿前需核对，见 MEMORY.md 的"尺度不严谨"备注）。
    """
    out = {
        "psnr": calc_psnr(pred, gt, data_range),
        "ssim": calc_ssim(pred, gt, data_range),
        "sam_deg": calc_sam(pred, gt),
        "ergas": calc_ergas(pred, gt, scale=scale),
        "cc": calc_cc(pred, gt),
        "rmse": calc_rmse(pred, gt),
    }
    if with_lpips:
        out["lpips"] = calc_lpips(pred, gt)
    return out


def calc_lpips(pred, gt):
    """LPIPS（感知相似度，越低越好）。需要 alexnet 权重，本机下载慢 -> 失败返回 None。"""
    if not _HAS_TORCH:
        return None
    try:
        import lpips as lpips_mod
        fn = lpips_mod.LPIPS(net="alex")
        p = torch.as_tensor(_to_np(pred), dtype=torch.float32)
        g = torch.as_tensor(_to_np(gt), dtype=torch.float32)
        if p.ndim == 3:
            p, g = p[None], g[None]
        # 值域归一到 [-1, 1]（LPIPS 约定）
        p = p * 2.0 - 1.0
        g = g * 2.0 - 1.0
        with torch.no_grad():
            v = fn(p, g)
        return float(v.mean())
    except Exception:
        return None


if __name__ == "__main__":
    # 自检：构造一对带轻微噪声/偏移的假数据，确认各指标方向与量级合理
    rng = np.random.default_rng(0)
    gt = rng.random((3, 64, 64)).astype(np.float32)
    pred = np.clip(gt + rng.normal(0, 0.01, gt.shape).astype(np.float32), 0, 1)
    pred_shift = np.roll(gt, 2, axis=2)          # 2px 平移 -> 应显著变差
    dr = float(gt.max() - gt.min())
    print("clean-ish :", {k: round(v, 4) if v is not None else None
                          for k, v in evaluate_all(pred, gt, data_range=dr).items()})
    print("shifted   :", {k: round(v, 4) if v is not None else None
                          for k, v in evaluate_all(pred_shift, gt, data_range=dr).items()})
