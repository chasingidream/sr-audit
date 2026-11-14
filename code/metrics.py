"""Evaluation metrics: PSNR / SSIM on the Y (luma) channel, plus a generic
MSE-based fallback for multi-spectral inputs (band-wise mean)."""
import torch
import torch.nn.functional as F


def _rgb_to_ycbcr(img):
    # img: [B,3,H,W] in [0,1]
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def _psnr_channel(x, y, data_range=1.0):
    mse = F.mse_loss(x, y)
    if mse <= 1e-12:
        return torch.tensor(100.0, device=x.device)
    return 20 * torch.log10(torch.tensor(data_range, device=x.device)) - 10 * torch.log10(mse)


def _ssim_channel(x, y, data_range=1.0):
    # simplified SSIM on a single channel
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
          ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim.mean()


@torch.no_grad()
def calc_psnr_ssim(pred, target, y_only=True, data_range=None):
    """pred, target: [B,C,H,W] in [0,1]. Returns (psnr, ssim) tensors.

    data_range: PSNR/SSIM 的满量程。遥感跨传感器数据 HR/LR 辐射定标不一致
    （HR 值域可能仅 1e-3 量级），若固定 data_range=1.0 会严重虚高 PSNR。
    默认取 target 的全局动态范围 (t.max()-t.min())，使 pred/target 在同一量纲
    下比较，跨方法可比。也可由调用方传入整个验证集的全局 data_range 以保证
    不同方法之间绝对可比。
    """
    if y_only and pred.shape[1] == 3:
        p = _rgb_to_ycbcr(pred)
        t = _rgb_to_ycbcr(target)
    else:
        p, t = pred, target
    if data_range is None:
        dr = (t.max() - t.min())
        data_range = float(dr.item()) if dr.item() > 1e-8 else 1.0
    psnr = _psnr_channel(p, t, data_range)
    ssim = _ssim_channel(p, t, data_range)
    return psnr, ssim
