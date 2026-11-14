"""P2-1 ③：构造一个受控的「floor-failure 负对照臂」，直接演示原 thesis 的一半
（audit-fail / floor-failure → 下游有害）。

机制：论文真实 floor-failure 来自跨传感器/多时相**配准失配**。这里用同源 EuroSAT
构造一个「配准失配的 SR 方法」负对照：从同一 LR 做 bicubic 上采样（与 GT 对齐），
再施加固定像素位移 → 与 GT 失配，PSNR 低于 bicubic（floor-failure），且下游分类器
在域内训练后仍因错位而掉点（下游有害）。

这是**受控负对照**（演示机制），不是真实竞品 SOTA；用于闭合
"audit-fail → downstream harmful" 这一半 thesis，配合 ①② 的「两闸门必要性」结论。
"""
import argparse
import numpy as np
from scipy.ndimage import zoom


def bicubic_up(lr, size):
    B, C, h, w = lr.shape
    fx, fy = size[0] / h, size[1] / w
    out = np.empty((B, C, size[0], size[1]), dtype=lr.dtype)
    for b in range(B):
        for c in range(C):
            out[b, c] = zoom(lr[b, c], (fx, fy), order=3)
    return out


def shift_images(imgs, shift, fill=0.0):
    """对每张图施加固定像素位移（失配），边缘 zero-pad。"""
    import scipy.ndimage as ndi
    B, C, H, W = imgs.shape
    out = np.full_like(imgs, fill)
    for b in range(B):
        for c in range(C):
            out[b, c] = ndi.shift(imgs[b, c], shift=(shift, shift), mode="constant", cval=fill)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift", type=int, default=4, help="像素位移（失配强度）")
    ap.add_argument("--variant", default="misreg", choices=["misreg", "meancollapse"])
    args = ap.parse_args()

    lr = np.load(args.lr)
    gt = np.load(args.gt)
    H, W = gt.shape[2], gt.shape[3]
    bic = bicubic_up(lr, (H, W))

    if args.variant == "misreg":
        sr = shift_images(bic, args.shift, fill=0.0)
        tag = "misreg(shift=%d)" % args.shift
    else:  # meancollapse：退化为空间常数（审计的 floor 退化预测器）
        mu = bic.mean(axis=(2, 3), keepdims=True)
        sr = np.broadcast_to(mu, bic.shape).copy()
        tag = "meancollapse"

    np.save(args.out, sr.astype(np.float32))
    # 自检：与 bicubic 的 PSNR 差（应显著为负 → floor-failure）
    def psnr(a, b):
        a = a.astype(np.float64); b = b.astype(np.float64)
        mse = ((a - b) ** 2).mean(axis=(1, 2, 3))
        return 10 * np.log10(1.0 / mse)
    p_bic = psnr(bic, gt); p_sr = psnr(sr, gt)
    print("[neg-ctrl] variant=%s" % tag)
    print("  gain(SR - bicubic) = %.3f dB  (bicubic=%.2f, SR=%.2f) -> floor-failure? %s"
          % (p_sr.mean() - p_bic.mean(), p_bic.mean(), p_sr.mean(),
             "YES" if p_sr.mean() < p_bic.mean() else "NO"))
    print("[saved] %s" % args.out)


if __name__ == "__main__":
    main()
