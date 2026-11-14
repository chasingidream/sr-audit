"""Loss functions for spatiotemporal SR.

Combines:
  * fidelity: L1 / Charbonnier on HR reference
  * perceptual: LPIPS (alex) with graceful fallback if weights are unavailable
  * frequency: Fourier magnitude+phase loss (dependency-free)
  * wavelet: 3-level Haar detail loss (dependency-free)
  * self-alignment regulariser: flow smoothness + cyclic consistency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))


def l1(x, y):
    return F.l1_loss(x, y)


def fourier_loss(x, y):
    """Compare amplitude & phase in the Fourier domain (high-freq aware)."""
    fx = torch.fft.fft2(x, norm="ortho")
    fy = torch.fft.fft2(y, norm="ortho")
    ax, ay = torch.abs(fx), torch.abs(fy)
    px, py = torch.angle(fx), torch.angle(fy)
    return F.l1_loss(ax, ay) + F.l1_loss(px, py)


def _haar_kernel():
    k = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.float32) / 2.0
    return k


def haar_wavelet_loss(x, y, levels=3):
    """Compare detail (high-pass) subbands via separable Haar decomposition."""
    loss = 0.0
    xa, ya = x, y
    for _ in range(levels):
        # average-pooling approximates the low-pass; details = input - upsampled low
        xl = F.avg_pool2d(xa, 2)
        yl = F.avg_pool2d(ya, 2)
        xu = F.interpolate(xl, size=xa.shape[-2:], mode="nearest")
        yu = F.interpolate(yl, size=ya.shape[-2:], mode="nearest")
        dx, dy = xa - xu, ya - yu  # high-frequency detail
        loss = loss + F.l1_loss(dx, dy)
        xa, ya = xl, yl
    return loss / levels


class PerceptualLoss(nn.Module):
    """LPIPS wrapper that is initialised lazily (on first call) and degrades
    gracefully when the pretrained weights cannot be fetched on this network."""

    def __init__(self, net="alex"):
        super().__init__()
        self.net = net
        self._fn = None
        self.disabled = False

    def _ensure(self):
        if self._fn is None and not self.disabled:
            try:
                import lpips  # lazy import
                self._fn = lpips.LPIPS(net=self.net).eval()
                for p in self._fn.parameters():
                    p.requires_grad_(False)
            except Exception as e:  # pragma: no cover - environment dependent
                print(f"[warn] LPIPS unavailable ({e}); perceptual loss disabled.")
                self.disabled = True

    def forward(self, x, y):
        self._ensure()
        if self._fn is None:
            return torch.tensor(0.0, device=x.device)
        self._fn = self._fn.to(x.device)
        # LPIPS expects inputs in [-1, 1]
        return self._fn(2 * x - 1, 2 * y - 1).mean()


class CompositeLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_l1 = cfg.get("w_l1", 1.0)
        self.w_char = cfg.get("w_char", 0.0)
        self.w_perc = cfg.get("w_perc", 0.1)
        self.w_fourier = cfg.get("w_fourier", 0.1)
        self.w_wavelet = cfg.get("w_wavelet", 0.1)
        self.w_flow = cfg.get("w_flow", 1.0)
        self.w_cyclic = cfg.get("w_cyclic", 0.1)
        self.perc = PerceptualLoss(cfg.get("lpips_net", "alex"))

    def forward(self, pred, target, aux=None, model=None, lrs=None):
        loss = 0.0
        parts = {}
        if self.w_l1:
            loss += self.w_l1 * l1(pred, target); parts["l1"] = self.w_l1 * l1(pred, target).item()
        if self.w_char:
            loss += self.w_char * charbonnier(pred, target); parts["char"] = self.w_char * charbonnier(pred, target).item()
        if self.w_perc:
            lp = self.w_perc * self.perc(pred, target); parts["perc"] = lp.item(); loss += lp
        if self.w_fourier:
            lf = self.w_fourier * fourier_loss(pred, target); parts["fourier"] = lf.item(); loss += lf
        if self.w_wavelet:
            lw = self.w_wavelet * haar_wavelet_loss(pred, target); parts["wavelet"] = lw.item(); loss += lw
        if aux is not None and self.w_flow:
            flows = aux.get("flows", [])
            if flows:  # use_rsa 关闭时 flows 为空，跳过 flow 正则
                flow_smooth = sum(flow.abs().mean() for flow in flows)
                loss += self.w_flow * 1e-3 * flow_smooth
                parts["flow"] = (self.w_flow * 1e-3 * flow_smooth).item()
        if model is not None and lrs is not None and self.w_cyclic:
            lc = self.w_cyclic * model.cyclic_loss(lrs); parts["cyclic"] = lc.item(); loss += lc
        parts["total"] = loss.item()
        return loss, parts
