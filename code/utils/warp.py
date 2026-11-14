"""Differentiable warping & correlation utilities for the RSA module.

All flow tensors are in *pixel* units with layout [B, 2, H, W] where channel 0 = dx
(horizontal) and channel 1 = dy (vertical). Warping uses align_corners=True so that
the flow is consistent with torch.nn.functional.grid_sample at integer pixel shifts.
"""
import torch
import torch.nn.functional as F


def mesh_grid(B, H, W, device, dtype=torch.float32):
    yy = torch.arange(0, H, device=device, dtype=dtype)
    xx = torch.arange(0, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")  # [H, W]
    grid = torch.stack((gx, gy), dim=0)  # [2, H, W]
    return grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]


def flow_warp(x, flow, padding_mode="border"):
    """Warp x by dense flow (pixel units).

    Args:
        x: [B, C, H, W] feature maps.
        flow: [B, 2, H, W] pixel displacement (dx, dy).
    Returns:
        warped x with the same shape.
    """
    B, C, H, W = x.shape
    base = mesh_grid(B, H, W, x.device, x.dtype)  # [B, 2, H, W]
    grid = base + flow
    grid_n = torch.empty_like(grid)
    grid_n[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
    grid_n[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
    grid_n = grid_n.permute(0, 2, 3, 1)  # [B, H, W, 2]
    return F.grid_sample(x, grid_n, mode="bilinear", align_corners=True,
                         padding_mode=padding_mode)


def compute_cost_volume(f1, f2, radius=4):
    """Limited-displacement correlation cost volume.

    Args:
        f1: [B, C, H, W] reference features (normalised internally).
        f2: [B, C, H, W] target features.
        radius: max displacement in pixels.
    Returns:
        cost: [B, (2*radius+1)**2, H, W] correlation volume.
    """
    f1n = F.normalize(f1, dim=1)
    f2n = F.normalize(f2, dim=1)
    f2p = F.pad(f2n, [radius, radius, radius, radius])  # [B, C, H+2R, W+2R]
    B, C, H, W = f1n.shape
    cost_list = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sl = f2p[:, :, radius + dy:radius + dy + H,
                     radius + dx:radius + dx + W]
            corr = (f1n * sl).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            cost_list.append(corr)
    return torch.cat(cost_list, dim=1)  # [B, (2R+1)^2, H, W]
