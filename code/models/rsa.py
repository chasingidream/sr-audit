"""RSA - Registration-free Self-Alignment module.

Core differentiator vs. STANet (ISPRS J 2026):
  * STANet uses an affine Spatial Alignment Module (SAM) + window Temporal Alignment.
  * RSA drops the affine assumption and learns a *dense deformable* flow directly
    from a correlation cost volume (no external registration, non-linear geometry).
  * A cyclic-consistency regulariser lets the alignment be trained even without
    strictly paired HR supervision.

The module operates on same-resolution feature maps, so in the full model it is
used to align the multi-temporal LR frames to a reference frame before fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warp import compute_cost_volume, flow_warp


class FlowHead(nn.Module):
    def __init__(self, in_ch, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class RSAModule(nn.Module):
    def __init__(self, feat_dim, radius=4, hidden=64, iters=1, use_conf=True):
        super().__init__()
        self.radius = radius
        self.iters = iters
        self.use_conf = use_conf
        cv_ch = (2 * radius + 1) ** 2
        self.flow_head = FlowHead(cv_ch, hidden)
        if use_conf:
            self.conf_head = nn.Sequential(
                nn.Conv2d(cv_ch, hidden, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 1, 3, 1, 1),
                nn.Sigmoid(),
            )

    def estimate_flow(self, f_ref, f_t):
        cost = compute_cost_volume(f_ref, f_t, self.radius)
        flow = self.flow_head(cost)
        for _ in range(self.iters - 1):  # iterative RAFT-style refinement
            warped = flow_warp(f_t, flow)
            cost = compute_cost_volume(f_ref, warped, self.radius)
            flow = flow + self.flow_head(cost)
        return flow, cost

    def forward(self, f_ref, f_t):
        flow, cost = self.estimate_flow(f_ref, f_t)
        aligned = flow_warp(f_t, flow)
        conf = self.conf_head(cost) if self.use_conf else None
        return aligned, flow, conf

    def cyclic_loss(self, f_ref, f_t):
        """Self-supervised alignment regulariser: align t->ref then ref->aligned,
        enforce round-trip consistency. No HR pairing required."""
        flow_tr, _ = self.estimate_flow(f_ref, f_t)      # warp f_t -> f_ref
        warped_t = flow_warp(f_t, flow_tr)               # ~ f_ref
        flow_rt, _ = self.estimate_flow(warped_t, f_ref)  # warp f_ref -> warped_t
        warped_back = flow_warp(warped_t, flow_rt)
        recon = (warped_back - f_ref).abs().mean()
        inverse = (flow_tr + flow_rt).abs().mean()
        return recon + 0.1 * inverse
