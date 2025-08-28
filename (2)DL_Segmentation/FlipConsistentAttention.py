import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FCRA(nn.Module):
    """
    Flip-consistent self-attention for radial slices.

    Enforces two symmetries in attention computation:
      • Angular symmetry (θ ↔ π−θ): implemented by flipping along the slice axis S.
      • Spatial symmetry (left ↔ right): implemented by flipping along the width axis W.

    Input : x ∈ ℝ[B, S, C, H, W]
            B = batch, S = number of radial slices (angle index),
            C = channels/features, H/W = spatial dims per slice.
    Output: y ∈ ℝ[B, S, C, H, W]  (same shape as input)
    """
    def __init__(self, dim, use_residual=True, flip_spatial=True):
        super().__init__()
        self.q = nn.Linear(dim, dim)          # Query projection over channel dim
        self.k = nn.Linear(dim, dim)          # Key projection over channel dim
        self.v = nn.Linear(dim, dim)          # Value projection over channel dim
        self.scale = dim ** -0.5              # Scaled dot-product factor (1/√dim)
        self.use_residual = use_residual      # If True, add input (x_flat) as residual
        self.flip_spatial = flip_spatial      # If True, also enforce width (W) flip symmetry
        # self.CrossTimeCausalMHAttns = CrossTimeCausalMHAttn(dim=dim, heads=8)

    def _flip_theta(self, t):
        """
        Flip along the angular/slice axis to map θ → π−θ.

        Args:
            t: tensor of shape [N, S, C] where N = B·H·W after spatial flattening.
        Returns:
            t flipped along dim=1 (the S axis) → shape unchanged.
        """
        # t shape: [N, S, C]  (N = B·H·W)
        return torch.flip(t, dims=[1])  # θ → π−θ

    def _flip_width(self, t, B, S, C, H, W):
        """
        Flip along image width axis W inside each slice, then return to [N,S,C].

        Steps:
          1) reshape [N,S,C] → [B,S,C,H,W]
          2) flip along W
          3) reshape back to [N,S,C]
        """
        # t shape: [N, S, C]  (N = B·H·W)
        t = t.view(B, H, W, S, C).permute(0, 3, 4, 1, 2)  # [B, S, C, H, W]
        t = torch.flip(t, dims=[4])                       # flip width (W)
        return t.permute(0, 3, 4, 1, 2).reshape(-1, S, C) # back to [N, S, C]

    def forward(self, x):           # x: [B, S, C, H, W]
        B, S, C, H, W = x.shape

        # ---- Flatten spatial dims (H, W) so attention is across slices (S) per spatial location ----
        # Result: treat each spatial coordinate (per batch) as an independent sequence over S
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(-1, S, C)  # [B·H·W, S, C]

        # ---- Linear projections to Q, K, V ---------------------------------------------------------
        q = self.q(x_flat)                                   # [N, S, C]

        # Keys/values are taken on the angularly flipped sequence to enforce θ-symmetry
        k = self._flip_theta(self.k(x_flat))                 # [N, S, C]
        v = self._flip_theta(self.v(x_flat))                 # [N, S, C]

        # Optionally, also apply a left–right (W) flip to K and V to enforce spatial symmetry
        if self.flip_spatial:
            k = self._flip_width(k, B, S, C, H, W)           # [N, S, C]
            v = self._flip_width(v, B, S, C, H, W)           # [N, S, C]

        # ---- Scaled dot-product attention over the S dimension -------------------------------------
        # attn: [N, S, S]  where each row attends across angular positions
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out  = torch.matmul(attn, v)                       # [N, S, C]

        # Optional residual connection in the flattened [N,S,C] space
        if self.use_residual:
            out = x_flat + out

        # ---- Restore original tensor layout --------------------------------------------------------
        out = out.view(B, H, W, S, C).permute(0, 3, 4, 1, 2)  # [B, S, C, H, W]

        return out
