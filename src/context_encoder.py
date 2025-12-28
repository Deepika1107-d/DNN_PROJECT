"""
Context encoder used for conditioning the diffusion model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CondEncoder(nn.Module):
    def __init__(self, cond_dim: int = 256):
        super().__init__()
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, cond_dim)
        self.gru = nn.GRU(input_size=cond_dim, hidden_size=cond_dim, batch_first=True)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = ctx.shape
        x = ctx.view(B * K, C, H, W)
        h = self.frame_cnn(x).flatten(1)
        e = self.proj(h).view(B, K, -1)
        _, hN = self.gru(e)
        return hN.squeeze(0)
