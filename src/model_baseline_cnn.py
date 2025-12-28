"""
Baseline GRU model for next-frame prediction (image-only).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEncoder(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.proj(h)


class FrameDecoder(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 256 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), 256, 8, 8)
        return self.net(h)


class GRUNextFrameBaseline(nn.Module):
    def __init__(self, emb_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.enc = FrameEncoder(emb_dim=emb_dim)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, emb_dim)
        self.dec = FrameDecoder(emb_dim=emb_dim)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = ctx.shape
        ctx_flat = ctx.view(B * K, C, H, W)
        emb = self.enc(ctx_flat).view(B, K, -1)
        _, hN = self.gru(emb)
        h = hN.squeeze(0)
        z = self.to_latent(h)
        return self.dec(z)


def psnr_from_mse(mse: float) -> float:
    return 20.0 * math.log10(2.0) - 10.0 * math.log10(max(mse, 1e-12))


@torch.no_grad()
def temporal_consistency(pred: torch.Tensor, ctx_last: torch.Tensor, tgt: torch.Tensor):
    tc1_pred = (pred - ctx_last).abs().mean().item()
    tc1_true = (tgt - ctx_last).abs().mean().item()
    tc2 = ((pred - ctx_last) - (tgt - ctx_last)).abs().mean().item()
    return tc1_pred, tc1_true, tc2
