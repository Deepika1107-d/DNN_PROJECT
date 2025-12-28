"""
Conditional diffusion model and utilities for next-frame prediction.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = f / f[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(1e-5, 0.999)


def extract(a: torch.Tensor, t: torch.Tensor, xshape: torch.Size) -> torch.Tensor:
    out = a.gather(0, t).view(-1, 1, 1, 1)
    return out.expand(xshape)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


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


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = h + self.emb(emb).view(emb.size(0), -1, 1, 1)
        h = F.relu(self.conv2(h))
        return h + self.skip(x)


class UNetLite(nn.Module):
    def __init__(self, emb_dim: int = 256, base: int = 64):
        super().__init__()
        self.down1 = ResBlock(3, base, emb_dim)
        self.down2 = ResBlock(base, base * 2, emb_dim)
        self.down3 = ResBlock(base * 2, base * 4, emb_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid = ResBlock(base * 4, base * 4, emb_dim)
        self.up3 = ResBlock(base * 4 + base * 4, base * 2, emb_dim)
        self.up2 = ResBlock(base * 2 + base * 2, base, emb_dim)
        self.up1 = ResBlock(base + base, base, emb_dim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x, emb)
        x = self.pool(d1)
        d2 = self.down2(x, emb)
        x = self.pool(d2)
        d3 = self.down3(x, emb)
        x = self.pool(d3)
        m = self.mid(x, emb)
        x = self.up(m)
        x = torch.cat([x, d3], dim=1)
        x = self.up3(x, emb)
        x = self.up(x)
        x = torch.cat([x, d2], dim=1)
        x = self.up2(x, emb)
        x = self.up(x)
        x = torch.cat([x, d1], dim=1)
        x = self.up1(x, emb)
        return self.out(x)


class ConditionalDiffusion(nn.Module):
    def __init__(self, cond_dim: int = 256, time_dim: int = 256):
        super().__init__()
        self.cond_enc = CondEncoder(cond_dim=cond_dim)
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(cond_dim + time_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.unet = UNetLite(emb_dim=256, base=64)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        c = self.cond_enc(ctx)
        te = self.t_embed(t)
        emb = self.emb_mlp(torch.cat([c, te], dim=1))
        return self.unet(x_t, emb)


def build_schedule(T: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(alphas_bar)
    sqrt_1mab = torch.sqrt(1.0 - alphas_bar)
    return betas, alphas, alphas_bar, sqrt_ab, sqrt_1mab


@torch.no_grad()
def p_sample(x_t, t, ctx, diff_model, betas, alphas_bar, sqrt_recip_alpha, sqrt_one_minus_ab, posterior_var):
    eps_hat = diff_model(x_t, t, ctx)
    beta_t = extract(betas, t, x_t.shape)
    sqrt_recip_a = extract(sqrt_recip_alpha, t, x_t.shape)
    sqrt_1mab = extract(sqrt_one_minus_ab, t, x_t.shape)
    mean = sqrt_recip_a * (x_t - (beta_t / sqrt_1mab) * eps_hat)
    var = extract(posterior_var, t, x_t.shape)
    if (t == 0).all():
        return mean
    z = torch.randn_like(x_t)
    return mean + torch.sqrt(var) * z
