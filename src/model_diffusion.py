"""
Sampling helpers for conditional diffusion (DDPM, DDIM warm start, refine).
"""
from __future__ import annotations

import math
import torch

from .diffusion_model_wrapper import extract


@torch.no_grad()
def sample_x0(ctx, diff_model, T, betas, alphas, alphas_bar, posterior_var):
    device = ctx.device
    sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_ab = torch.sqrt(1.0 - alphas_bar)
    B, K, C, H, W = ctx.shape
    x = torch.randn((B, C, H, W), device=device)
    for step in range(T - 1, -1, -1):
        t = torch.full((B,), step, device=device, dtype=torch.long)
        eps_hat = diff_model(x, t, ctx)
        beta_t = extract(betas, t, x.shape)
        sqrt_recip_a = extract(sqrt_recip_alpha, t, x.shape)
        sqrt_1mab = extract(sqrt_one_minus_ab, t, x.shape)
        mean = sqrt_recip_a * (x - (beta_t / sqrt_1mab) * eps_hat)
        var = extract(posterior_var, t, x.shape)
        if (t == 0).all():
            x = mean
        else:
            x = mean + torch.sqrt(var) * torch.randn_like(x)
    return x.clamp(-1, 1)


@torch.no_grad()
def ddim_sample_warm(ctx, diff_model, alphas_bar, ddim_steps: int = 50, eta: float = 0.0):
    device = ctx.device
    T = int(alphas_bar.numel())
    ddim_ts = torch.linspace(0, T - 1, ddim_steps).long().to(device)
    ddim_ts = torch.unique(ddim_ts).tolist()

    B, K, C, H, W = ctx.shape
    ctx_last = ctx[:, -1]
    t_start = ddim_ts[-1]
    ab_t = alphas_bar[t_start]
    x = torch.sqrt(ab_t) * ctx_last + torch.sqrt(1 - ab_t) * torch.randn_like(ctx_last)

    for idx in range(len(ddim_ts) - 1, 0, -1):
        t = ddim_ts[idx]
        t_prev = ddim_ts[idx - 1]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        eps = diff_model(x, t_batch, ctx)
        ab = alphas_bar[t]
        ab_prev = alphas_bar[t_prev]
        x0_pred = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
        x0_pred = x0_pred.clamp(-1, 1)
        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev))
        noise = torch.randn_like(x) if eta > 0 else 0.0
        dir_xt = torch.sqrt(1 - ab_prev - sigma**2) * eps
        x = torch.sqrt(ab_prev) * x0_pred + dir_xt + sigma * noise
    return x.clamp(-1, 1)


@torch.no_grad()
def refine_from_ctx_last(ctx, diff_model, alphas, alphas_bar, betas, t_start: int = 20):
    device = ctx.device
    B, K, C, H, W = ctx.shape
    ctx_last = ctx[:, -1]
    ab = alphas_bar[t_start]
    x = torch.sqrt(ab) * ctx_last + torch.sqrt(1.0 - ab) * torch.randn_like(ctx_last)

    sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_ab = torch.sqrt(1.0 - alphas_bar)
    for step in range(t_start, -1, -1):
        t = torch.full((B,), step, device=device, dtype=torch.long)
        eps_hat = diff_model(x, t, ctx)
        beta_t = extract(betas, t, x.shape)
        sqrt_recip_a = extract(sqrt_recip_alpha, t, x.shape)
        sqrt_1mab = extract(sqrt_one_minus_ab, t, x.shape)
        x = sqrt_recip_a * (x - (beta_t / sqrt_1mab) * eps_hat)
    return x.clamp(-1, 1)
