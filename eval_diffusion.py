"""
Evaluation utilities for baseline and diffusion sampling.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .model_baseline_cnn import psnr_from_mse
from .model_diffusion import sample_x0, ddim_sample_warm, refine_from_ctx_last


@torch.no_grad()
def tc2(pred, ctx_last, tgt):
    return ((pred - ctx_last) - (tgt - ctx_last)).abs().mean().item()


@torch.no_grad()
def eval_baseline(model, loader, device, max_batches: int | None = None):
    model.eval()
    tot = 0
    sum_l1 = 0.0
    sum_mse = 0.0
    sum_tc2 = 0.0
    for bi, batch in enumerate(tqdm(loader, desc="baseline eval", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        ctx = batch["ctx_images"].to(device)
        tgt = batch["target_image"].to(device)
        ctx_last = ctx[:, -1]
        pred = model(ctx)
        l1 = F.l1_loss(pred, tgt).item()
        mse = F.mse_loss(pred, tgt).item()
        tcv = tc2(pred, ctx_last, tgt)
        bs = ctx.size(0)
        tot += bs
        sum_l1 += l1 * bs
        sum_mse += mse * bs
        sum_tc2 += tcv * bs
    avg_l1 = sum_l1 / max(tot, 1)
    avg_mse = sum_mse / max(tot, 1)
    return {"l1": avg_l1, "mse": avg_mse, "psnr": psnr_from_mse(avg_mse), "tc2": sum_tc2 / max(tot, 1)}


@torch.no_grad()
def eval_diffusion_sampling(
    loader,
    device,
    diff_model,
    betas,
    alphas,
    alphas_bar,
    posterior_var,
    mode: str = "ddpm",
    max_batches: int = 25,
):
    tot = 0
    sum_l1 = 0.0
    sum_mse = 0.0
    sum_tc2 = 0.0
    for bi, batch in enumerate(tqdm(loader, desc=f"{mode} eval", leave=False)):
        if bi >= max_batches:
            break
        ctx = batch["ctx_images"].to(device)
        tgt = batch["target_image"].to(device)
        ctx_last = ctx[:, -1]
        if mode == "ddim":
            pred = ddim_sample_warm(ctx, diff_model, alphas_bar, ddim_steps=50, eta=0.0)
        elif mode == "refine":
            pred = refine_from_ctx_last(ctx, diff_model, alphas, alphas_bar, betas, t_start=20)
        else:
            pred = sample_x0(ctx, diff_model, int(alphas_bar.numel()), betas, alphas, alphas_bar, posterior_var)

        l1 = F.l1_loss(pred, tgt).item()
        mse = F.mse_loss(pred, tgt).item()
        tcv = tc2(pred, ctx_last, tgt)
        bs = ctx.size(0)
        tot += bs
        sum_l1 += l1 * bs
        sum_mse += mse * bs
        sum_tc2 += tcv * bs
    avg_l1 = sum_l1 / max(tot, 1)
    avg_mse = sum_mse / max(tot, 1)
    return {"l1": avg_l1, "mse": avg_mse, "psnr": psnr_from_mse(avg_mse), "tc2": sum_tc2 / max(tot, 1)}
