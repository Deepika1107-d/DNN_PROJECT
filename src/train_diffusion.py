"""
Train baseline GRU model or conditional diffusion model for next-frame prediction.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataloader import TemporalImageDataset, build_index_table
from .model_baseline_cnn import GRUNextFrameBaseline, psnr_from_mse, temporal_consistency
from .diffusion_model_wrapper import ConditionalDiffusion, build_schedule, extract


def build_splits(train_raw, test_raw, val_frac: float, seed: int, K: int):
    unique_ids = np.unique(np.array(train_raw["story_id"]))
    train_ids, val_ids = train_test_split(unique_ids, test_size=val_frac, random_state=seed, shuffle=True)
    storyid_to_idx = {sid: i for i, sid in enumerate(train_raw["story_id"])}
    train_story_indices = [storyid_to_idx[sid] for sid in train_ids]
    val_story_indices = [storyid_to_idx[sid] for sid in val_ids]

    train_index = build_index_table(train_raw, train_story_indices, K=K)
    val_index = build_index_table(train_raw, val_story_indices, K=K)
    test_index = build_index_table(test_raw, list(range(len(test_raw))), K=K)
    return train_index, val_index, test_index


def run_epoch_baseline(model, loader, device, train: bool):
    model.train(train)
    tot = 0
    sum_l1 = 0.0
    sum_mse = 0.0
    sum_tc2 = 0.0
    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        ctx = batch["ctx_images"].to(device, non_blocking=True)
        tgt = batch["target_image"].to(device, non_blocking=True)
        ctx_last = ctx[:, -1]
        pred = model(ctx)
        l1 = F.l1_loss(pred, tgt)
        mse = F.mse_loss(pred, tgt)
        loss = l1 + 0.1 * mse
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            _, _, tc2 = temporal_consistency(pred, ctx_last, tgt)
        bs = ctx.size(0)
        tot += bs
        sum_l1 += l1.item() * bs
        sum_mse += mse.item() * bs
        sum_tc2 += tc2 * bs
        pbar.set_postfix({"l1": sum_l1 / tot, "mse": sum_mse / tot, "tc2": sum_tc2 / tot})
    avg_l1 = sum_l1 / max(tot, 1)
    avg_mse = sum_mse / max(tot, 1)
    avg_psnr = psnr_from_mse(avg_mse)
    avg_tc2 = sum_tc2 / max(tot, 1)
    return {"l1": avg_l1, "mse": avg_mse, "psnr": avg_psnr, "tc2": avg_tc2}


def train_baseline(train_loader, val_loader, device, epochs: int, save_path: Path):
    model = GRUNextFrameBaseline(emb_dim=256, hidden_dim=256).to(device)
    global optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_val = 1e9
    for epoch in range(1, epochs + 1):
        tr = run_epoch_baseline(model, train_loader, device, train=True)
        va = run_epoch_baseline(model, val_loader, device, train=False)
        print(f"\nEpoch {epoch}")
        print(f"  train: l1={tr['l1']:.4f} mse={tr['mse']:.4f} psnr={tr['psnr']:.2f} tc2={tr['tc2']:.4f}")
        print(f"  val:   l1={va['l1']:.4f} mse={va['mse']:.4f} psnr={va['psnr']:.2f} tc2={va['tc2']:.4f}")
        if va["l1"] < best_val:
            best_val = va["l1"]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "opt_state": optimizer.state_dict(), "val": va}, str(save_path))
            print("  saved", str(save_path))


def train_diffusion(train_loader, val_loader, device, epochs: int, T_steps: int, save_path: Path):
    diff_model = ConditionalDiffusion(cond_dim=256, time_dim=256).to(device)
    optimizer = torch.optim.AdamW(diff_model.parameters(), lr=2e-4, weight_decay=0.01)
    betas, alphas, alphas_bar, sqrt_ab, sqrt_1mab = build_schedule(T_steps, device)

    def train_epoch(epoch: int):
        diff_model.train()
        tot = 0
        sum_loss = 0.0
        pbar = tqdm(train_loader, desc=f"diff train {epoch}", leave=False)
        for batch in pbar:
            ctx = batch["ctx_images"].to(device)
            x0 = batch["target_image"].to(device)
            B = x0.size(0)
            t = torch.randint(0, T_steps, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(x0)
            x_t = extract(sqrt_ab, t, x0.shape) * x0 + extract(sqrt_1mab, t, x0.shape) * eps
            eps_hat = diff_model(x_t, t, ctx)
            loss = F.mse_loss(eps_hat, eps)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(diff_model.parameters(), 1.0)
            optimizer.step()
            bs = B
            tot += bs
            sum_loss += loss.item() * bs
            pbar.set_postfix({"mse_eps": sum_loss / tot})
        return {"mse_eps": sum_loss / max(tot, 1)}

    @torch.no_grad()
    def eval_epoch(epoch: int, max_batches: int = 120):
        diff_model.eval()
        tot = 0
        sum_loss = 0.0
        pbar = tqdm(val_loader, desc=f"diff val {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            if i >= max_batches:
                break
            ctx = batch["ctx_images"].to(device)
            x0 = batch["target_image"].to(device)
            B = x0.size(0)
            t = torch.randint(0, T_steps, (B,), device=device, dtype=torch.long)
            eps = torch.randn_like(x0)
            x_t = extract(sqrt_ab, t, x0.shape) * x0 + extract(sqrt_1mab, t, x0.shape) * eps
            eps_hat = diff_model(x_t, t, ctx)
            loss = F.mse_loss(eps_hat, eps)
            tot += B
            sum_loss += loss.item() * B
            pbar.set_postfix({"mse_eps": sum_loss / tot})
        return {"mse_eps": sum_loss / max(tot, 1)}

    best_val = 1e9
    for epoch in range(1, epochs + 1):
        tr = train_epoch(epoch)
        va = eval_epoch(epoch)
        print(f"\nEpoch {epoch} (diffusion noise-pred)")
        print(f"  train mse_eps={tr['mse_eps']:.6f}")
        print(f"  val   mse_eps={va['mse_eps']:.6f}")
        if va["mse_eps"] < best_val:
            best_val = va["mse_eps"]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"epoch": epoch, "model_state": diff_model.state_dict(), "opt_state": optimizer.state_dict(), "val": va, "T_STEPS": T_steps},
                str(save_path),
            )
            print("  saved", str(save_path))


def main():
    parser = argparse.ArgumentParser(description="Train baseline or diffusion model for Project 2.")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_steps", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--mode", choices=["baseline", "diffusion"], default="baseline")
    parser.add_argument("--t_steps", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("daniel3303/StoryReasoning", cache_dir=args.cache_dir)
    train_raw = ds["train"]
    test_raw = ds["test"]

    train_index, val_index, _ = build_splits(train_raw, test_raw, args.val_frac, args.seed, args.k_steps)
    train_ds = TemporalImageDataset(train_raw, train_index, K=args.k_steps, img_size=args.img_size)
    val_ds = TemporalImageDataset(train_raw, val_index, K=args.k_steps, img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    save_path = Path(args.save_path)
    if args.mode == "baseline":
        train_baseline(train_loader, val_loader, device, args.epochs, save_path)
    else:
        train_diffusion(train_loader, val_loader, device, args.epochs, args.t_steps, save_path)


if __name__ == "__main__":
    main()
