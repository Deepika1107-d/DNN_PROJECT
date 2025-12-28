"""
Dataset utilities for image-only temporal prediction (Project 2).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_index_table(split, story_row_indices: Sequence[int], K: int = 4) -> List[Tuple[int, int, int]]:
    rows: List[Tuple[int, int, int]] = []
    for row_idx in story_row_indices:
        ex = split[int(row_idx)]
        fc = int(ex["frame_count"])
        if fc < K + 1:
            continue
        for t in range(K, fc):
            rows.append((int(row_idx), int(t), int(fc)))
    return rows


def build_image_transform(img_size: int = 128):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def preprocess_pil(im, transform) -> torch.Tensor:
    return transform(im.convert("RGB"))


class StoryFramesWindowDataset(Dataset):
    """
    Lightweight dataset for EDA/visualization that keeps PIL images.
    """

    def __init__(self, base_split, index_rows: Sequence[Tuple[int, int, int]], K: int = 4):
        self.base = base_split
        self.index = list(index_rows)
        self.K = K

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row_idx, t, fc = self.index[i]
        ex = self.base[row_idx]
        imgs = ex["images"]
        ctx = imgs[t - self.K : t]
        tgt = imgs[t]
        return {
            "story_id": ex["story_id"],
            "row_idx": row_idx,
            "t_index": t,
            "frame_count": fc,
            "ctx_images": ctx,
            "target_image": tgt,
        }


class TemporalImageDataset(Dataset):
    """
    Image-only temporal dataset. Returns context frames and target frame as tensors in [-1, 1].
    """

    def __init__(self, base_split, index_rows: Sequence[Tuple[int, int, int]], K: int = 4, img_size: int = 128):
        self.base = base_split
        self.index = list(index_rows)
        self.K = K
        self.transform = build_image_transform(img_size)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row_idx, t, _ = self.index[i]
        ex = self.base[row_idx]
        imgs = ex["images"]
        ctx_imgs = imgs[t - self.K : t]
        tgt_img = imgs[t]

        ctx = torch.stack([preprocess_pil(im, self.transform) for im in ctx_imgs], dim=0)
        tgt = preprocess_pil(tgt_img, self.transform)

        return {
            "ctx_images": ctx,
            "target_image": tgt,
            "story_id": ex["story_id"],
            "t_index": t,
        }
