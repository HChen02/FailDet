"""Shared utilities for the DA3-as-preprocessor methods.

DA3 itself is never loaded here — it ran once during
``experiments/preprocess_da3.py`` and the depth maps are now on disk at
``data/depth_cache/<dataset_short>/<sample_idx>/frame_<i>.npy``.

Pipeline at train/eval time:
    depth map (fp16, [H, W])
      → per-sample min-max normalise to [0, 1]
      → replicate to 3 channels
      → ImageNet-normalise
      → DINOv2-Large (frozen) → CLS token [1024]
      → (cached in RAM per cell — same pattern as dino_ce_attn)

The trainable decoder is AttentionPool + CNN + head (classifier or
projection). DINOv2 stays frozen.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DINO_HIDDEN = 1024
DINO_REPO = "facebook/dinov2-large"

_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEPTH_CACHE_ROOT = PROJECT_ROOT / "data" / "depth_cache"

# DINOv2-of-depth cache — built lazily on first encode_depth_samples call
# per dataset. Subsequent cells (different methods / seeds) hit the cache
# and skip the ~5-7 min DINOv2 forward over all depth frames.
DEPTH_DINO_CACHE_ROOT = PROJECT_ROOT / "data" / "depth_dino_cache"


def _dataset_short(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def _load_depth_frames(repo_id: str, sample_idx: int) -> list[np.ndarray]:
    """Read all per-frame depth maps for one sample from the disk cache."""
    sample_dir = DEPTH_CACHE_ROOT / _dataset_short(repo_id) / f"{int(sample_idx):06d}"
    if not sample_dir.exists():
        raise FileNotFoundError(
            f"Depth cache missing for {repo_id} sample {sample_idx} "
            f"at {sample_dir}. Run experiments/preprocess_da3.py first.")
    frames = sorted(sample_dir.glob("frame_*.npy"),
                    key=lambda p: int(p.stem.split("_")[1]))
    if not frames:
        raise FileNotFoundError(
            f"Depth cache dir exists but is empty: {sample_dir}")
    return [np.load(p) for p in frames]


def _depth_to_dino_input(depths: list[np.ndarray]) -> torch.Tensor:
    """Convert N depth maps → [N, 3, H, W] tensor ready for DINOv2.

    Per-frame min-max normalisation to [0, 1] (depth ranges vary per scene),
    replicate single channel to 3, then ImageNet-normalise so DINOv2 sees
    pixel statistics it was trained on.
    """
    out = []
    for d in depths:
        t = torch.from_numpy(np.asarray(d, dtype=np.float32))
        # min-max normalise per frame
        dmin, dmax = t.min(), t.max()
        t = (t - dmin) / (dmax - dmin + 1e-9)
        # [H, W] -> [3, H, W]
        t = t.unsqueeze(0).repeat(3, 1, 1)
        out.append(t)
    x = torch.stack(out, dim=0)  # [N, 3, H, W]
    mean = torch.tensor(_NORM_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_NORM_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def load_dinov2_l(device: str = "cuda"):
    """Frozen DINOv2-L loader."""
    from transformers import AutoModel
    log.info(f"Loading {DINO_REPO} (frozen) ...")
    model = AutoModel.from_pretrained(DINO_REPO).to(device).bfloat16()
    for p in model.parameters():
        p.requires_grad = False
    model.train(False)
    return model


def _dino_cache_path(repo_id: str, sample_idx: int) -> Path:
    return (DEPTH_DINO_CACHE_ROOT / _dataset_short(repo_id) /
            f"{int(sample_idx):06d}.npy")


def encode_depth_samples(
    repo_id: str, indices: Sequence[int], encoder,
    device: str = "cuda", encode_chunk: int = 16,
    progress_every: int = 200,
) -> tuple[torch.Tensor, int]:
    """Read depth maps for the given indices, run DINOv2 on them, return
    CLS tokens [N, n_frames, 1024] on CPU + the n_frames count.

    Cached per-sample at ``data/depth_dino_cache/<dataset_short>/<idx>.npy``
    as fp16 [n_frames, 1024]. First call builds the cache; later calls hit
    it. The cache is keyed by repo_id+sample_idx, NOT by indices subset, so
    different methods running on the same dataset reuse it.

    All samples in `indices` must have the same frame count (asserted —
    matches the AttentionPool's uniform-n requirement).
    """
    if not indices:
        return torch.empty(0, 1, DINO_HIDDEN), 0

    cache_dir = DEPTH_DINO_CACHE_ROOT / _dataset_short(repo_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reference frame count from the first sample's depth cache.
    first = _load_depth_frames(repo_id, int(indices[0]))
    n_ref = len(first)

    # Split indices into cache hits vs misses.
    feats_per_sample: dict[int, np.ndarray] = {}
    todo: list[int] = []
    for i in indices:
        p = _dino_cache_path(repo_id, int(i))
        if p.exists():
            arr = np.load(p)
            if arr.shape == (n_ref, DINO_HIDDEN):
                feats_per_sample[int(i)] = arr
                continue
            else:
                log.warning(f"  dino-cache shape mismatch at {p}: "
                            f"got {arr.shape}, expected ({n_ref},{DINO_HIDDEN}). "
                            f"Will recompute.")
        todo.append(int(i))

    if todo:
        log.info(f"  encoding {len(todo)}/{len(indices)} depth-sample CLS tokens "
                 f"(cache hits: {len(feats_per_sample)})")
        for start in range(0, len(todo), encode_chunk):
            batch_idx = todo[start:start + encode_chunk]
            all_imgs: list[torch.Tensor] = []
            for i in batch_idx:
                depths = _load_depth_frames(repo_id, int(i))
                if len(depths) != n_ref:
                    raise RuntimeError(
                        f"Sample {i} has {len(depths)} frames, expected {n_ref}.")
                x = _depth_to_dino_input(depths)
                all_imgs.append(x)
            x = torch.cat(all_imgs, dim=0).to(device).bfloat16()
            with torch.no_grad():
                out = encoder(pixel_values=x)
            cls = out.last_hidden_state[:, 0, :].float()
            cls = cls.view(len(batch_idx), n_ref, -1)
            cls_np = cls.detach().cpu().numpy().astype(np.float16)
            for k, i in enumerate(batch_idx):
                np.save(_dino_cache_path(repo_id, i), cls_np[k])
                feats_per_sample[int(i)] = cls_np[k]
            if ((start // encode_chunk) + 1) * encode_chunk % progress_every < encode_chunk:
                log.info(f"    DINO-on-depth {min(start + encode_chunk, len(todo))}/"
                         f"{len(todo)}")

    # Assemble in the requested order, fp32 for downstream.
    feats = torch.from_numpy(
        np.stack([feats_per_sample[int(i)].astype(np.float32) for i in indices],
                 axis=0)
    )
    return feats, n_ref


# ----------------------------------------------------------------------
# Trainable decoder modules
# ----------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Identical shape to dino_ce_attn.AttentionPool — DINO_HIDDEN = 1024."""
    def __init__(self, dim: int = DINO_HIDDEN, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, n_frames, dim]
        q = self.query.expand(x.size(0), -1, -1)
        out, attn_w = self.mha(q, x, x)
        return self.norm(out.squeeze(1)), attn_w


class CNNBlock(nn.Module):
    """Trainable Linear+BN+ReLU stack — n_layers={2,3,4} per user spec."""
    def __init__(self, in_dim: int = DINO_HIDDEN, hidden: int = 512,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        assert n_layers >= 2
        layers: list[nn.Module] = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden
            layers += [nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden),
                       nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.body = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x):
        return self.body(x)


def build_classifier_head(in_dim: int, n_classes: int) -> nn.Linear:
    return nn.Linear(in_dim, n_classes)


def build_projection_head(in_dim: int, embed_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.GELU(),
        nn.Linear(512, embed_dim),
    )
