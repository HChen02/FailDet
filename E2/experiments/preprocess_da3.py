"""DA3 depth-map preprocessing — one-time, cached to disk.

Walks the 9 Guardian splits (RLBench / UR5 / BDV2 × train/val/test) and writes a
per-frame depth map to disk:

    data/depth_cache/<dataset_short>/<sample_idx>/frame_<i>.npy   (float16)

Saves 5 colormapped sanity-check visualizations to results/depth_samples/.

DA3 is loaded only here. The training and eval scripts never touch DA3 — they
just read the cached depth maps. Run once before any da3_dino_* training.

Usage:
    python experiments/preprocess_da3.py                # all 9 splits
    python experiments/preprocess_da3.py --splits rlbench_train rlbench_test
    python experiments/preprocess_da3.py --resize 252  # default resize H=W
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from depth_anything_3.cfg import load_config, create_object
from data.dataset import GuardianDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

CACHE_ROOT = PROJECT_ROOT / "data" / "depth_cache"
DA3_REPO = "depth-anything/DA3-BASE"
# Locate the bundled YAML — depth_anything_3 is pip-installed editable from
# /srv/robotica/dataset/hao/Depth-Anything-3, so probe its installed location.
import depth_anything_3 as _da3_pkg
DA3_CONFIG = Path(_da3_pkg.__path__[0]) / "configs" / "da3-base.yaml"

# Map between our split keys and HuggingFace repo IDs.
ALL_SPLITS = {
    "rlbench_train": "paulpacaud/rlbenchfail_train_dataset",
    "rlbench_val":   "paulpacaud/rlbenchfail_val_dataset",
    "rlbench_test":  "paulpacaud/rlbenchfail_test_dataset",
    "ur5_train":     "paulpacaud/ur5fail_train_dataset",
    "ur5_val":       "paulpacaud/ur5fail_val_dataset",
    "ur5_test":      "paulpacaud/ur5fail_test_dataset",
    "bdv2_train":    "paulpacaud/bdv2fail_train_dataset",
    "bdv2_val":      "paulpacaud/bdv2fail_val_dataset",
    "bdv2_test":     "paulpacaud/bdv2fail_test_dataset",
}

# DA3 input preprocessing — ImageNet stats. Patch size = 14 → resize H=W divisible by 14.
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


def _dataset_short(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def load_da3_net(device: str = "cuda"):
    """Build DepthAnything3Net from yaml, load weights from HF safetensors,
    bypass the api.py / DPT-export / moviepy / pycolmap chain."""
    log.info(f"Loading {DA3_REPO} (yaml + safetensors path, no api.py) ...")
    cfg = load_config(str(DA3_CONFIG))
    net = create_object(cfg).to(device)
    for p in net.parameters():
        p.requires_grad = False
    net.train(False)

    ckpt = hf_hub_download(repo_id=DA3_REPO, filename="model.safetensors")
    sd = {k[len("model."):]: v for k, v in load_file(ckpt).items()
          if k.startswith("model.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        log.warning(f"DA3 load: missing={len(missing)} unexpected={len(unexpected)} "
                    f"(missing are aux-output convs unused for inference)")
    return net


def preprocess_split(
    net, repo_id: str, *,
    resize: int = 252,
    cache_root: Path = CACHE_ROOT,
    batch_size: int = 1,
    device: str = "cuda",
    progress_every: int = 100,
):
    """Run DA3 over every (sample, frame) in the split and write fp16 .npy."""
    ds = GuardianDataset(repo_id)
    short = _dataset_short(repo_id)
    out_dir = cache_root / short
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
    ])

    todo = []
    for i in range(len(ds)):
        sample_dir = out_dir / f"{i:06d}"
        if sample_dir.exists():
            # Skip if all frames already written.
            sample = ds[i]
            n_frames = len(sample["images"])
            if all((sample_dir / f"frame_{j}.npy").exists() for j in range(n_frames)):
                continue
        todo.append(i)
    log.info(f"  {short}: {len(todo)}/{len(ds)} samples to write "
             f"(skipping {len(ds) - len(todo)} already done)")

    if not todo:
        return

    t0 = time.time()
    for k, i in enumerate(todo):
        sample = ds[i]
        imgs = sample["images"]
        sample_dir = out_dir / f"{i:06d}"
        sample_dir.mkdir(exist_ok=True)
        # DA3 wants [B, S, 3, H, W]
        x = torch.stack([tfm(im.convert("RGB")) for im in imgs]).unsqueeze(0).to(device)
        with torch.no_grad():
            out = net(x)
        depth = out["depth"][0]  # [S, H, W] float32
        for j in range(depth.shape[0]):
            np.save(sample_dir / f"frame_{j}.npy",
                    depth[j].cpu().numpy().astype(np.float16))
        if (k + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (k + 1) / elapsed
            eta = (len(todo) - k - 1) / rate
            log.info(f"    {short}: {k + 1}/{len(todo)}  "
                     f"({rate:.1f} samp/s, ETA {eta:.0f}s)")
    log.info(f"  {short}: done in {time.time() - t0:.1f}s")


def save_visualizations(net, repo_id: str, n_samples: int = 5,
                        out_dir: Path = PROJECT_ROOT / "results" / "depth_samples",
                        resize: int = 252, device: str = "cuda"):
    """Save n colormapped depth visualizations for sanity check."""
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = GuardianDataset(repo_id)
    short = _dataset_short(repo_id)

    # Pick 2 success + 3 failure if possible.
    succ = [i for i in range(len(ds)) if ds[i]["failure_mode"] == "success"][:2]
    fail = [i for i in range(len(ds)) if ds[i]["failure_mode"] != "success"][:3]
    sample_idx = succ + fail
    if not sample_idx:
        sample_idx = list(range(min(n_samples, len(ds))))

    tfm = T.Compose([
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
    ])

    for i in sample_idx[:n_samples]:
        sample = ds[i]
        imgs = sample["images"]
        x = torch.stack([tfm(im.convert("RGB")) for im in imgs]).unsqueeze(0).to(device)
        with torch.no_grad():
            depth = net(x)["depth"][0]
        n = min(4, depth.shape[0])
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
        if n == 1:
            axes = axes.reshape(2, 1)
        for j in range(n):
            axes[0, j].imshow(imgs[j])
            axes[0, j].set_title(f"RGB frame {j}")
            axes[0, j].axis("off")
            d = depth[j].cpu().numpy()
            axes[1, j].imshow(d, cmap="turbo")
            axes[1, j].set_title(f"DA3 depth [{d.min():.2f}, {d.max():.2f}]")
            axes[1, j].axis("off")
        fig.suptitle(f"{short} / sample {i} / failure_mode={sample['failure_mode']}")
        fig.tight_layout()
        out_path = out_dir / f"{short}_sample{i:06d}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        log.info(f"  saved viz: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=list(ALL_SPLITS.keys()),
                    choices=list(ALL_SPLITS.keys()),
                    help="Which splits to preprocess (default: all 9)")
    ap.add_argument("--resize", type=int, default=252,
                    help="DA3 input H=W (must be divisible by 14, default 252)")
    ap.add_argument("--no-viz", action="store_true",
                    help="Skip the depth visualization step")
    ap.add_argument("--viz-only", action="store_true",
                    help="Skip cache build, only save visualizations from RLBench train")
    args = ap.parse_args()

    assert args.resize % 14 == 0, "--resize must be divisible by 14 (DA3 patch size)"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    net = load_da3_net(device=device)

    if args.viz_only:
        save_visualizations(net, ALL_SPLITS["rlbench_train"], resize=args.resize)
        return

    log.info("=" * 64)
    log.info(f"DA3 preprocessing — {len(args.splits)} splits, resize={args.resize}")
    log.info("=" * 64)

    t0 = time.time()
    for split_key in args.splits:
        repo_id = ALL_SPLITS[split_key]
        log.info(f"\n--- {split_key} ({repo_id}) ---")
        preprocess_split(net, repo_id, resize=args.resize)

    log.info(f"\n=== ALL SPLITS DONE in {(time.time() - t0)/60:.1f}m ===")
    log.info(f"Cache root: {CACHE_ROOT}")
    log.info(f"  total cache size: {sum(p.stat().st_size for p in CACHE_ROOT.rglob('*.npy')) / 1e9:.2f} GB")

    if not args.no_viz:
        log.info("\n--- Saving depth visualizations ---")
        save_visualizations(net, ALL_SPLITS["rlbench_train"], resize=args.resize)


if __name__ == "__main__":
    main()
