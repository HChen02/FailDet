"""DINOv2-CL: pure-vision supervised contrastive learning.

The contrastive counterpart to DINOv2-CE. Same encoder, no language at
all, but the loss is supervised InfoNCE on a 128-d projected feature
instead of cross-entropy on logits.

Architecture
------------
    inputs : 6 viewpoint PIL images (no text).
    encoder: facebook/dinov2-large (300M params, 1024-d CLS).
    pool   : mean over the 6 per-viewpoint CLS tokens.
    head   : Linear(1024, 512) -> GELU -> Linear(512, 128) -> L2-norm.
    loss   : supervised InfoNCE (losses.infonce.InfoNCELoss). Same-class
             samples are positives, different-class are negatives.
    eval   : nearest train-class centroid in 128-d projected space.

Variants (config["variant"])
----------------------------
    frozen - DINOv2 weights frozen. We CANNOT cache projections like
             DINOv2-CE does, because the projection head trains and its
             weights change every step. We can however cache the frozen
             encoder's CLS features once (still ~50 MB) and run only the
             projection head per step. lr_proj=1e-2, epochs=30,
             batch_size=32, temperature=1.0.
    lora   - LoRA r=16 on DINOv2 attention + train projection head.
             lr_encoder=1e-4, lr_proj=1e-2, epochs=15, batch_size=32.

Why this method matters
-----------------------
This is the direct test of "does CL fail because the encoder isn't right
for it?" CL-FT (vision-only LoRA on Qwen3.5-VL's vision tower) flat-lined
at log(B) loss; the conjecture in CLAUDE.md is that Qwen's vision
features are too entangled with the language objective to reshape into
class-discriminative directions. DINOv2 was trained with self-supervised
contrastive (DINO/iBOT) - its features should already be much more
separable. If DINOv2-CL succeeds where CL-FT failed, the encoder choice
was the bottleneck. If it also stalls, the issue is deeper.

At batch_size=32 on a binary task with ~50/50 split there are >=15 same-
class anchors and >=15 different-class negatives in every batch, so we
do NOT need GradCache here.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.methods.common import (  # noqa: E402
    UNIFIED_LABEL_NAMES, capture_environment, compute_present_classes,
    finalize_metrics_schema, save_results_atomically, select_indices,
    set_seed, setup_run_logger, write_done_flag,
)


_DINO_MODEL = "facebook/dinov2-large"
_DINO_HIDDEN = 1024


def _task_n_classes(task: str) -> int:
    return {"binary": 2, "7class": 7, "8class": 8}[task]


def _filter_idx_for_task(raw, indices, task: str):
    if task != "7class":
        return list(indices)
    return [i for i in indices if int(raw[i]["failure_label"]) >= 1]


def _remap_label_for_task(failure_label: int, task: str) -> int:
    if task == "8class":
        return int(failure_label)
    if task == "binary":
        return 0 if int(failure_label) == 0 else 1
    if task == "7class":
        return int(failure_label) - 1
    raise ValueError(f"Unknown task: {task!r}")


def _build_proj_head(in_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.GELU(),
        nn.Linear(512, 128),
    )


def _apply_dino_lora(encoder, *, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    return get_peft_model(encoder, cfg)


def train_dino_cl(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    variant = str(config.get("variant", "frozen"))
    if variant not in ("frozen", "lora"):
        raise ValueError(f"variant must be 'frozen' or 'lora', got {variant!r}")

    run_dir = Path(config["run_dir"])
    log = setup_run_logger(run_dir, name=f"dino_cl.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"

    log.info(f"=== train_dino_cl | exp_id={config.get('exp_id')} task={task} "
             f"variant={variant} seed={seed} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "dino_cl",
        "task": task,
        "variant": variant,
        "seed": seed,
        "config": dict(config),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at": time.time(),
        "environment": capture_environment(),
    }
    save_results_atomically(blob, metrics_path)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoImageProcessor, Dinov2Model
    from data.dataset import GuardianDataset
    from losses.infonce import InfoNCELoss
    from evaluation.metrics import compute_classification_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optional per-process VRAM cap (so DINOv2 can co-exist with another
    # CUDA job without monopolising the GPU). Set CUDA_MEMORY_FRACTION
    # in the environment to a float in (0, 1]; default = no cap.
    mem_frac = float(os.environ.get("CUDA_MEMORY_FRACTION", "0") or 0)
    if device == "cuda" and 0.0 < mem_frac <= 1.0:
        torch.cuda.set_per_process_memory_fraction(mem_frac, device=0)
        cap_gb = torch.cuda.get_device_properties(0).total_memory * mem_frac / 1e9
        log.info(f"CUDA_MEMORY_FRACTION={mem_frac:.2f} -> capping this "
                 f"process at ~{cap_gb:.1f} GB VRAM")

    if variant == "frozen":
        EPOCHS = int(config.get("epochs", 30))
        BATCH_SIZE = int(config.get("batch_size", 32))
        LR_PROJ = float(config.get("lr_proj", 1e-2))
        LR_ENCODER = None
    else:
        EPOCHS = int(config.get("epochs", 15))
        BATCH_SIZE = int(config.get("batch_size", 32))
        LR_PROJ = float(config.get("lr_proj", 1e-2))
        LR_ENCODER = float(config.get("lr_encoder", 1e-4))
    TEMPERATURE = float(config.get("temperature", 1.0))
    EARLY_STOPPING_PATIENCE = int(config.get("early_stopping_patience", 2))

    log.info(f"Loading {_DINO_MODEL} ...")
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(_DINO_MODEL)
    encoder = Dinov2Model.from_pretrained(_DINO_MODEL, torch_dtype=torch.float32)
    encoder.to(device)
    blob["load_time_sec"] = round(time.time() - t0, 2)

    if variant == "frozen":
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.train(False)
    else:
        encoder = _apply_dino_lora(encoder, r=16, alpha=16, dropout=0.05)
        n_train = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        log.info(f"DINOv2 LoRA: trainable encoder params = {n_train:,}")

    raw_train = GuardianDataset(config["dataset_train"])
    raw_eval = GuardianDataset(config["dataset_eval"], corruption_name=config.get("corruption"), severity=config.get("severity"))

    train_idx = select_indices(
        len(raw_train),
        data_fraction=float(config.get("data_fraction", 1.0)),
        seed=seed,
    )

    val_fraction = float(config.get("val_fraction", 0.05))
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    n_val = max(8, int(round(val_fraction * len(train_idx)))) if val_fraction > 0 else 0
    if n_val >= len(train_idx):
        n_val = max(0, len(train_idx) // 10)
    val_idx_split = [train_idx[int(perm[i])] for i in range(n_val)]
    train_only_idx = [train_idx[int(perm[i])] for i in range(n_val, len(train_idx))]
    train_only_idx = _filter_idx_for_task(raw_train, train_only_idx, task)
    val_idx_split = _filter_idx_for_task(raw_train, val_idx_split, task)

    present_classes, present_label_names, _ = (
        compute_present_classes(raw_train, train_only_idx, task))
    cid_to_pos = {c: i for i, c in enumerate(present_classes)}
    k_present = len(present_classes)
    log.info(f"task '{task}' variant '{variant}': present classes = "
             f"{present_classes}  ({k_present} of {_task_n_classes(task)}) "
             f"names={present_label_names}")

    test_indices = list(range(len(raw_eval)))
    if task == "7class":
        test_indices = [i for i in test_indices
                        if int(raw_eval[i]["failure_label"]) >= 1]
    n_eval_before = len(test_indices)
    test_indices = [
        i for i in test_indices
        if _remap_label_for_task(int(raw_eval[i]["failure_label"]), task)
        in cid_to_pos
    ]
    n_eval_dropped = n_eval_before - len(test_indices)

    log.info(f"data: train={len(train_only_idx)}  val={len(val_idx_split)}  "
             f"test={len(test_indices)}  (dropped {n_eval_dropped} "
             f"absent-class test samples)")

    proj_head = _build_proj_head(_DINO_HIDDEN).to(device)
    contrastive_loss = InfoNCELoss(temperature=TEMPERATURE)

    # Cap the encoder forward chunk so memory stays bounded even when the
    # contrastive batch_size is large (e.g. 512).
    ENCODE_CHUNK = min(BATCH_SIZE, 32)

    def _encode_indices(ds, indices):
        """Frozen DINOv2 forward over each sample's 6 images, mean-pool to
        a single 1024-d feature. (Used to build the cache for the frozen
        variant + the test/val features for both variants at eval time.)
        Returns (features [N, 1024], labels [N])."""
        feats: list = []
        labels: list[int] = []
        chunk = max(1, ENCODE_CHUNK)
        for start in range(0, len(indices), chunk):
            batch_idx = indices[start:start + chunk]
            samples = [ds[int(i)] for i in batch_idx]
            flat_images: list = []
            n_per_sample: list[int] = []
            for s in samples:
                flat_images.extend(s["images"])
                n_per_sample.append(len(s["images"]))
            inputs = processor(images=flat_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                out = encoder(pixel_values=pixel_values)
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.stack(
                [p.mean(dim=0) for p in cls.split(n_per_sample, dim=0)],
                dim=0,
            )
            feats.append(cls.detach().cpu())
            for s in samples:
                labels.append(_remap_label_for_task(int(s["failure_label"]), task))
        return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.long)

    if variant == "frozen":
        log.info("Pre-computing frozen DINOv2 features for train + val ...")
        t_feat = time.time()
        train_feats, train_labels = _encode_indices(raw_train, train_only_idx)
        if val_idx_split:
            val_feats, val_labels = _encode_indices(raw_train, val_idx_split)
        else:
            val_feats = torch.empty(0, _DINO_HIDDEN)
            val_labels = torch.empty(0, dtype=torch.long)
        train_label_pos = torch.tensor(
            [cid_to_pos[int(x.item())] for x in train_labels], dtype=torch.long)
        val_label_pos = (torch.tensor(
            [cid_to_pos[int(x.item())] for x in val_labels], dtype=torch.long)
            if len(val_labels) > 0 else torch.empty(0, dtype=torch.long))
        log.info(f"  feature extraction time: {time.time() - t_feat:.1f}s "
                 f"(train={tuple(train_feats.shape)})")

        train_loader = DataLoader(
            TensorDataset(train_feats, train_label_pos),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False,
        )
        val_loader = (DataLoader(
            TensorDataset(val_feats, val_label_pos),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            if len(val_label_pos) > 0 else None)

        optimizer = torch.optim.AdamW(
            proj_head.parameters(), lr=LR_PROJ, weight_decay=0.01)
    else:
        from torch.utils.data import Dataset as TorchDataset

        class _SampleDS(TorchDataset):
            def __init__(self, ds, idx_list):
                self.ds = ds
                self.idx = list(idx_list)
            def __len__(self):
                return len(self.idx)
            def __getitem__(self, i):
                s = self.ds[self.idx[i]]
                return {
                    "images": s["images"],
                    "label": cid_to_pos[
                        _remap_label_for_task(int(s["failure_label"]), task)],
                }

        def _collate(batch):
            return {
                "images": [b["images"] for b in batch],
                "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            }

        train_ds = _SampleDS(raw_train, train_only_idx)
        val_ds = _SampleDS(raw_train, val_idx_split) if val_idx_split else None

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, drop_last=False, collate_fn=_collate)
        val_loader = (DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=_collate)
            if val_ds is not None else None)

        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": LR_ENCODER, "name": "dino_lora"},
                {"params": list(proj_head.parameters()), "lr": LR_PROJ, "name": "proj"},
            ],
            weight_decay=0.01,
        )

    use_early_stopping = val_loader is not None and EARLY_STOPPING_PATIENCE > 0
    blob["early_stopping"] = {
        "enabled": use_early_stopping,
        "patience": EARLY_STOPPING_PATIENCE if use_early_stopping else None,
        "n_val": len(val_idx_split),
    }

    total_optim_steps = max(1, EPOCHS * max(1, len(train_loader)))
    warmup_steps = max(1, int(round(total_optim_steps * 0.1)))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log.info(f"Training: variant={variant} epochs={EPOCHS} bs={BATCH_SIZE} "
             f"lr_proj={LR_PROJ}{' lr_encoder=' + str(LR_ENCODER) if LR_ENCODER else ''} "
             f"tau={TEMPERATURE} early_stopping={use_early_stopping}")

    epoch_losses: list[float] = []
    epoch_val_losses: list[float] = []
    epoch_intra_sim: list[float] = []
    epoch_inter_sim: list[float] = []
    best_val = float("inf")
    epochs_since_improvement = 0
    early_stopped_at = None

    def _embed_from_features(feats: "torch.Tensor"):
        return F.normalize(proj_head(feats), dim=-1)

    def _embed_from_images(images_list):
        flat_images = [img for sublist in images_list for img in sublist]
        n_per_sample = [len(sub) for sub in images_list]
        inputs = processor(images=flat_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = encoder(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :]
        feats = torch.stack(
            [p.mean(dim=0) for p in cls.split(n_per_sample, dim=0)],
            dim=0,
        )
        return F.normalize(proj_head(feats), dim=-1)

    def _diag_intra_inter(embeds, labels):
        with torch.no_grad():
            sim = (embeds @ embeds.T).cpu()
            B = embeds.size(0)
            eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).cpu()
            eye = torch.eye(B, dtype=torch.bool)
            pos_mask = eq & ~eye
            neg_mask = ~eq & ~eye
            intra = float(sim[pos_mask].mean()) if pos_mask.any() else 0.0
            inter = float(sim[neg_mask].mean()) if neg_mask.any() else 0.0
        return intra, inter

    def _val_loss():
        proj_head.train(False)
        if variant == "lora":
            encoder.train(False)
        embeds: list = []
        all_l: list = []
        with torch.no_grad():
            if variant == "frozen":
                for f, l in val_loader:
                    e = _embed_from_features(f.to(device))
                    embeds.append(e); all_l.append(l.to(device))
            else:
                for batch in val_loader:
                    e = _embed_from_images(batch["images"])
                    embeds.append(e); all_l.append(batch["labels"].to(device))
        proj_head.train(True)
        if variant == "lora":
            encoder.train(True)
        if not embeds:
            return float("nan")
        return float(contrastive_loss(
            torch.cat(embeds, dim=0), labels=torch.cat(all_l, dim=0)).detach())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for epoch in range(EPOCHS):
        proj_head.train(True)
        if variant == "lora":
            encoder.train(True)
        running = 0.0; n_batches = 0
        intra_sum = 0.0; inter_sum = 0.0; sim_n = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if variant == "frozen":
                feats, labels = batch
                feats = feats.to(device); labels = labels.to(device)
                emb = _embed_from_features(feats)
            else:
                labels = batch["labels"].to(device)
                emb = _embed_from_images(batch["images"])
            loss = contrastive_loss(emb, labels=labels)
            if loss.grad_fn is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    max_norm=1.0)
                optimizer.step()
            scheduler.step()
            running += float(loss.detach()); n_batches += 1
            intra, inter = _diag_intra_inter(emb, labels)
            intra_sum += intra; inter_sum += inter; sim_n += 1
        avg = running / max(1, n_batches)
        intra = intra_sum / max(1, sim_n)
        inter = inter_sum / max(1, sim_n)
        epoch_losses.append(avg)
        epoch_intra_sim.append(intra)
        epoch_inter_sim.append(inter)

        val_loss = None
        if use_early_stopping:
            val_loss = _val_loss()
            epoch_val_losses.append(val_loss)
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            log.info(f"  [DINOv2-CL ep {epoch+1:02d}/{EPOCHS}] loss={avg:.4f} "
                     f"val={val_loss:.4f} best={best_val:.4f} "
                     f"intra={intra:.4f} inter={inter:.4f} "
                     f"gap={(intra-inter):.4f}")
        else:
            log.info(f"  [DINOv2-CL ep {epoch+1:02d}/{EPOCHS}] loss={avg:.4f} "
                     f"intra={intra:.4f} inter={inter:.4f} "
                     f"gap={(intra-inter):.4f}")

        blob["train_progress"] = {
            "epochs_done": epoch + 1,
            "epoch_losses": epoch_losses,
            "epoch_val_losses": epoch_val_losses,
            "epoch_intra_class_sim": epoch_intra_sim,
            "epoch_inter_class_sim": epoch_inter_sim,
            "best_val_loss": best_val if use_early_stopping else None,
        }
        save_results_atomically(blob, metrics_path)

        if use_early_stopping and epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped_at = epoch + 1
            log.info(f"  [DINOv2-CL] early stopping at epoch {early_stopped_at}")
            break

    train_time = time.time() - t0
    blob["train"] = {
        "variant": variant, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr_proj": LR_PROJ, "lr_encoder": LR_ENCODER, "temperature": TEMPERATURE,
        "epoch_losses": epoch_losses,
        "epoch_val_losses": epoch_val_losses,
        "epoch_intra_class_sim": epoch_intra_sim,
        "epoch_inter_class_sim": epoch_inter_sim,
        "best_val_loss": best_val if use_early_stopping else None,
        "early_stopped_at_epoch": early_stopped_at,
        "train_time_sec": round(train_time, 2),
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                            if torch.cuda.is_available() else None),
    }
    save_results_atomically(blob, metrics_path)

    # Inference: nearest-centroid in projected 128-d space ------------------
    log.info("Building train-set centroids in 128-d projected space ...")
    proj_head.train(False)
    if variant == "lora":
        encoder.train(False)

    centroid_sum = torch.zeros(k_present, 128)
    centroid_counts: dict[int, int] = defaultdict(int)

    if variant == "frozen":
        # We already have train_feats; just project them.
        with torch.no_grad():
            for start in range(0, len(train_feats), BATCH_SIZE):
                f_batch = train_feats[start:start + BATCH_SIZE].to(device)
                pos_batch = train_label_pos[start:start + BATCH_SIZE]
                e = _embed_from_features(f_batch).cpu()
                for vec, pos in zip(e, pos_batch.tolist()):
                    centroid_sum[int(pos)] += vec
                    centroid_counts[int(pos)] += 1
    else:
        # Re-encode train through (LoRA-updated) DINOv2 + proj.
        for start in range(0, len(train_only_idx), BATCH_SIZE):
            batch_idx = train_only_idx[start:start + BATCH_SIZE]
            samples = [raw_train[int(i)] for i in batch_idx]
            images_list = [s["images"] for s in samples]
            labels_pos = [cid_to_pos[
                _remap_label_for_task(int(s["failure_label"]), task)]
                for s in samples]
            with torch.no_grad():
                e = _embed_from_images(images_list).cpu()
            for vec, pos in zip(e, labels_pos):
                centroid_sum[int(pos)] += vec
                centroid_counts[int(pos)] += 1

    centroids = torch.zeros_like(centroid_sum)
    for pos in range(k_present):
        if centroid_counts[pos] > 0:
            centroids[pos] = centroid_sum[pos] / centroid_counts[pos]
    centroids = F.normalize(centroids, dim=-1)
    log.info(f"  centroid counts: "
             f"{ {present_label_names[pos]: centroid_counts.get(pos, 0) for pos in range(k_present)} }")

    label_names = list(present_label_names)
    predictions: list[str] = []
    ground_truths: list[str] = []
    t_eval = time.time()
    with torch.no_grad():
        if variant == "frozen":
            test_feats, test_labels = _encode_indices(raw_eval, test_indices)
            test_label_pos = [cid_to_pos[int(x.item())] for x in test_labels]
            for start in range(0, len(test_feats), BATCH_SIZE):
                f_batch = test_feats[start:start + BATCH_SIZE].to(device)
                e = _embed_from_features(f_batch).cpu()
                for vec, gt_pos in zip(e, test_label_pos[start:start + BATCH_SIZE]):
                    sims = (vec.unsqueeze(0) @ centroids.T).squeeze(0)
                    pred_pos = int(sims.argmax().item())
                    predictions.append(label_names[pred_pos])
                    ground_truths.append(label_names[int(gt_pos)])
        else:
            for start in range(0, len(test_indices), BATCH_SIZE):
                batch_idx = test_indices[start:start + BATCH_SIZE]
                samples = [raw_eval[int(i)] for i in batch_idx]
                images_list = [s["images"] for s in samples]
                gt_pos_list = [cid_to_pos[
                    _remap_label_for_task(int(s["failure_label"]), task)]
                    for s in samples]
                e = _embed_from_images(images_list).cpu()
                for vec, gt_pos in zip(e, gt_pos_list):
                    sims = (vec.unsqueeze(0) @ centroids.T).squeeze(0)
                    pred_pos = int(sims.argmax().item())
                    predictions.append(label_names[pred_pos])
                    ground_truths.append(label_names[int(gt_pos)])
    eval_time = time.time() - t_eval

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"DINOv2-CL eval: acc={m['accuracy']:.4f} "
             f"f1_macro={m['f1_macro']:.4f}")
    if m.get("majority_class_warning"):
        log.warning(m["majority_class_warning"])

    blob.update({
        "metrics": m,
        "eval_time_sec": round(eval_time, 2),
        "predictions": predictions,
        "ground_truths": ground_truths,
        "centroid_counts": {present_label_names[pos]: int(centroid_counts.get(pos, 0))
                            for pos in range(k_present)},
        "present_classes": present_classes,
        "present_label_names": present_label_names,
        "n_classes_present": k_present,
        "n_eval_dropped_absent_class": n_eval_dropped,
        "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": True,
    })
    finalize_metrics_schema(blob)
    save_results_atomically(blob, metrics_path)
    # Persist projection head + trained centroids (and LoRA adapter if
    # applicable) for cross-domain eval reuse — added 2026-05-11. The
    # centroids are derived from the RLBench train set; a future
    # cross-domain eval can either re-derive them or reuse these.
    head_payload = {
        "method": "dino_cl",
        "variant": variant,
        "proj_head_state": proj_head.state_dict(),
        "centroids": centroids.detach().cpu(),
        "encoder_lora_state": (
            encoder.state_dict() if variant == "lora" else None),
        "present_classes": present_classes,
        "present_label_names": present_label_names,
    }
    torch.save(head_payload, run_dir / "head.pt")
    log.info(f"persisted head -> {run_dir / 'head.pt'}")
    write_done_flag(run_dir, {
        "method": "dino_cl",
        "variant": variant,
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
    })
    return blob


# ---------------------------------------------------------------------------
# Eval-only mode — load proj_head + centroids from head.pt.
# ---------------------------------------------------------------------------

def eval_dino_cl(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    ckpt_dir = Path(config["from_checkpoint"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(
        run_dir, name=f"dino_cl.eval.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    log.info(f"=== eval_dino_cl | from_checkpoint={ckpt_dir} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "dino_cl",
        "task": task, "seed": seed,
        "config": dict(config),
        "from_checkpoint": str(ckpt_dir),
        "eval_only": True,
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at": time.time(),
        "environment": capture_environment(),
    }
    save_results_atomically(blob, metrics_path)

    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, Dinov2Model
    from data.dataset import GuardianDataset
    from evaluation.metrics import compute_classification_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = int(config.get("batch_size", 32))

    head_pt = ckpt_dir / "head.pt"
    if not head_pt.exists():
        raise FileNotFoundError(f"head.pt not found at {head_pt}")
    head = torch.load(head_pt, map_location="cpu", weights_only=False)
    variant = head.get("variant", "frozen")
    present_classes = head["present_classes"]
    present_label_names = head["present_label_names"]
    k_present = len(present_classes)
    cid_to_pos = {c: i for i, c in enumerate(present_classes)}
    centroids = head["centroids"].to(device)  # [k_present, 128], L2-normed

    processor = AutoImageProcessor.from_pretrained(_DINO_MODEL)
    encoder = Dinov2Model.from_pretrained(_DINO_MODEL, torch_dtype=torch.float32)
    encoder.to(device)
    if variant == "lora":
        encoder = _apply_dino_lora(encoder, r=16, alpha=16, dropout=0.05)
        encoder.load_state_dict(head["encoder_lora_state"], strict=False)
    encoder.train(False)
    for p in encoder.parameters():
        p.requires_grad_(False)

    proj_head = _build_proj_head(_DINO_HIDDEN).to(device)
    proj_head.load_state_dict(head["proj_head_state"])
    proj_head.train(False)

    raw_eval = GuardianDataset(config["dataset_eval"], corruption_name=config.get("corruption"), severity=config.get("severity"))
    test_indices = list(range(len(raw_eval)))
    if task == "7class":
        test_indices = [i for i in test_indices
                        if int(raw_eval[i]["failure_label"]) >= 1]
    n_eval_before = len(test_indices)
    test_indices = [
        i for i in test_indices
        if _remap_label_for_task(int(raw_eval[i]["failure_label"]), task)
        in cid_to_pos
    ]
    n_eval_dropped = n_eval_before - len(test_indices)
    log.info(f"eval samples: {len(test_indices)} "
             f"(dropped {n_eval_dropped} absent-class)")

    ENCODE_CHUNK = min(BATCH_SIZE, 32)

    def _encode_indices(ds, indices):
        feats: list = []
        labels: list[int] = []
        chunk = max(1, ENCODE_CHUNK)
        for start in range(0, len(indices), chunk):
            batch_idx = indices[start:start + chunk]
            samples = [ds[int(i)] for i in batch_idx]
            flat_images: list = []
            n_per_sample: list[int] = []
            for s in samples:
                flat_images.extend(s["images"])
                n_per_sample.append(len(s["images"]))
            inputs = processor(images=flat_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                out = encoder(pixel_values=pixel_values)
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.stack(
                [p.mean(dim=0) for p in cls.split(n_per_sample, dim=0)],
                dim=0,
            )
            feats.append(cls.detach().cpu())
            for s in samples:
                labels.append(_remap_label_for_task(int(s["failure_label"]), task))
        return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.long)

    log.info("Encoding eval set ...")
    t_eval = time.time()
    test_feats, test_labels = _encode_indices(raw_eval, test_indices)
    test_label_pos = [cid_to_pos[int(x.item())] for x in test_labels]

    label_names = list(present_label_names)
    predictions: list[str] = []
    ground_truths: list[str] = []
    with torch.no_grad():
        for start in range(0, len(test_feats), BATCH_SIZE):
            f_batch = test_feats[start:start + BATCH_SIZE].to(device)
            e = F.normalize(proj_head(f_batch), dim=-1)
            for vec, gt_pos in zip(e.cpu(), test_label_pos[start:start + BATCH_SIZE]):
                sims = (vec.unsqueeze(0) @ centroids.cpu().T).squeeze(0)
                pred_pos = int(sims.argmax().item())
                predictions.append(label_names[pred_pos])
                ground_truths.append(label_names[int(gt_pos)])
    eval_time = time.time() - t_eval

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"DINOv2-CL eval-only: acc={m['accuracy']:.4f} "
             f"f1_macro={m['f1_macro']:.4f}")
    if m.get("majority_class_warning"):
        log.warning(m["majority_class_warning"])

    blob.update({
        "metrics": m,
        "eval_time_sec": round(eval_time, 2),
        "predictions": predictions,
        "ground_truths": ground_truths,
        "present_classes": present_classes,
        "present_label_names": present_label_names,
        "n_classes_present": k_present,
        "n_eval_dropped_absent_class": n_eval_dropped,
        "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": True,
    })
    finalize_metrics_schema(blob)
    save_results_atomically(blob, metrics_path)
    write_done_flag(run_dir, {
        "method": "dino_cl", "variant": variant,
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
        "eval_only": True,
    })
    return blob
