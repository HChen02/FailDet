"""DINOv2-CLIP: vision-language contrastive on dedicated encoders.

The vision-text counterpart to DINOv2-CL. Same DINOv2 image branch, but
contrast images against text rather than against same-class images. Tests
whether adding language signal helps when the vision backbone is already
strong (DINOv2 self-supervised, not VLM-language-grounded).

Architecture
------------
    image branch:
        encoder    : facebook/dinov2-large (1024-d CLS, mean over 6 views).
        image_proj : Linear(1024, 512) -> GELU -> Linear(512, 128) -> L2.
    text branch:
        encoder    : sentence-transformers/all-MiniLM-L6-v2 (FROZEN, 384-d).
        text_proj  : Linear(384,  512) -> GELU -> Linear(512, 128) -> L2.
    logit scale:
        log_temp   : nn.Parameter, init log(1.0) = 0. tau = log_temp.exp().
    loss:
        symmetric InfoNCE on the [B,B] image-text similarity matrix
        (losses.infonce.InfoNCELoss in CLIP mode).

Training text per sample (8-class granularity, regardless of task):
    "Task: {task_instruction}. Subtask: {detailed_subtask_name}. "
    "Outcome: {failure_mode}"

Class-prototype texts at inference time. Two layers:
  - 8-class prototypes (used for 7class and 8class tasks; success +
    7 failure prototypes, filtered by present_classes — wrong_state is
    dropped on RLBench because no train sample carries it):
        success        -> "The robot successfully completed the task"
        no_grasp       -> "The robot failed: gripper did not close on the object"
        slip           -> "The robot failed: object slipped from the gripper"
        translation    -> "The robot failed: object grasped or placed imprecisely"
        rotation       -> "The robot failed: object rotated incorrectly"
        wrong_object   -> "The robot failed: manipulated the wrong object"
        wrong_sequence -> "The robot failed: executed steps in wrong order"
        wrong_state    -> "The robot failed: object ended in the wrong state"
  - Binary uses a coarser pair so the failure prototype isn't biased
    toward any particular failure mode:
        success -> "The robot successfully completed the task"
        failure -> "The robot failed to complete the task"
    Argmax index 0 -> "success", index 1 -> "failure".

Variants (config["variant"])
----------------------------
    frozen - DINOv2 + text encoder both frozen. Only image_proj +
             text_proj + log_temp train. We pre-cache image features once
             and pre-cache text features once - the entire training loop
             is then "tiny linear projections + InfoNCE" on cached
             tensors. Hyperparams: lr=1e-2, epochs=30, batch_size=32.
    lora   - LoRA r=16 on DINOv2 attention. Text encoder stays frozen
             (text features are still cached). Hyperparams:
             lr_dino=1e-4, lr_proj=1e-2, epochs=15, batch_size=32.

Why this method matters
-----------------------
- vs DINOv2-CL: does adding text help, when the encoder is already a
  contrastive-trained ViT?
- vs CL-Embed (VLM, last-token InfoNCE): does pulling text out of the
  VLM and into a separate cheap MiniLM still let CL learn?
- vs CL-FT (vision-only LoRA on Qwen3.5-VL + InfoNCE - retired, flat
  loss): same vision-only setup but with a contrastive-trained encoder.
"""

from __future__ import annotations

import math
import os
import sys
import time
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
_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TEXT_HIDDEN = 384

_PROTOTYPE_TEXTS: list[str] = [
    "The robot successfully completed the task",
    "The robot failed: gripper did not close on the object",
    "The robot failed: object slipped from the gripper",
    "The robot failed: object grasped or placed imprecisely",
    "The robot failed: object rotated incorrectly",
    "The robot failed: manipulated the wrong object",
    "The robot failed: executed steps in wrong order",
    "The robot failed: object ended in the wrong state",
]

# Binary uses a separate two-prototype list. The failure prototype is
# generic so it doesn't lean toward any specific failure mode.
_BINARY_PROTOTYPE_TEXTS: list[str] = [
    "The robot successfully completed the task",
    "The robot failed to complete the task",
]


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


def _build_training_text(sample: dict) -> str:
    return (f"Task: {sample['task_instruction']}. "
            f"Subtask: {sample['detailed_subtask_name']}. "
            f"Outcome: {sample['failure_mode']}")


def _prototype_texts_for_task(task: str) -> tuple[list[str], list[int]]:
    if task == "8class":
        return list(_PROTOTYPE_TEXTS), list(range(8))
    if task == "7class":
        return list(_PROTOTYPE_TEXTS[1:]), list(range(1, 8))
    if task == "binary":
        # class_id 0 = success, 1 = failure (post-binary remap).
        return list(_BINARY_PROTOTYPE_TEXTS), [0, 1]
    raise ValueError(f"Unknown task: {task!r}")


def _binary_label_string(pos_in_present_set: int) -> str:
    return "success" if pos_in_present_set == 0 else "failure"


def train_dino_clip(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    variant = str(config.get("variant", "frozen"))
    if variant not in ("frozen", "lora"):
        raise ValueError(f"variant must be 'frozen' or 'lora', got {variant!r}")

    run_dir = Path(config["run_dir"])
    log = setup_run_logger(run_dir, name=f"dino_clip.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"

    log.info(f"=== train_dino_clip | exp_id={config.get('exp_id')} task={task} "
             f"variant={variant} seed={seed} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "dino_clip",
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
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoImageProcessor, Dinov2Model
    from sentence_transformers import SentenceTransformer
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
        LR_PROJ = float(config.get("lr_proj", config.get("lr", 1e-2)))
        LR_ENCODER = None
    else:
        EPOCHS = int(config.get("epochs", 15))
        BATCH_SIZE = int(config.get("batch_size", 32))
        LR_PROJ = float(config.get("lr_proj", 1e-2))
        LR_ENCODER = float(config.get("lr_encoder", config.get("lr_dino", 1e-4)))
    EARLY_STOPPING_PATIENCE = int(config.get("early_stopping_patience", 2))

    log.info(f"Loading {_DINO_MODEL} (image branch) ...")
    t0 = time.time()
    img_processor = AutoImageProcessor.from_pretrained(_DINO_MODEL)
    img_encoder = Dinov2Model.from_pretrained(_DINO_MODEL, torch_dtype=torch.float32)
    img_encoder.to(device)
    log.info(f"Loading {_TEXT_MODEL} (text branch, frozen) ...")
    text_encoder = SentenceTransformer(_TEXT_MODEL, device=device)
    text_encoder.train(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    blob["load_time_sec"] = round(time.time() - t0, 2)

    if variant == "frozen":
        for p in img_encoder.parameters():
            p.requires_grad_(False)
        img_encoder.train(False)
    else:
        img_encoder = _apply_dino_lora(img_encoder, r=16, alpha=16, dropout=0.05)
        n_train = sum(p.numel() for p in img_encoder.parameters() if p.requires_grad)
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

    # Cap encoder forward chunk so memory stays bounded even when the
    # contrastive batch_size is large (e.g. 512).
    ENCODE_CHUNK = min(BATCH_SIZE, 32)

    def _encode_image_indices(ds, indices):
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
            inputs = img_processor(images=flat_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                out = img_encoder(pixel_values=pixel_values)
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.stack(
                [p.mean(dim=0) for p in cls.split(n_per_sample, dim=0)],
                dim=0,
            )
            feats.append(cls.detach().cpu())
            for s in samples:
                labels.append(_remap_label_for_task(int(s["failure_label"]), task))
        return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.long)

    def _build_texts_for(ds, indices) -> list[str]:
        return [_build_training_text(ds[int(i)]) for i in indices]

    log.info("Encoding training texts with frozen MiniLM ...")
    t_text = time.time()
    train_texts = _build_texts_for(raw_train, train_only_idx)
    train_text_features = text_encoder.encode(
        train_texts, batch_size=64, convert_to_tensor=True,
        show_progress_bar=False,
    ).detach().cpu().float()
    log.info(f"  train text features = {tuple(train_text_features.shape)} "
             f"({time.time() - t_text:.1f}s)")
    if val_idx_split:
        val_texts = _build_texts_for(raw_train, val_idx_split)
        val_text_features = text_encoder.encode(
            val_texts, batch_size=64, convert_to_tensor=True,
            show_progress_bar=False,
        ).detach().cpu().float()
    else:
        val_text_features = torch.empty(0, _TEXT_HIDDEN)

    if variant == "frozen":
        log.info("Pre-computing frozen DINOv2 image features (train + val) ...")
        t_feat = time.time()
        train_img_features, train_labels = _encode_image_indices(
            raw_train, train_only_idx)
        if val_idx_split:
            val_img_features, val_labels = _encode_image_indices(
                raw_train, val_idx_split)
        else:
            val_img_features = torch.empty(0, _DINO_HIDDEN)
            val_labels = torch.empty(0, dtype=torch.long)
        train_label_pos = torch.tensor(
            [cid_to_pos[int(x.item())] for x in train_labels], dtype=torch.long)
        val_label_pos = (torch.tensor(
            [cid_to_pos[int(x.item())] for x in val_labels], dtype=torch.long)
            if len(val_labels) > 0 else torch.empty(0, dtype=torch.long))
        log.info(f"  image feature extraction time: {time.time() - t_feat:.1f}s")

        train_dataset = TensorDataset(
            train_img_features, train_text_features, train_label_pos)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, drop_last=False)
        if len(val_label_pos) > 0:
            val_dataset = TensorDataset(
                val_img_features, val_text_features, val_label_pos)
            val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        else:
            val_loader = None
    else:
        from torch.utils.data import Dataset as TorchDataset

        class _SampleDS(TorchDataset):
            def __init__(self, ds, idx_list, text_feats):
                self.ds = ds
                self.idx = list(idx_list)
                self.text_feats = text_feats
            def __len__(self):
                return len(self.idx)
            def __getitem__(self, i):
                s = self.ds[self.idx[i]]
                return {
                    "images": s["images"],
                    "text_feat": self.text_feats[i],
                    "label_pos": cid_to_pos[
                        _remap_label_for_task(int(s["failure_label"]), task)],
                }

        def _collate(batch):
            return {
                "images": [b["images"] for b in batch],
                "text_feat": torch.stack([b["text_feat"] for b in batch], dim=0),
                "labels_pos": torch.tensor(
                    [b["label_pos"] for b in batch], dtype=torch.long),
            }

        train_ds = _SampleDS(raw_train, train_only_idx, train_text_features)
        val_ds = (_SampleDS(raw_train, val_idx_split, val_text_features)
                  if val_idx_split else None)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, drop_last=False, collate_fn=_collate)
        val_loader = (DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=_collate)
            if val_ds is not None else None)

    image_proj = _build_proj_head(_DINO_HIDDEN).to(device)
    text_proj = _build_proj_head(_TEXT_HIDDEN).to(device)
    log_temp = nn.Parameter(torch.tensor(math.log(1.0), device=device))

    contrastive_loss = InfoNCELoss(temperature=1.0)

    proj_params = list(image_proj.parameters()) + list(text_proj.parameters()) + [log_temp]
    if variant == "frozen":
        optimizer = torch.optim.AdamW(
            proj_params, lr=LR_PROJ, weight_decay=0.01)
    else:
        encoder_params = [p for p in img_encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": LR_ENCODER, "name": "dino_lora"},
                {"params": proj_params,    "lr": LR_PROJ,    "name": "proj_and_temp"},
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
             f"early_stopping={use_early_stopping}")

    epoch_losses: list[float] = []
    epoch_val_losses: list[float] = []
    epoch_log_temp: list[float] = []
    best_val = float("inf")
    epochs_since_improvement = 0
    early_stopped_at = None

    def _embed_image_from_features(img_feats):
        return F.normalize(image_proj(img_feats), dim=-1)

    def _embed_image_from_pixels(images_list):
        flat_images = [img for sublist in images_list for img in sublist]
        n_per_sample = [len(sub) for sub in images_list]
        inputs = img_processor(images=flat_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = img_encoder(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :]
        cls = torch.stack(
            [p.mean(dim=0) for p in cls.split(n_per_sample, dim=0)],
            dim=0,
        )
        return F.normalize(image_proj(cls), dim=-1)

    def _embed_text(text_feats):
        return F.normalize(text_proj(text_feats), dim=-1)

    def _step_loss(img_emb, text_emb):
        # Clamp the learnable temperature so training stays stable
        # (analogous to CLIP's logit_scale clamp).
        tau = log_temp.exp().clamp(min=1e-2, max=1e2)
        return contrastive_loss(img_emb, features_b=text_emb, temp_override=tau)

    def _val_loss():
        image_proj.train(False); text_proj.train(False)
        if variant == "lora":
            img_encoder.train(False)
        embeds_img: list = []; embeds_txt: list = []
        with torch.no_grad():
            if variant == "frozen":
                for img_f, txt_f, _ in val_loader:
                    img_e = _embed_image_from_features(img_f.to(device))
                    txt_e = _embed_text(txt_f.to(device))
                    embeds_img.append(img_e); embeds_txt.append(txt_e)
            else:
                for batch in val_loader:
                    img_e = _embed_image_from_pixels(batch["images"])
                    txt_e = _embed_text(batch["text_feat"].to(device))
                    embeds_img.append(img_e); embeds_txt.append(txt_e)
        image_proj.train(True); text_proj.train(True)
        if variant == "lora":
            img_encoder.train(True)
        if not embeds_img:
            return float("nan")
        img_all = torch.cat(embeds_img, dim=0)
        txt_all = torch.cat(embeds_txt, dim=0)
        return float(_step_loss(img_all, txt_all).detach())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for epoch in range(EPOCHS):
        image_proj.train(True); text_proj.train(True)
        if variant == "lora":
            img_encoder.train(True)
        running = 0.0; n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if variant == "frozen":
                img_f, txt_f, _pos = batch
                img_emb = _embed_image_from_features(img_f.to(device))
                txt_emb = _embed_text(txt_f.to(device))
            else:
                img_emb = _embed_image_from_pixels(batch["images"])
                txt_emb = _embed_text(batch["text_feat"].to(device))
            loss = _step_loss(img_emb, txt_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()); n_batches += 1
        avg = running / max(1, n_batches)
        epoch_losses.append(avg)
        epoch_log_temp.append(float(log_temp.detach()))

        val_loss = None
        if use_early_stopping:
            val_loss = _val_loss()
            epoch_val_losses.append(val_loss)
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            log.info(f"  [DINOv2-CLIP ep {epoch+1:02d}/{EPOCHS}] loss={avg:.4f} "
                     f"val={val_loss:.4f} best={best_val:.4f} "
                     f"log_temp={float(log_temp):.4f}")
        else:
            log.info(f"  [DINOv2-CLIP ep {epoch+1:02d}/{EPOCHS}] loss={avg:.4f} "
                     f"log_temp={float(log_temp):.4f}")

        blob["train_progress"] = {
            "epochs_done": epoch + 1,
            "epoch_losses": epoch_losses,
            "epoch_val_losses": epoch_val_losses,
            "epoch_log_temp": epoch_log_temp,
            "best_val_loss": best_val if use_early_stopping else None,
        }
        save_results_atomically(blob, metrics_path)

        if use_early_stopping and epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped_at = epoch + 1
            log.info(f"  [DINOv2-CLIP] early stopping at epoch {early_stopped_at}")
            break

    train_time = time.time() - t0
    blob["train"] = {
        "variant": variant, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr_proj": LR_PROJ, "lr_encoder": LR_ENCODER,
        "epoch_losses": epoch_losses,
        "epoch_val_losses": epoch_val_losses,
        "epoch_log_temp": epoch_log_temp,
        "final_log_temp": float(log_temp.detach()),
        "best_val_loss": best_val if use_early_stopping else None,
        "early_stopped_at_epoch": early_stopped_at,
        "train_time_sec": round(train_time, 2),
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                            if torch.cuda.is_available() else None),
    }
    save_results_atomically(blob, metrics_path)

    log.info("Encoding class-prototype texts and projecting ...")
    proto_texts, proto_class_ids = _prototype_texts_for_task(task)
    present_set = set(present_classes)
    kept = [(t, cid) for t, cid in zip(proto_texts, proto_class_ids)
            if cid in present_set]
    if not kept:
        raise RuntimeError(
            "No class prototypes survived the present-classes filter "
            f"(present={present_classes}, proto_class_ids={proto_class_ids}). "
            "Cannot run DINOv2-CLIP inference.")
    proto_texts_kept = [t for t, _ in kept]
    proto_class_ids_kept = [cid for _, cid in kept]
    proto_pos_in_present = [cid_to_pos[cid] for cid in proto_class_ids_kept]
    log.info(f"  using {len(kept)} prototypes for present classes "
             f"{[present_label_names[p] for p in proto_pos_in_present]}")

    image_proj.train(False); text_proj.train(False)
    if variant == "lora":
        img_encoder.train(False)

    with torch.no_grad():
        proto_text_features = text_encoder.encode(
            proto_texts_kept, convert_to_tensor=True, show_progress_bar=False,
        ).to(device).float()
        proto_emb = _embed_text(proto_text_features)

    test_predictions: list[str] = []
    test_ground_truths: list[str] = []
    t_eval = time.time()
    proto_emb_cpu = proto_emb.cpu()
    with torch.no_grad():
        if variant == "frozen":
            test_img_features, test_labels = _encode_image_indices(
                raw_eval, test_indices)
            test_label_pos = [cid_to_pos[int(x.item())] for x in test_labels]
            for start in range(0, len(test_img_features), BATCH_SIZE):
                f_batch = test_img_features[start:start + BATCH_SIZE].to(device)
                img_e = _embed_image_from_features(f_batch).cpu()
                for vec, gt_pos in zip(
                        img_e, test_label_pos[start:start + BATCH_SIZE]):
                    sims = (vec.unsqueeze(0) @ proto_emb_cpu.T).squeeze(0)
                    pred_proto = int(sims.argmax().item())
                    pred_pos_present = proto_pos_in_present[pred_proto]
                    if task == "binary":
                        test_predictions.append(_binary_label_string(pred_pos_present))
                        test_ground_truths.append(_binary_label_string(int(gt_pos)))
                    else:
                        test_predictions.append(present_label_names[pred_pos_present])
                        test_ground_truths.append(present_label_names[int(gt_pos)])
        else:
            for start in range(0, len(test_indices), BATCH_SIZE):
                batch_idx = test_indices[start:start + BATCH_SIZE]
                samples = [raw_eval[int(i)] for i in batch_idx]
                images_list = [s["images"] for s in samples]
                gt_pos_list = [cid_to_pos[
                    _remap_label_for_task(int(s["failure_label"]), task)]
                    for s in samples]
                img_e = _embed_image_from_pixels(images_list).cpu()
                for vec, gt_pos in zip(img_e, gt_pos_list):
                    sims = (vec.unsqueeze(0) @ proto_emb_cpu.T).squeeze(0)
                    pred_proto = int(sims.argmax().item())
                    pred_pos_present = proto_pos_in_present[pred_proto]
                    if task == "binary":
                        test_predictions.append(_binary_label_string(pred_pos_present))
                        test_ground_truths.append(_binary_label_string(int(gt_pos)))
                    else:
                        test_predictions.append(present_label_names[pred_pos_present])
                        test_ground_truths.append(present_label_names[int(gt_pos)])
    eval_time = time.time() - t_eval

    if task == "binary":
        metric_label_names = ["success", "failure"]
    else:
        metric_label_names = list(present_label_names)
    m = compute_classification_metrics(
        test_ground_truths, test_predictions, metric_label_names)
    log.info(f"DINOv2-CLIP eval: acc={m['accuracy']:.4f} "
             f"f1_macro={m['f1_macro']:.4f}")
    if m.get("majority_class_warning"):
        log.warning(m["majority_class_warning"])

    blob.update({
        "metrics": m,
        "eval_time_sec": round(eval_time, 2),
        "predictions": test_predictions,
        "ground_truths": test_ground_truths,
        "prototype_texts_used": proto_texts_kept,
        "prototype_class_ids": proto_class_ids_kept,
        "present_classes": present_classes,
        "present_label_names": present_label_names,
        "n_classes_present": k_present,
        "n_eval_dropped_absent_class": n_eval_dropped,
        "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": True,
    })
    finalize_metrics_schema(blob)
    save_results_atomically(blob, metrics_path)
    # Persist image_proj + text_proj + log_temp (+ LoRA adapter if
    # applicable) and the cached class-prototype embeddings — added
    # 2026-05-11 for cross-domain eval reuse. The text encoder itself is
    # always frozen so we don't save it.
    head_payload = {
        "method": "dino_clip",
        "variant": variant,
        "image_proj_state": image_proj.state_dict(),
        "text_proj_state": text_proj.state_dict(),
        "log_temp": float(log_temp.detach()),
        "proto_emb": proto_emb.detach().cpu(),
        "proto_class_ids": proto_class_ids,
        "encoder_lora_state": (
            img_encoder.state_dict() if variant == "lora" else None),
        "present_classes": present_classes,
        "present_label_names": present_label_names,
    }
    torch.save(head_payload, run_dir / "head.pt")
    log.info(f"persisted head -> {run_dir / 'head.pt'}")
    write_done_flag(run_dir, {
        "method": "dino_clip",
        "variant": variant,
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
    })
    return blob


# ---------------------------------------------------------------------------
# Eval-only mode — load image_proj + text_proj + log_temp + proto_emb.
# ---------------------------------------------------------------------------

def eval_dino_clip(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    ckpt_dir = Path(config["from_checkpoint"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(
        run_dir, name=f"dino_clip.eval.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    log.info(f"=== eval_dino_clip | from_checkpoint={ckpt_dir} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "dino_clip",
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
    proto_emb = head["proto_emb"].to(device)
    proto_class_ids = head["proto_class_ids"]

    img_processor = AutoImageProcessor.from_pretrained(_DINO_MODEL)
    img_encoder = Dinov2Model.from_pretrained(_DINO_MODEL, torch_dtype=torch.float32)
    img_encoder.to(device)
    if variant == "lora":
        img_encoder = _apply_dino_lora(img_encoder, r=16, alpha=16, dropout=0.05)
        img_encoder.load_state_dict(head["encoder_lora_state"], strict=False)
    img_encoder.train(False)
    for p in img_encoder.parameters():
        p.requires_grad_(False)

    image_proj = _build_proj_head(_DINO_HIDDEN).to(device)
    image_proj.load_state_dict(head["image_proj_state"])
    image_proj.train(False)
    # text_proj is also persisted, but we use it only via the cached
    # proto_emb (already projected), so no need to instantiate it here.

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

    # Map prototype class IDs to positions in present_classes (binary task
    # uses only the first 2 prototypes, for example).
    proto_pos_in_present = [cid_to_pos[c] for c in proto_class_ids
                            if c in cid_to_pos]
    proto_emb_kept = proto_emb[[i for i, c in enumerate(proto_class_ids)
                                if c in cid_to_pos]]
    log.info(f"  using {len(proto_pos_in_present)} prototypes for "
             f"{[present_label_names[p] for p in proto_pos_in_present]}")

    label_names = list(present_label_names)
    predictions: list[str] = []
    ground_truths: list[str] = []
    t_eval = time.time()

    ENCODE_CHUNK = min(BATCH_SIZE, 32)
    with torch.no_grad():
        for start in range(0, len(test_indices), ENCODE_CHUNK):
            batch_idx = test_indices[start:start + ENCODE_CHUNK]
            samples = [raw_eval[int(i)] for i in batch_idx]
            flat_images: list = []
            n_per_sample: list[int] = []
            for s in samples:
                flat_images.extend(s["images"])
                n_per_sample.append(len(s["images"]))
            inputs = img_processor(images=flat_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            out = img_encoder(pixel_values=pixel_values)
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.stack(
                [p.mean(dim=0) for p in cls.split(n_per_sample, dim=0)],
                dim=0,
            )
            e = F.normalize(image_proj(cls), dim=-1)
            for vec_idx, sample in enumerate(samples):
                vec = e[vec_idx]
                sims = (vec.unsqueeze(0) @ proto_emb_kept.T).squeeze(0)
                pred_proto = int(sims.argmax().item())
                pred_pos = proto_pos_in_present[pred_proto]
                gt_label_id = _remap_label_for_task(
                    int(sample["failure_label"]), task)
                gt_pos = cid_to_pos[gt_label_id]
                predictions.append(label_names[pred_pos])
                ground_truths.append(label_names[gt_pos])
    eval_time = time.time() - t_eval

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"DINOv2-CLIP eval-only: acc={m['accuracy']:.4f} "
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
        "method": "dino_clip", "variant": variant,
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
        "eval_only": True,
    })
    return blob
