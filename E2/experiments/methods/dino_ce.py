"""DINOv2-CE: pure-vision supervised classification for failure detection.

A "no language model, no contrastive" baseline. Strips out the VLM
entirely and asks: how far does a dedicated self-supervised vision
encoder + classifier head get on this task?

Architecture
------------
    inputs   : 6 viewpoint PIL images (no text — task / subtask are
               ignored).
    encoder  : facebook/dinov2-large (300M params, 1024-d CLS token,
               trained with DINO/iBOT self-supervision).
    pool     : mean over the 6 per-viewpoint CLS tokens.
    head     : Linear(1024, 512) -> BatchNorm1d -> ReLU -> Dropout(0.3)
               -> Linear(512, K)   where K = len(present_classes).
    loss     : nn.CrossEntropyLoss().
    inference: argmax(classifier(features)).

Variants (config["variant"])
----------------------------
    frozen - DINOv2 weights frozen. Features for the train + val + test
             splits are extracted once with the frozen encoder and cached
             in memory; only the classifier head is trained against those
             cached features. ~50 MB for 12 358 train samples x 1024 fp32.
             Hyperparams: lr=1e-3, epochs=10, batch_size=32.
    lora   - LoRA r=16 on DINOv2 self-attention (query / key / value).
             Classifier head trains from scratch. The encoder runs every
             step (no caching) so the LoRA gradient flows. Hyperparams:
             lr_encoder=1e-4, lr_classifier=1e-3, epochs=5, batch_size=32.

Why this method matters
-----------------------
SFT compares "VLM-as-classifier" against full fine-tuning of Qwen3.5-4B.
DINOv2-CE asks the inverse question: with no language tokens, no chat
template, no VQA decoding - just a 300M ViT-L/14 + a 0.5M-param head -
can pure vision crack failure detection? It also gives the cheapest
sanity check for whether the dataset is learnable at all.
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


def _build_classifier(in_dim: int, n_classes: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, n_classes),
    )


def _apply_dino_lora(encoder, *, r: int, alpha: int, dropout: float):
    """Wrap DINOv2 self-attention in LoRA adapters."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        # Hugging Face Dinov2SelfAttention uses query / key / value names.
        # We deliberately skip "dense" - that name is shared with the MLP
        # module, which would balloon trainable-param count without a
        # principled reason under this method.
        target_modules=["query", "key", "value"],
        bias="none",
    )
    return get_peft_model(encoder, cfg)


def train_dino_ce(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    variant = str(config.get("variant", "frozen"))
    if variant not in ("frozen", "lora"):
        raise ValueError(f"variant must be 'frozen' or 'lora', got {variant!r}")

    run_dir = Path(config["run_dir"])
    log = setup_run_logger(run_dir, name=f"dino_ce.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"

    log.info(f"=== train_dino_ce | exp_id={config.get('exp_id')} task={task} "
             f"variant={variant} seed={seed} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "dino_ce",
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
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoImageProcessor, Dinov2Model
    from data.dataset import GuardianDataset
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

    # Hyperparameters.
    if variant == "frozen":
        EPOCHS = int(config.get("epochs", 10))
        BATCH_SIZE = int(config.get("batch_size", 32))
        LR = float(config.get("lr", 1e-3))
        LR_ENCODER = None
    else:
        EPOCHS = int(config.get("epochs", 5))
        BATCH_SIZE = int(config.get("batch_size", 32))
        LR = float(config.get("lr", 1e-3))
        LR_ENCODER = float(config.get("lr_encoder", 1e-4))
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
        log.info("DINOv2 frozen - only the classifier head will train.")
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
    log.info(f"task '{task}' variant '{variant}': present classes = "
             f"{present_classes}  ({len(present_classes)} of "
             f"{_task_n_classes(task)}) names={present_label_names}")
    cid_to_pos = {c: i for i, c in enumerate(present_classes)}
    k_present = len(present_classes)

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
    if n_eval_dropped:
        log.info(f"  dropped {n_eval_dropped} test samples whose true "
                 f"class is absent from train")

    log.info(f"data: train={len(train_only_idx)}  val={len(val_idx_split)}  "
             f"test={len(test_indices)}")

    # Cap the encoder forward chunk so memory stays bounded even when the
    # contrastive batch_size is large (e.g. 512). BATCH_SIZE drives the
    # head's training batches, not how many images we encode in one shot.
    ENCODE_CHUNK = min(BATCH_SIZE, 32)

    def _encode_indices(ds, indices, *, train_mode: bool):
        """Run DINOv2 over each sample's images and mean-pool the per-image
        CLS tokens into a single 1024-d feature. The per-sample image count
        is read from the data (RLBench=8, BDV2=2). Returns
        (features [N, 1024], labels [N])."""
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
            with torch.set_grad_enabled(train_mode):
                out = encoder(pixel_values=pixel_values)
            cls = out.last_hidden_state[:, 0, :]
            cls = torch.stack(
                [p.mean(dim=0) for p in cls.split(n_per_sample, dim=0)],
                dim=0,
            )
            feats.append(cls.detach().cpu() if not train_mode else cls)
            for s in samples:
                labels.append(_remap_label_for_task(int(s["failure_label"]), task))
        f = torch.cat(feats, dim=0)
        l = torch.tensor(labels, dtype=torch.long)
        return f, l

    classifier = _build_classifier(_DINO_HIDDEN, k_present).to(device)

    if variant == "frozen":
        log.info("Pre-computing frozen DINOv2 features (train + val + test) ...")
        t_feat = time.time()
        train_feats, train_labels = _encode_indices(
            raw_train, train_only_idx, train_mode=False)
        if val_idx_split:
            val_feats, val_labels = _encode_indices(
                raw_train, val_idx_split, train_mode=False)
        else:
            val_feats = torch.empty(0, _DINO_HIDDEN)
            val_labels = torch.empty(0, dtype=torch.long)
        test_feats, test_labels = _encode_indices(
            raw_eval, test_indices, train_mode=False)
        train_label_pos = torch.tensor(
            [cid_to_pos[int(x.item())] for x in train_labels], dtype=torch.long)
        val_label_pos = (torch.tensor(
            [cid_to_pos[int(x.item())] for x in val_labels], dtype=torch.long)
            if len(val_labels) > 0 else torch.empty(0, dtype=torch.long))
        test_label_pos = torch.tensor(
            [cid_to_pos[int(x.item())] for x in test_labels], dtype=torch.long)
        log.info(f"  feature extraction time: {time.time() - t_feat:.1f}s "
                 f"(train={tuple(train_feats.shape)} test={tuple(test_feats.shape)})")

        train_loader = DataLoader(
            TensorDataset(train_feats, train_label_pos),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False,
        )
        val_loader = (DataLoader(
            TensorDataset(val_feats, val_label_pos),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            if len(val_label_pos) > 0 else None)

        optimizer = torch.optim.AdamW(
            classifier.parameters(), lr=LR, weight_decay=0.01)
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
        test_ds_obj = _SampleDS(raw_eval, test_indices)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, drop_last=False, collate_fn=_collate)
        val_loader = (DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=_collate)
            if val_ds is not None else None)
        test_loader = DataLoader(
            test_ds_obj, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, collate_fn=_collate)

        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": LR_ENCODER, "name": "dino_lora"},
                {"params": list(classifier.parameters()), "lr": LR, "name": "classifier"},
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
    ce_loss = nn.CrossEntropyLoss()

    log.info(f"Training: variant={variant} epochs={EPOCHS} bs={BATCH_SIZE} "
             f"lr={LR}{' lr_encoder=' + str(LR_ENCODER) if LR_ENCODER else ''} "
             f"steps={total_optim_steps} warmup={warmup_steps} "
             f"early_stopping={use_early_stopping}")

    epoch_losses: list[float] = []
    epoch_val_losses: list[float] = []
    best_val = float("inf")
    epochs_since_improvement = 0
    early_stopped_at = None

    def _frozen_step(feat_batch, label_batch):
        feat_batch = feat_batch.to(device)
        label_batch = label_batch.to(device)
        logits = classifier(feat_batch)
        return ce_loss(logits, label_batch)

    def _lora_step(images_list, label_batch):
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
        logits = classifier(feats)
        return ce_loss(logits, label_batch.to(device))

    def _val_loss():
        classifier.train(False)
        if variant == "lora":
            encoder.train(False)
        total = 0.0; n = 0
        with torch.no_grad():
            if variant == "frozen":
                for f, l in val_loader:
                    loss = _frozen_step(f, l)
                    total += float(loss.item()) * f.size(0); n += f.size(0)
            else:
                for batch in val_loader:
                    loss = _lora_step(batch["images"], batch["labels"])
                    total += float(loss.item()) * batch["labels"].size(0)
                    n += batch["labels"].size(0)
        classifier.train(True)
        if variant == "lora":
            encoder.train(True)
        return total / max(1, n)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for epoch in range(EPOCHS):
        classifier.train(True)
        if variant == "lora":
            encoder.train(True)
        running = 0.0; n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if variant == "frozen":
                loss = _frozen_step(batch[0], batch[1])
            else:
                loss = _lora_step(batch["images"], batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()); n_batches += 1
        avg = running / max(1, n_batches)
        epoch_losses.append(avg)

        val_loss = None
        if use_early_stopping:
            val_loss = _val_loss()
            epoch_val_losses.append(val_loss)
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            log.info(f"  [DINOv2-CE ep {epoch+1:02d}/{EPOCHS}] loss={avg:.4f} "
                     f"val={val_loss:.4f} best={best_val:.4f} "
                     f"no_improve={epochs_since_improvement}")
        else:
            log.info(f"  [DINOv2-CE ep {epoch+1:02d}/{EPOCHS}] loss={avg:.4f}")

        blob["train_progress"] = {
            "epochs_done": epoch + 1,
            "epoch_losses": epoch_losses,
            "epoch_val_losses": epoch_val_losses,
            "best_val_loss": best_val if use_early_stopping else None,
        }
        save_results_atomically(blob, metrics_path)

        if use_early_stopping and epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped_at = epoch + 1
            log.info(f"  [DINOv2-CE] early stopping at epoch {early_stopped_at}")
            break

    train_time = time.time() - t0
    blob["train"] = {
        "variant": variant, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "lr_encoder": LR_ENCODER,
        "epoch_losses": epoch_losses,
        "epoch_val_losses": epoch_val_losses,
        "best_val_loss": best_val if use_early_stopping else None,
        "early_stopped_at_epoch": early_stopped_at,
        "train_time_sec": round(train_time, 2),
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                            if torch.cuda.is_available() else None),
    }
    save_results_atomically(blob, metrics_path)

    # Inference --------------------------------------------------------------
    log.info("Running test inference (argmax classifier) ...")
    classifier.train(False)
    if variant == "lora":
        encoder.train(False)
    label_names = list(present_label_names)
    predictions: list[str] = []
    ground_truths: list[str] = []
    t_eval = time.time()
    with torch.no_grad():
        if variant == "frozen":
            test_loader = DataLoader(
                TensorDataset(test_feats, test_label_pos),
                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            for f, l in test_loader:
                logits = classifier(f.to(device))
                pred_pos = logits.argmax(dim=-1).cpu().tolist()
                for p, g in zip(pred_pos, l.tolist()):
                    predictions.append(label_names[int(p)])
                    ground_truths.append(label_names[int(g)])
        else:
            for batch in test_loader:
                flat_images = [img for sublist in batch["images"] for img in sublist]
                n_per_sample = [len(sub) for sub in batch["images"]]
                inputs = processor(images=flat_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                out = encoder(pixel_values=pixel_values)
                cls = out.last_hidden_state[:, 0, :]
                feats = torch.stack(
                    [p.mean(dim=0) for p in cls.split(n_per_sample, dim=0)],
                    dim=0,
                )
                logits = classifier(feats)
                pred_pos = logits.argmax(dim=-1).cpu().tolist()
                gt_pos = batch["labels"].tolist()
                for p, g in zip(pred_pos, gt_pos):
                    predictions.append(label_names[int(p)])
                    ground_truths.append(label_names[int(g)])
    eval_time = time.time() - t_eval

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"DINOv2-CE eval: acc={m['accuracy']:.4f} "
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
    # Persist trained head (and LoRA adapter state if applicable) for
    # cross-domain eval reuse — added 2026-05-11 to support E5.x without
    # full retraining. See plan: smooth-skipping-boot.md Step 1.
    head_payload = {
        "method": "dino_ce",
        "variant": variant,
        "classifier_state": classifier.state_dict(),
        "encoder_lora_state": (
            encoder.state_dict() if variant == "lora" else None),
        "present_classes": present_classes,
        "present_label_names": present_label_names,
    }
    torch.save(head_payload, run_dir / "head.pt")
    log.info(f"persisted head -> {run_dir / 'head.pt'}")
    write_done_flag(run_dir, {
        "method": "dino_ce",
        "variant": variant,
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
    })
    return blob


# ---------------------------------------------------------------------------
# Eval-only mode (cross-domain transfer without retraining).
# Reuses the existing classifier head from a prior training run's head.pt.
# Added 2026-05-13 to support E5.2 (RLBench-trained → BDV2-test).
# ---------------------------------------------------------------------------

def eval_dino_ce(config: dict) -> dict:
    """Load a previously-trained DINOv2-CE head from
    config['from_checkpoint']/head.pt and evaluate on config['dataset_eval'].
    Skips training entirely. Writes metrics.json + done.flag to
    config['run_dir']."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    ckpt_dir = Path(config["from_checkpoint"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(
        run_dir, name=f"dino_ce.eval.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"

    log.info(f"=== eval_dino_ce | exp_id={config.get('exp_id')} task={task} "
             f"seed={seed} | from_checkpoint={ckpt_dir} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "dino_ce",
        "task": task,
        "seed": seed,
        "config": dict(config),
        "from_checkpoint": str(ckpt_dir),
        "eval_only": True,
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at": time.time(),
        "environment": capture_environment(),
    }
    save_results_atomically(blob, metrics_path)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoImageProcessor, Dinov2Model
    from data.dataset import GuardianDataset
    from evaluation.metrics import compute_classification_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = int(config.get("batch_size", 32))

    # Load head.pt to get classifier_state, encoder_lora_state, present_classes
    head_pt = ckpt_dir / "head.pt"
    if not head_pt.exists():
        raise FileNotFoundError(f"head.pt not found at {head_pt}")
    head = torch.load(head_pt, map_location="cpu", weights_only=False)
    variant = head.get("variant", "frozen")
    present_classes = head["present_classes"]
    present_label_names = head["present_label_names"]
    k_present = len(present_classes)
    cid_to_pos = {c: i for i, c in enumerate(present_classes)}
    log.info(f"loaded head.pt: variant={variant}  present_classes={present_classes}")

    # Build encoder (frozen, or with LoRA adapter)
    processor = AutoImageProcessor.from_pretrained(_DINO_MODEL)
    encoder = Dinov2Model.from_pretrained(_DINO_MODEL, torch_dtype=torch.float32)
    encoder.to(device)
    if variant == "lora":
        encoder = _apply_dino_lora(encoder, r=16, alpha=16, dropout=0.05)
        encoder.load_state_dict(head["encoder_lora_state"], strict=False)
    encoder.train(False)
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Build classifier and load state
    classifier = _build_classifier(_DINO_HIDDEN, k_present).to(device)
    classifier.load_state_dict(head["classifier_state"])
    classifier.train(False)

    # Load eval data
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
    test_label_pos = torch.tensor(
        [cid_to_pos[int(x.item())] for x in test_labels], dtype=torch.long)

    label_names = list(present_label_names)
    predictions: list[str] = []
    ground_truths: list[str] = []
    with torch.no_grad():
        test_loader = DataLoader(
            TensorDataset(test_feats, test_label_pos),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        for f, l in test_loader:
            logits = classifier(f.to(device))
            pred_pos = logits.argmax(dim=-1).cpu().tolist()
            for p, g in zip(pred_pos, l.tolist()):
                predictions.append(label_names[int(p)])
                ground_truths.append(label_names[int(g)])
    eval_time = time.time() - t_eval

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"DINOv2-CE eval-only: acc={m['accuracy']:.4f} "
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
        "method": "dino_ce", "variant": variant,
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
        "eval_only": True,
    })
    return blob
