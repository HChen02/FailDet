"""DA3-as-preprocessor → DINOv2 (frozen) → AttentionPool → CNN → projection
→ supervised InfoNCE → nearest-centroid inference.

Mirror of da3_dino_ce.py but with the InfoNCE loss + centroid evaluation
path from dino_cl.py. DA3 is never loaded here — depth maps are read from
the on-disk cache built by experiments/preprocess_da3.py.
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
from experiments.methods._da3_dino_common import (  # noqa: E402
    AttentionPool, CNNBlock, DINO_HIDDEN, build_projection_head,
    encode_depth_samples, load_dinov2_l,
)
from losses.infonce import InfoNCELoss  # noqa: E402


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


def train_da3_dino_cl(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    run_dir = Path(config["run_dir"])
    log = setup_run_logger(
        run_dir, name=f"da3_dino_cl.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    log.info(f"=== train_da3_dino_cl | exp_id={config.get('exp_id')} "
             f"task={task} seed={seed} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "da3_dino_cl",
        "task": task, "seed": seed,
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
    import numpy as np
    from data.dataset import GuardianDataset
    from evaluation.metrics import compute_classification_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = int(config.get("epochs", 30))
    BATCH_SIZE = int(config.get("batch_size", 32))
    LR = float(config.get("lr", 1e-2))
    TEMPERATURE = float(config.get("temperature", 1.0))
    EARLY_STOPPING_PATIENCE = int(config.get("early_stopping_patience", 2))
    ATTN_HEADS = int(config.get("attn_pool_heads", 4))
    CNN_HIDDEN = int(config.get("cnn_hidden", 512))
    CNN_LAYERS = int(config.get("cnn_layers", 3))
    CNN_DROPOUT = float(config.get("cnn_dropout", 0.1))
    EMBED_DIM = int(config.get("embed_dim", 128))

    raw_train = GuardianDataset(config["dataset_train"])
    raw_eval = GuardianDataset(config["dataset_eval"], corruption_name=config.get("corruption"), severity=config.get("severity"))

    train_idx = select_indices(
        len(raw_train),
        data_fraction=float(config.get("data_fraction", 1.0)),
        seed=seed,
    )
    val_fraction = float(config.get("val_fraction", 0.05))
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
    log.info(f"task '{task}': present classes = {present_classes} "
             f"({len(present_classes)} of {_task_n_classes(task)}) "
             f"names={present_label_names}")
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

    encoder = load_dinov2_l(device)
    log.info("Encoding depth-map CLS tokens for train/val/test ...")
    t_feat = time.time()
    train_feats, n_ref = encode_depth_samples(
        config["dataset_train"], train_only_idx, encoder, device)
    if val_idx_split:
        val_feats, _ = encode_depth_samples(
            config["dataset_train"], val_idx_split, encoder, device)
    else:
        val_feats = torch.empty(0, n_ref, DINO_HIDDEN)
    test_feats, _ = encode_depth_samples(
        config["dataset_eval"], test_indices, encoder, device)

    train_label_pos = torch.tensor(
        [cid_to_pos[_remap_label_for_task(int(raw_train[i]["failure_label"]), task)]
         for i in train_only_idx], dtype=torch.long)
    val_label_pos = (torch.tensor(
        [cid_to_pos[_remap_label_for_task(int(raw_train[i]["failure_label"]), task)]
         for i in val_idx_split], dtype=torch.long)
        if val_idx_split else torch.empty(0, dtype=torch.long))
    test_label_pos = torch.tensor(
        [cid_to_pos[_remap_label_for_task(int(raw_eval[i]["failure_label"]), task)]
         for i in test_indices], dtype=torch.long)
    log.info(f"  encode time: {time.time() - t_feat:.1f}s "
             f"(train={tuple(train_feats.shape)} test={tuple(test_feats.shape)})")
    del encoder
    torch.cuda.empty_cache()

    train_loader = DataLoader(
        TensorDataset(train_feats, train_label_pos),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = (DataLoader(
        TensorDataset(val_feats, val_label_pos),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        if len(val_label_pos) > 0 else None)
    test_loader = DataLoader(
        TensorDataset(test_feats, test_label_pos),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    attn_pool = AttentionPool(dim=DINO_HIDDEN, n_heads=ATTN_HEADS).to(device)
    cnn = CNNBlock(in_dim=DINO_HIDDEN, hidden=CNN_HIDDEN,
                   n_layers=CNN_LAYERS, dropout=CNN_DROPOUT).to(device)
    proj_head = build_projection_head(cnn.out_dim, embed_dim=EMBED_DIM).to(device)

    n_pool = sum(p.numel() for p in attn_pool.parameters() if p.requires_grad)
    n_cnn = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    n_proj = sum(p.numel() for p in proj_head.parameters() if p.requires_grad)
    log.info(f"params: attn_pool={n_pool:,}  cnn={n_cnn:,}  proj={n_proj:,}  "
             f"total={n_pool + n_cnn + n_proj:,}")
    blob["decoder"] = {
        "attn_pool_heads": ATTN_HEADS, "attn_pool_params": n_pool,
        "cnn_layers": CNN_LAYERS, "cnn_hidden": CNN_HIDDEN, "cnn_params": n_cnn,
        "proj_params": n_proj, "embed_dim": EMBED_DIM,
        "temperature": TEMPERATURE, "n_frames": n_ref,
    }

    head_params = (list(attn_pool.parameters())
                   + list(cnn.parameters())
                   + list(proj_head.parameters()))
    optimizer = torch.optim.AdamW(head_params, lr=LR, weight_decay=0.01)

    use_early_stopping = val_loader is not None and EARLY_STOPPING_PATIENCE > 0
    blob["early_stopping"] = {
        "enabled": use_early_stopping,
        "patience": EARLY_STOPPING_PATIENCE if use_early_stopping else None,
        "n_val": len(val_idx_split),
    }

    total_steps = max(1, EPOCHS * max(1, len(train_loader)))
    warmup_steps = max(1, int(round(total_steps * 0.1)))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    info_nce = InfoNCELoss(temperature=TEMPERATURE)

    def _embed(feats):
        pooled, attn_w = attn_pool(feats)
        return F.normalize(proj_head(cnn(pooled)), dim=-1), attn_w

    def _val_loss():
        attn_pool.train(False); cnn.train(False); proj_head.train(False)
        embeds: list = []; all_l: list = []
        with torch.no_grad():
            for f, l in val_loader:
                e, _ = _embed(f.to(device))
                embeds.append(e); all_l.append(l.to(device))
        attn_pool.train(True); cnn.train(True); proj_head.train(True)
        if not embeds:
            return float("inf")
        emb = torch.cat(embeds, dim=0); lab = torch.cat(all_l, dim=0)
        return float(info_nce(emb, labels=lab).item())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    epoch_losses: list[float] = []
    epoch_val_losses: list[float] = []
    epoch_intra: list[float] = []
    epoch_inter: list[float] = []
    best_val = float("inf")
    epochs_since_improvement = 0
    early_stopped_at = None

    log.info(f"Training: epochs={EPOCHS} bs={BATCH_SIZE} lr={LR} "
             f"tau={TEMPERATURE} steps={total_steps} warmup={warmup_steps}")
    t0 = time.time()
    for epoch in range(EPOCHS):
        attn_pool.train(True); cnn.train(True); proj_head.train(True)
        running = 0.0; n_batches = 0
        batch_intras: list[float] = []; batch_inters: list[float] = []
        for f, l in train_loader:
            optimizer.zero_grad(set_to_none=True)
            f = f.to(device); l = l.to(device)
            emb, _ = _embed(f)
            loss = info_nce(emb, labels=l)
            if not torch.isfinite(loss) or loss.requires_grad is False:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()); n_batches += 1
            with torch.no_grad():
                B = emb.size(0)
                sim = (emb @ emb.T).cpu()
                eq = (l.unsqueeze(0) == l.unsqueeze(1)).cpu()
                eye = torch.eye(B, dtype=torch.bool)
                pos = eq & ~eye; neg = ~eq & ~eye
                if pos.any(): batch_intras.append(float(sim[pos].mean()))
                if neg.any(): batch_inters.append(float(sim[neg].mean()))
        avg = running / max(1, n_batches)
        epoch_losses.append(avg)
        ei = float(np.mean(batch_intras)) if batch_intras else 0.0
        er = float(np.mean(batch_inters)) if batch_inters else 0.0
        epoch_intra.append(ei); epoch_inter.append(er)

        val_loss = None
        if use_early_stopping:
            val_loss = _val_loss()
            epoch_val_losses.append(val_loss)
            if val_loss < best_val - 1e-4:
                best_val = val_loss; epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            log.info(f"  [DA3+DINO CL ep {epoch+1:02d}/{EPOCHS}] "
                     f"loss={avg:.4f} val={val_loss:.4f} best={best_val:.4f} "
                     f"intra={ei:.3f} inter={er:.3f} "
                     f"no_improve={epochs_since_improvement}")
        else:
            log.info(f"  [DA3+DINO CL ep {epoch+1:02d}/{EPOCHS}] "
                     f"loss={avg:.4f} intra={ei:.3f} inter={er:.3f}")

        blob["train_progress"] = {
            "epochs_done": epoch + 1,
            "epoch_losses": epoch_losses,
            "epoch_val_losses": epoch_val_losses,
            "epoch_intra_class_sim": epoch_intra,
            "epoch_inter_class_sim": epoch_inter,
            "best_val_loss": best_val if use_early_stopping else None,
        }
        save_results_atomically(blob, metrics_path)
        if use_early_stopping and epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped_at = epoch + 1
            log.info(f"  early stopping at epoch {early_stopped_at}")
            break

    train_time = time.time() - t0
    blob["train"] = {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "temperature": TEMPERATURE,
        "epoch_losses": epoch_losses,
        "epoch_val_losses": epoch_val_losses,
        "epoch_intra_class_sim": epoch_intra,
        "epoch_inter_class_sim": epoch_inter,
        "best_val_loss": best_val if use_early_stopping else None,
        "early_stopped_at_epoch": early_stopped_at,
        "train_time_sec": round(train_time, 2),
        "peak_gpu_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                            if torch.cuda.is_available() else None),
    }
    save_results_atomically(blob, metrics_path)

    # Build centroids ------------------------------------------------------
    log.info("Building train centroids in 128-d projected space ...")
    attn_pool.train(False); cnn.train(False); proj_head.train(False)
    centroid_sum = torch.zeros(k_present, EMBED_DIM)
    centroid_counts: dict[int, int] = defaultdict(int)
    cent_loader = DataLoader(
        TensorDataset(train_feats, train_label_pos),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for f, l in cent_loader:
            f = f.to(device)
            e, _ = _embed(f)
            e = e.cpu()
            for vec, pos in zip(e, l.tolist()):
                centroid_sum[int(pos)] += vec
                centroid_counts[int(pos)] += 1
    centroids = torch.zeros_like(centroid_sum)
    for pos in range(k_present):
        if centroid_counts[pos] > 0:
            centroids[pos] = centroid_sum[pos] / centroid_counts[pos]
    centroids = F.normalize(centroids, dim=-1)
    log.info(f"  centroid counts: "
             f"{ {present_label_names[pos]: centroid_counts.get(pos, 0) for pos in range(k_present)} }")

    # Test inference -------------------------------------------------------
    label_names = list(present_label_names)
    predictions: list[str] = []
    ground_truths: list[str] = []
    all_attn: list[list[float]] = []
    t_eval = time.time()
    with torch.no_grad():
        for f, l in test_loader:
            f = f.to(device)
            e, attn_w = _embed(f)
            e = e.cpu()
            sims = e @ centroids.T
            pred_pos = sims.argmax(dim=-1).tolist()
            aw = attn_w.squeeze(1).detach().cpu().tolist()
            for p, g, a in zip(pred_pos, l.tolist(), aw):
                predictions.append(label_names[int(p)])
                ground_truths.append(label_names[int(g)])
                all_attn.append([float(x) for x in a])
    eval_time = time.time() - t_eval

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"DA3+DINO CL test: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")
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
        "attention_weights": {
            "n_test_samples": len(all_attn),
            "per_sample": [[round(float(x), 6) for x in row] for row in all_attn],
        },
        "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": True,
    })
    finalize_metrics_schema(blob)
    save_results_atomically(blob, metrics_path)

    head_payload = {
        "method": "da3_dino_cl",
        "attn_pool_state": attn_pool.state_dict(),
        "cnn_state": cnn.state_dict(),
        "proj_head_state": proj_head.state_dict(),
        "centroids": centroids.detach().cpu(),
        "attn_pool_heads": ATTN_HEADS,
        "cnn_hidden": CNN_HIDDEN, "cnn_layers": CNN_LAYERS,
        "cnn_dropout": CNN_DROPOUT,
        "embed_dim": EMBED_DIM, "temperature": TEMPERATURE,
        "n_frames_train": n_ref,
        "present_classes": present_classes,
        "present_label_names": present_label_names,
    }
    torch.save(head_payload, run_dir / "head.pt")
    log.info(f"persisted head -> {run_dir / 'head.pt'}")
    write_done_flag(run_dir, {
        "method": "da3_dino_cl",
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
    })
    return blob


# ---------------------------------------------------------------------------
# Eval-only mode
# ---------------------------------------------------------------------------

def eval_da3_dino_cl(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    run_dir = Path(config["run_dir"])
    log = setup_run_logger(
        run_dir, name=f"da3_dino_cl.eval.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    log.info(f"=== eval_da3_dino_cl | exp_id={config.get('exp_id')} "
             f"task={task} seed={seed} ===")

    ckpt_dir = Path(config["from_checkpoint"])
    head_path = ckpt_dir / "head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"head.pt not found at {head_path}")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    from data.dataset import GuardianDataset
    from evaluation.metrics import compute_classification_metrics

    device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = torch.load(head_path, map_location=device, weights_only=False)
    ATTN_HEADS = int(payload.get("attn_pool_heads", 4))
    CNN_HIDDEN = int(payload.get("cnn_hidden", 512))
    CNN_LAYERS = int(payload.get("cnn_layers", 3))
    CNN_DROPOUT = float(payload.get("cnn_dropout", 0.1))
    EMBED_DIM = int(payload.get("embed_dim", 128))
    present_classes = list(payload["present_classes"])
    present_label_names = list(payload["present_label_names"])
    k_present = len(present_classes)
    cid_to_pos = {c: i for i, c in enumerate(present_classes)}

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

    encoder = load_dinov2_l(device)
    test_feats, n_ref = encode_depth_samples(
        config["dataset_eval"], test_indices, encoder, device)
    del encoder
    torch.cuda.empty_cache()
    test_label_pos = torch.tensor(
        [cid_to_pos[_remap_label_for_task(int(raw_eval[i]["failure_label"]), task)]
         for i in test_indices], dtype=torch.long)

    attn_pool = AttentionPool(dim=DINO_HIDDEN, n_heads=ATTN_HEADS).to(device)
    cnn = CNNBlock(in_dim=DINO_HIDDEN, hidden=CNN_HIDDEN, n_layers=CNN_LAYERS,
                   dropout=CNN_DROPOUT).to(device)
    proj_head = build_projection_head(cnn.out_dim, embed_dim=EMBED_DIM).to(device)
    attn_pool.load_state_dict(payload["attn_pool_state"])
    cnn.load_state_dict(payload["cnn_state"])
    proj_head.load_state_dict(payload["proj_head_state"])
    attn_pool.train(False); cnn.train(False); proj_head.train(False)
    centroids = payload["centroids"].to(device)

    test_loader = DataLoader(
        TensorDataset(test_feats, test_label_pos),
        batch_size=32, shuffle=False, num_workers=0)

    label_names = list(present_label_names)
    predictions: list[str] = []
    ground_truths: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for f, l in test_loader:
            f = f.to(device)
            pooled, _ = attn_pool(f)
            e = F.normalize(proj_head(cnn(pooled)), dim=-1)
            sims = e @ centroids.T
            pred_pos = sims.argmax(dim=-1).cpu().tolist()
            for p, g in zip(pred_pos, l.tolist()):
                predictions.append(label_names[int(p)])
                ground_truths.append(label_names[int(g)])
    eval_time = time.time() - t0

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"DA3+DINO CL eval-only: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")

    blob = {
        "exp_id": config.get("exp_id"),
        "method": "da3_dino_cl", "task": task, "seed": seed,
        "config": dict(config),
        "metrics": m,
        "eval_time_sec": round(eval_time, 2),
        "predictions": predictions, "ground_truths": ground_truths,
        "present_classes": present_classes,
        "present_label_names": present_label_names,
        "n_classes_present": k_present,
        "n_eval_dropped_absent_class": n_eval_dropped,
        "eval_only": True,
        "from_checkpoint": str(ckpt_dir),
        "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": True,
    }
    finalize_metrics_schema(blob)
    save_results_atomically(blob, metrics_path)
    write_done_flag(run_dir, {
        "method": "da3_dino_cl", "eval_only": True,
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
    })
    return blob
