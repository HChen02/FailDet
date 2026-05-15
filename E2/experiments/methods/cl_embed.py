"""CL-Embed: contrastive learning through the FULL Qwen3.5-4B VLM
using last-token (pre-generation) embeddings.

Method shape (literature: VLM2Vec ICLR 2025; VladVA 2024; CAFe 2025
inspired the joint CE+CL variant CL+CE — but CL-Embed itself is pure CL):

    inputs   : 6 viewpoint images + Task / Subtask prompt (USER turn only,
               NO assistant turn — we are extracting an embedding, not
               generating).
    encoder  : full Qwen3.5-4B with QLoRA on ALL layers (vision + language,
               attention + mlp), same setup as SFT/CL+CE.
    head     : Linear(hidden_dim, 512) → GELU → Linear(512, 128) → L2-norm.
    loss     : supervised InfoNCE (same-class positives) on the projected
               last-token hidden state, with task-aware label remapping
               (binary 0/1 ; 7class 0..6 with success filtered ; 8class 0..7).
    eval     : nearest train-class centroid in 128-d projected space.

GradCache (Gao et al. 2021; VLM2Vec ICLR 2025) is applied so the
contrastive batch is decoupled from the per-step VRAM budget. Each
optimizer step accumulates ``gc_batch_size`` samples in mini-batches of
``batch_size``: Step 1 forwards every mini-batch under ``no_grad`` and
caches the projected embedding (RNG state captured for replay); Step 2
computes the full InfoNCE on the macro-batch and back-propagates onto
the cached embeddings; Step 3 re-forwards each mini-batch with grad
(restoring the captured RNG so dropout matches Step 1) and chains the
gradient through ``surrogate = (projected · cached.grad).sum()``. This
gives 64× more negatives at the same VRAM footprint as ``batch_size=4``,
at the cost of a second forward per mini-batch. With
``gc_batch_size <= batch_size`` the loop falls back to the original
gradient-accumulation path.

This is the natural "CL through the VLM, embedding-first" baseline.
The model never has to generate text — at training time we just optimize
the projected last-token to cluster by label, and at inference we
classify by nearest centroid (~50 FPS instead of ~5 FPS).

Compared to CL+CE: same architecture and forward path, but without the
CE term — pure InfoNCE through the VLM. So this isolates "how well does
contrastive-only training of the full VLM separate failure classes"
without the auxiliary supervision of next-token CE.

Compared to CL-FT: CL-FT runs vision-only (LoRA on the vision tower
qkv/proj). CL-Embed runs the FULL VLM forward (vision + language + cross
attention) and extracts the [EOS] embedding — much richer information
than vision features alone, but more expensive per-step.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.methods.common import (  # noqa: E402
    UNIFIED_LABEL_NAMES, apply_chat_template_safe, capture_environment,
    compute_present_classes, finalize_metrics_schema,
    save_results_atomically, select_indices, set_seed, setup_run_logger,
    write_done_flag,
)
from data.balanced_sampler import ClassBalancedBatchSampler  # noqa: E402


# Prompt — user-side only (no assistant turn). Matches the user spec for
# CL-Embed: shorter than SFT's PROMPT_TEMPLATE so the embedding is anchored
# in task/subtask context rather than the enumerated answer vocabulary.
_PROMPT = ("Task: {task}\n"
           "Subtask: {subtask}\n"
           "Classify the robot manipulation outcome.")


def _build_user_messages(sample: dict) -> list[dict]:
    user_content: list[dict] = [
        {"type": "image", "image": img} for img in sample["images"]
    ]
    user_content.append({
        "type": "text",
        "text": _PROMPT.format(
            task=sample["task_instruction"],
            subtask=sample["detailed_subtask_name"],
        ),
    })
    return [{"role": "user", "content": user_content}]


# Task plumbing -------------------------------------------------------------

def _task_n_classes(task: str) -> int:
    return {"binary": 2, "7class": 7, "8class": 8}[task]


def _task_label_names(task: str) -> list[str]:
    if task == "binary":
        return ["success", "failure"]
    if task == "7class":
        return list(UNIFIED_LABEL_NAMES[1:])
    if task == "8class":
        return list(UNIFIED_LABEL_NAMES)
    raise ValueError(f"Unknown task: {task!r}")


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


def _discover_hidden(model) -> int:
    """Find the LM hidden size on a (PEFT-wrapped) Qwen3.5-VL — same logic
    as cl_ce.py / _vision_helpers.py."""
    for path in ("config", "base_model.model.config",
                 "base_model.config", "model.config"):
        cur = model
        ok = True
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if not ok:
            continue
        if hasattr(cur, "hidden_size") and isinstance(cur.hidden_size, int):
            return int(cur.hidden_size)
        for sub in ("text_config", "language_config"):
            if hasattr(cur, sub):
                tc = getattr(cur, sub)
                if hasattr(tc, "hidden_size"):
                    return int(tc.hidden_size)
    emb = model.get_input_embeddings()
    if emb is not None and hasattr(emb, "embedding_dim"):
        return int(emb.embedding_dim)
    raise RuntimeError("Could not discover model.config.hidden_size")


# Main entry ----------------------------------------------------------------

def train_cl_embed(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))

    BATCH_SIZE = int(config.get("batch_size", 4))
    if BATCH_SIZE < 4 and not bool(config.get("allow_degenerate", False)):
        raise AssertionError(
            f"CL-Embed requires batch_size >= 4 for the InfoNCE term to find "
            f"positive same-class pairs. Got {BATCH_SIZE}. Use "
            f"gradient_accumulation_steps to compensate VRAM."
        )

    run_dir = Path(config["run_dir"])
    log = setup_run_logger(run_dir, name=f"cl_embed.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    ckpt_dir = run_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== train_cl_embed | exp_id={config.get('exp_id')} task={task} "
             f"seed={seed} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "cl_embed",
        "task": task,
        "seed": seed,
        "config": dict(config),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at": time.time(),
        "environment": capture_environment(),
    }
    save_results_atomically(blob, metrics_path)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from unsloth import FastModel
    from data.dataset import GuardianDataset
    from losses.infonce import InfoNCELoss
    from evaluation.metrics import compute_classification_metrics

    device = "cuda"
    dtype = torch.bfloat16

    log.info("Loading Qwen3.5-4B (4-bit) + LoRA (all layers) ...")
    t0 = time.time()
    model, processor = FastModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0.05,
        random_state=seed,
    )
    blob["load_time_sec"] = round(time.time() - t0, 2)
    hidden_dim = _discover_hidden(model)
    log.info(f"hidden_dim = {hidden_dim}")
    proj_head = _build_proj_head(hidden_dim).to(device=device, dtype=torch.float32)

    # Data ------------------------------------------------------------------
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

    # Audit P2.1 — drop empty classes (e.g. wrong_state on RLBench train).
    present_classes, present_label_names, train_per_index_labels = (
        compute_present_classes(raw_train, train_only_idx, task))
    log.info(f"task '{task}': present classes (>=1 train sample) = "
             f"{present_classes}  ({len(present_classes)} of "
             f"{_task_n_classes(task)})  names={present_label_names}")
    cid_to_pos = {c: i for i, c in enumerate(present_classes)}

    log.info(f"train: {len(raw_train)} -> {len(train_only_idx)} train + "
             f"{len(val_idx_split)} val (after task='{task}' filter)")

    class CLEmbedDS(TorchDataset):
        def __init__(self, ds, idx_list, *, task: str):
            self.ds = ds
            self.idx = list(idx_list)
            self.task = task

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            sample = self.ds[self.idx[i]]
            return {
                "messages": _build_user_messages(sample),
                "images": sample["images"],
                "label": _remap_label_for_task(
                    int(sample["failure_label"]), self.task),
            }

    train_ds = CLEmbedDS(raw_train, train_only_idx, task=task)
    val_ds = CLEmbedDS(raw_train, val_idx_split, task=task) if val_idx_split else None

    def collate(batch):
        # Tokenize+process the batch the same way Unsloth's collator would.
        texts = [
            apply_chat_template_safe(processor, b["messages"],
                                     add_generation_prompt=True)
            for b in batch
        ]
        images = [b["images"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        out = processor(
            text=texts, images=images,
            return_tensors="pt", padding=True,
        )
        if not isinstance(out, dict):
            out = dict(out)
        out["labels_cl"] = labels
        return out

    GRAD_ACCUM = int(config.get("grad_accum", 4))
    EPOCHS = int(config.get("epochs", 2))
    LR = float(config.get("lr", 2e-4))
    WARMUP_RATIO = 0.1
    TEMPERATURE = float(config.get("temperature", 1.0))
    EARLY_STOPPING_PATIENCE = int(config.get("early_stopping_patience", 2))

    # GradCache: when gc_batch_size > batch_size, run the GradCache
    # optimization (cf. VLM2Vec, ICLR 2025) so the InfoNCE batch can be
    # made arbitrarily large without VRAM blowup. Set gc_batch_size=0 to
    # disable and fall back to plain gradient accumulation.
    GC_BATCH_SIZE = int(config.get("gc_batch_size", 256))
    USE_GRADCACHE = GC_BATCH_SIZE > BATCH_SIZE
    if USE_GRADCACHE and GC_BATCH_SIZE % BATCH_SIZE != 0:
        raise ValueError(
            f"gc_batch_size ({GC_BATCH_SIZE}) must be a multiple of "
            f"batch_size ({BATCH_SIZE})")
    gc_accumulation = (GC_BATCH_SIZE // BATCH_SIZE) if USE_GRADCACHE else 0

    # Sampler routing:
    #   - binary: dataset is naturally ~50/50 on RLBench-Fail and the
    #     InfoNCE positives are abundant in any random batch — use
    #     standard shuffle, skip the balanced sampler.
    #   - non-binary at bs >= num_classes: use ClassBalancedBatchSampler
    #     so every step has at least one sample of every present class
    #     (prevents the rare-class-zero-positive failure mode).
    #   - non-binary at bs <  num_classes: random shuffle with a loud
    #     warning (the balanced sampler can't guarantee one-per-class).
    if task == "binary":
        log.info(f"binary task — using standard random shuffle "
                 f"(bs={BATCH_SIZE}, k={len(present_classes)}; data is "
                 f"already ~50/50 so balanced sampling is unnecessary).")
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, drop_last=False, collate_fn=collate,
        )
    elif BATCH_SIZE >= len(present_classes):
        train_batch_sampler = ClassBalancedBatchSampler(
            labels=train_per_index_labels,
            batch_size=BATCH_SIZE,
            num_batches=max(1, len(train_only_idx) // BATCH_SIZE),
            seed=seed,
        )
        log.info(f"using ClassBalancedBatchSampler: bs={BATCH_SIZE}, "
                 f"k={len(present_classes)}, "
                 f"num_batches={len(train_batch_sampler)}")
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=0, collate_fn=collate,
        )
    else:
        log.warning(f"batch_size={BATCH_SIZE} < num_classes="
                    f"{len(present_classes)} — falling back to random "
                    f"shuffle (some batches may have 0 positives for "
                    f"some classes; consider --batch-size "
                    f">={len(present_classes)}).")
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, drop_last=False, collate_fn=collate,
        )
    val_loader = (DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, drop_last=False, collate_fn=collate,
    ) if val_ds is not None else None)
    use_early_stopping = val_loader is not None and EARLY_STOPPING_PATIENCE > 0
    blob["early_stopping"] = {
        "enabled": use_early_stopping,
        "patience": EARLY_STOPPING_PATIENCE if use_early_stopping else None,
        "n_val": len(val_idx_split),
    }

    contrastive_loss = InfoNCELoss(temperature=TEMPERATURE)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    proj_params = list(proj_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": LR, "name": "lora"},
            {"params": proj_params, "lr": LR, "name": "proj_head"},
        ],
        weight_decay=0.01,
    )

    if USE_GRADCACHE:
        steps_per_epoch = max(1, len(train_loader) // gc_accumulation)
        effective_batch = GC_BATCH_SIZE
    else:
        steps_per_epoch = max(1, math.ceil(len(train_loader) / GRAD_ACCUM))
        effective_batch = BATCH_SIZE * GRAD_ACCUM
    total_optim_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(round(total_optim_steps * WARMUP_RATIO)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if USE_GRADCACHE:
        log.info(
            f"Training (GradCache): epochs={EPOCHS} mini_batch={BATCH_SIZE} "
            f"gc_batch={GC_BATCH_SIZE} (accumulation={gc_accumulation}) "
            f"steps_per_epoch={steps_per_epoch} total_steps={total_optim_steps} "
            f"warmup={warmup_steps} lr={LR} τ={TEMPERATURE} "
            f"early_stopping={use_early_stopping} patience={EARLY_STOPPING_PATIENCE}"
        )
        if GRAD_ACCUM != 1:
            log.warning(
                f"grad_accum={GRAD_ACCUM} is ignored under GradCache "
                f"(effective batch is gc_batch_size={GC_BATCH_SIZE}).")
    else:
        log.info(
            f"Training (legacy GA): epochs={EPOCHS} bs={BATCH_SIZE} "
            f"ga={GRAD_ACCUM} effective_batch={effective_batch} lr={LR} "
            f"τ={TEMPERATURE} early_stopping={use_early_stopping} "
            f"patience={EARLY_STOPPING_PATIENCE}"
        )

    def _forward_embeddings(batch_on_dev) -> torch.Tensor:
        """Forward pass + L2-normalized projected last-token embedding."""
        with torch.autocast("cuda", dtype=dtype):
            out = model(**batch_on_dev, output_hidden_states=True)
        hs = out.hidden_states[-1]
        attn = batch_on_dev["attention_mask"]
        last_idx = attn.cumsum(dim=1).argmax(dim=1)
        eos = hs[torch.arange(hs.size(0), device=device), last_idx]
        return F.normalize(proj_head(eos.float()), dim=-1)

    def _to_device(cpu_batch: dict) -> dict:
        on_dev: dict = {k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in cpu_batch.items()}
        if isinstance(on_dev.get("pixel_values"), torch.Tensor):
            on_dev["pixel_values"] = on_dev["pixel_values"].to(dtype=dtype)
        return on_dev

    def _save_rng():
        """Snapshot CPU + CUDA RNG state for GradCache replay (so dropout
        in Step 3's re-forward matches Step 1's no-grad forward)."""
        return (torch.get_rng_state(), torch.cuda.get_rng_state())

    def _restore_rng(state) -> None:
        torch.set_rng_state(state[0])
        torch.cuda.set_rng_state(state[1])

    def _compute_val_loss() -> float:
        """Single InfoNCE on the full val set (one big macro-batch).

        Matches the GradCache training objective: InfoNCE is a global
        operation over the batch, so summing per-mini-batch losses
        underestimates the actual classification difficulty (each
        mini-batch only sees 4 negatives). Aggregating embeddings first
        gives a number that's directly comparable across epochs.
        """
        model.train(False); proj_head.train(False)
        embeds: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        with torch.no_grad():
            for vbatch in val_loader:
                lbl = vbatch.pop("labels_cl").to(device)
                v = _to_device(vbatch)
                emb = _forward_embeddings(v)
                embeds.append(emb)
                labels_all.append(lbl)
        model.train(); proj_head.train()
        if not embeds:
            return float("nan")
        all_e = torch.cat(embeds, dim=0)
        all_l = torch.cat(labels_all, dim=0)
        return float(contrastive_loss(all_e, labels=all_l).detach())

    epoch_losses: list[float] = []
    epoch_val_losses: list[float] = []
    epoch_grad_norms: list[float] = []
    epoch_intra_sim: list[float] = []
    epoch_inter_sim: list[float] = []
    epoch_zero_pos_batches: list[int] = []
    n_zero_pos_total = 0
    best_val = float("inf")
    epochs_since_improvement = 0
    early_stopped_at = None

    def _diag_intra_inter(embeds: torch.Tensor, labels: torch.Tensor):
        """Cosine-similarity diagnostics on a (macro-)batch — returns
        (intra_mean, inter_mean) over off-diagonal entries."""
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

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); proj_head.train()
        running = 0.0; n_steps = 0
        zero_pos_this = 0
        intra_sum = 0.0; inter_sum = 0.0; sim_n = 0
        grad_norm_sum = 0.0; grad_norm_n = 0
        optimizer.zero_grad(set_to_none=True)

        if USE_GRADCACHE:
            # GradCache training loop. macro_buffer caches one
            # gc_batch_size-worth of (cached_embed, label, RNG state,
            # CPU inputs) tuples; once full we run Step 2/3/4.
            macro_buffer: list[dict] = []
            n_minis = len(train_loader)
            for batch_idx, batch in enumerate(train_loader):
                lbl = batch.pop("labels_cl")
                cpu_inputs = {k: v for k, v in batch.items()}

                # Step 1: no-grad forward, cache projected embedding as
                # a leaf with grad. RNG snapshot lets Step 3 replay
                # dropout exactly.
                rng_state = _save_rng()
                on_dev = _to_device(cpu_inputs)
                with torch.no_grad():
                    emb_detached = _forward_embeddings(on_dev)
                cached = emb_detached.clone().requires_grad_(True)
                del on_dev, emb_detached

                macro_buffer.append({
                    "cached": cached,
                    "label": lbl.to(device),
                    "rng": rng_state,
                    "cpu_inputs": cpu_inputs,
                })

                is_last = (batch_idx + 1) == n_minis
                ready = (len(macro_buffer) >= gc_accumulation) or is_last
                if not ready:
                    continue

                # Step 2: compute InfoNCE on the full macro-batch and
                # back-propagate onto the cached embeddings.
                all_embeds = torch.cat(
                    [m["cached"] for m in macro_buffer], dim=0)
                all_labels = torch.cat(
                    [m["label"] for m in macro_buffer], dim=0)
                loss = contrastive_loss(all_embeds, labels=all_labels)
                l_val = float(loss.detach())
                running += l_val; n_steps += 1
                if l_val == 0.0:
                    zero_pos_this += 1

                intra, inter = _diag_intra_inter(all_embeds, all_labels)
                intra_sum += intra; inter_sum += inter; sim_n += 1

                if loss.grad_fn is not None:
                    loss.backward()

                    # Step 3: re-forward each mini-batch WITH grad
                    # (replaying RNG so dropout matches), then chain-rule
                    # backward via surrogate = (projected · cached.grad).sum().
                    for m in macro_buffer:
                        cg = m["cached"].grad
                        if cg is None:
                            continue
                        _restore_rng(m["rng"])
                        on_dev = _to_device(m["cpu_inputs"])
                        emb = _forward_embeddings(on_dev)
                        surrogate = (emb * cg).sum()
                        surrogate.backward()
                        del on_dev, emb, surrogate

                    # Step 4: clip + optimizer step.
                    gn = torch.nn.utils.clip_grad_norm_(
                        lora_params + proj_params, max_norm=1.0)
                    grad_norm_sum += float(gn); grad_norm_n += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    log.warning(
                        f"  [CL-Embed ep {epoch+1:02d}] degenerate "
                        f"macro-batch (no positives in any class) — "
                        f"skipping optimizer step")

                # Drop references so the autograd graph + cached tensors
                # are freed before the next macro-batch.
                macro_buffer.clear()
        else:
            for batch_idx, batch in enumerate(train_loader):
                lbl = batch.pop("labels_cl").to(device)
                on_dev: dict = {
                    k: (v.to(device) if hasattr(v, "to") else v)
                    for k, v in batch.items()
                }
                if isinstance(on_dev.get("pixel_values"), torch.Tensor):
                    on_dev["pixel_values"] = on_dev["pixel_values"].to(dtype=dtype)

                emb = _forward_embeddings(on_dev)
                loss = contrastive_loss(emb, labels=lbl)
                l_val = float(loss.detach())
                if l_val == 0.0:
                    zero_pos_this += 1

                intra, inter = _diag_intra_inter(emb, lbl)
                intra_sum += intra; inter_sum += inter; sim_n += 1

                (loss / GRAD_ACCUM).backward()
                running += l_val; n_steps += 1

                is_boundary = (batch_idx + 1) % GRAD_ACCUM == 0
                is_last = (batch_idx + 1) == len(train_loader)
                if is_boundary or is_last:
                    gn = torch.nn.utils.clip_grad_norm_(
                        lora_params + proj_params, max_norm=1.0)
                    grad_norm_sum += float(gn); grad_norm_n += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        avg_loss = running / max(1, n_steps)
        avg_grad = grad_norm_sum / max(1, grad_norm_n)
        intra = intra_sum / max(1, sim_n)
        inter = inter_sum / max(1, sim_n)
        n_zero_pos_total += zero_pos_this
        epoch_losses.append(avg_loss)
        epoch_grad_norms.append(avg_grad)
        epoch_intra_sim.append(intra)
        epoch_inter_sim.append(inter)
        epoch_zero_pos_batches.append(zero_pos_this)

        val_loss = None
        if use_early_stopping:
            val_loss = _compute_val_loss()
            epoch_val_losses.append(val_loss)
            improved = val_loss < best_val - 1e-4
            if improved:
                best_val = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            log.info(
                f"  [CL-Embed ep {epoch+1:02d}/{EPOCHS}] loss={avg_loss:.4f} "
                f"val={val_loss:.4f} best={best_val:.4f} "
                f"no_improve={epochs_since_improvement} "
                f"grad_norm={avg_grad:.4f} intra={intra:.4f} inter={inter:.4f} "
                f"gap={(intra-inter):.4f} zero_pos={zero_pos_this}"
            )
        else:
            log.info(
                f"  [CL-Embed ep {epoch+1:02d}/{EPOCHS}] loss={avg_loss:.4f} "
                f"grad_norm={avg_grad:.4f} intra={intra:.4f} inter={inter:.4f} "
                f"gap={(intra-inter):.4f} zero_pos={zero_pos_this}"
            )

        blob["train_progress"] = {
            "epochs_done": epoch + 1,
            "epoch_losses": epoch_losses,
            "epoch_val_losses": epoch_val_losses,
            "epoch_grad_norms": epoch_grad_norms,
            "epoch_intra_class_sim": epoch_intra_sim,
            "epoch_inter_class_sim": epoch_inter_sim,
            "epoch_zero_pos_batches": epoch_zero_pos_batches,
            "best_val_loss": best_val if use_early_stopping else None,
        }
        save_results_atomically(blob, metrics_path)

        if use_early_stopping and epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped_at = epoch + 1
            log.info(f"  [CL-Embed] early stopping at epoch {early_stopped_at}")
            break

    train_time = time.time() - t0
    blob["train"] = {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "gc_batch_size": GC_BATCH_SIZE,
        "use_gradcache": USE_GRADCACHE,
        "gc_accumulation": gc_accumulation,
        "effective_batch": effective_batch,
        "lr": LR, "temperature": TEMPERATURE,
        "epoch_losses": epoch_losses,
        "epoch_val_losses": epoch_val_losses,
        "epoch_grad_norms": epoch_grad_norms,
        "epoch_intra_class_sim": epoch_intra_sim,
        "epoch_inter_class_sim": epoch_inter_sim,
        "epoch_zero_pos_batches": epoch_zero_pos_batches,
        "n_zero_pos_total": n_zero_pos_total,
        "best_val_loss": best_val if use_early_stopping else None,
        "early_stopped_at_epoch": early_stopped_at,
        "train_time_sec": round(train_time, 2),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    save_results_atomically(blob, metrics_path)

    try:
        model.save_pretrained(str(ckpt_dir / "lora_adapter"))
    except Exception as e:
        log.warning(f"adapter save failed: {e!r}")
    torch.save(
        {"proj_head_state_dict": proj_head.state_dict(),
         "hidden_dim": hidden_dim,
         "config": blob["train"]},
        ckpt_dir / "proj_head.pt",
    )

    # Inference: nearest-centroid in projected space ----------------------
    log.info("Computing train-set [EOS] centroids in 128-d projected space ...")
    model.train(False); proj_head.train(False)

    @torch.no_grad()
    def _embed_one(sample: dict) -> torch.Tensor:
        text = apply_chat_template_safe(
            processor, _build_user_messages(sample), add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[sample["images"]],
            return_tensors="pt", padding=True,
        ).to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        emb = _forward_embeddings(inputs)
        return emb.detach().cpu()[0]

    k_present = len(present_classes)
    centroid_sum = torch.zeros(k_present, 128)
    centroid_counts: dict[int, int] = defaultdict(int)
    for i, idx in enumerate(train_only_idx):
        s = raw_train[idx]
        emb = _embed_one(s)
        lab = _remap_label_for_task(int(s["failure_label"]), task)
        pos = cid_to_pos.get(lab)
        if pos is None:
            continue
        centroid_sum[pos] += emb
        centroid_counts[pos] += 1
        if (i + 1) % 200 == 0:
            log.info(f"  centroid build {i+1}/{len(train_only_idx)}")
    centroids = torch.zeros_like(centroid_sum)
    for pos in range(k_present):
        if centroid_counts[pos] > 0:
            centroids[pos] = centroid_sum[pos] / centroid_counts[pos]
    centroids = F.normalize(centroids, dim=-1)
    log.info(f"  centroid counts per present class: "
             f"{ {present_label_names[pos]: centroid_counts.get(pos, 0) for pos in range(k_present)} }")

    test_indices = list(range(len(raw_eval)))
    if task == "7class":
        test_indices = [i for i in test_indices
                        if int(raw_eval[i]["failure_label"]) >= 1]
    # Drop test samples whose true class is absent from the train present
    # set (no centroid to compare against).
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

    label_names = list(present_label_names)
    predictions: list[str] = []
    ground_truths: list[str] = []
    t_test = time.time()
    for i, idx in enumerate(test_indices):
        s = raw_eval[idx]
        emb = _embed_one(s)
        sims = (emb.unsqueeze(0) @ centroids.T).squeeze(0)
        pred_pos = int(sims.argmax().item())
        gt_cid = _remap_label_for_task(int(s["failure_label"]), task)
        predictions.append(label_names[pred_pos])
        ground_truths.append(label_names[cid_to_pos[gt_cid]])
        if (i + 1) % 200 == 0:
            log.info(f"  test eval {i+1}/{len(test_indices)}")
    eval_time = time.time() - t_test

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"CL-Embed centroid eval: acc={m['accuracy']:.4f} "
             f"f1_macro={m['f1_macro']:.4f}")
    if m["majority_class_warning"]:
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
    write_done_flag(run_dir, {
        "method": "cl_embed",
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
    })
    return blob


# ---------------------------------------------------------------------------
# Eval-only mode — load LoRA adapter + proj_head, rebuild centroids from
# the (RLBench) train split using the loaded model, then evaluate on the
# new --dataset-eval. The centroids must be rebuilt at eval-time because
# they depend on the trained projection head's geometry; the centroids
# saved alongside DINOv2 head.pt files are for DINOv2 methods only.
# ---------------------------------------------------------------------------

def eval_cl_embed(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    ckpt_dir = Path(config["from_checkpoint"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(
        run_dir, name=f"cl_embed.eval.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    log.info(f"=== eval_cl_embed | from_checkpoint={ckpt_dir} task={task} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "cl_embed",
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
    from unsloth import FastModel
    from data.dataset import GuardianDataset
    from evaluation.metrics import compute_classification_metrics

    log.info("Loading base Qwen3.5-4B (4-bit) + LoRA adapter + proj_head ...")
    t0 = time.time()
    model, processor = FastModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",
        max_seq_length=2048, load_in_4bit=True,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    adapter_path = ckpt_dir / "checkpoint" / "lora_adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"CL-Embed adapter not found at {adapter_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.train(False)

    # Load proj_head
    proj_head_pt = ckpt_dir / "checkpoint" / "proj_head.pt"
    if not proj_head_pt.exists():
        raise FileNotFoundError(f"CL-Embed proj_head not found at {proj_head_pt}")
    ph_payload = torch.load(proj_head_pt, map_location="cpu", weights_only=False)
    hidden_dim = ph_payload.get("hidden_dim", _discover_hidden(model))
    proj_head = _build_proj_head(hidden_dim).to(device="cuda", dtype=torch.float32)
    proj_head.load_state_dict(ph_payload["proj_head_state_dict"])
    proj_head.train(False)
    blob["load_time_sec"] = round(time.time() - t0, 2)

    device = "cuda"
    dtype = torch.bfloat16

    # Load RLBench train (for centroids) and eval (for test)
    raw_train = GuardianDataset(config["dataset_train"])
    raw_eval = GuardianDataset(config["dataset_eval"], corruption_name=config.get("corruption"), severity=config.get("severity"))

    # Replicate the train-subset selection logic so centroids match what
    # the original training run saw. Use the SAME seed + val_fraction.
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
    train_only_idx = [train_idx[int(perm[i])] for i in range(n_val, len(train_idx))]
    train_only_idx = _filter_idx_for_task(raw_train, train_only_idx, task)

    # Determine present classes from train
    if task == "binary":
        present_classes = [0, 1]
        present_label_names = ["success", "failure"]
    else:
        present_class_set = sorted({
            _remap_label_for_task(int(raw_train[i]["failure_label"]), task)
            for i in train_only_idx
        })
        present_classes = present_class_set
        present_label_names = ([UNIFIED_LABEL_NAMES[c] for c in present_classes]
                               if task == "8class"
                               else _task_label_names(task))
    cid_to_pos = {c: i for i, c in enumerate(present_classes)}
    k_present = len(present_classes)
    log.info(f"present classes={present_classes} ({k_present})")

    @torch.no_grad()
    def _embed_one(sample: dict) -> torch.Tensor:
        text = apply_chat_template_safe(
            processor, _build_user_messages(sample), add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[sample["images"]],
            return_tensors="pt", padding=True,
        ).to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
        with torch.autocast("cuda", dtype=dtype):
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[-1]
        attn = inputs["attention_mask"]
        last_idx = attn.cumsum(dim=1).argmax(dim=1)
        eos = hs[torch.arange(hs.size(0), device=device), last_idx]
        emb = F.normalize(proj_head(eos.float()), dim=-1)
        return emb.detach().cpu()[0]

    # Build centroids from RLBench train
    log.info(f"Building centroids over {len(train_only_idx)} RLBench train "
             f"samples ...")
    t_cent = time.time()
    centroid_sum = torch.zeros(k_present, 128)
    centroid_counts: dict[int, int] = defaultdict(int)
    for i, idx in enumerate(train_only_idx):
        s = raw_train[idx]
        emb = _embed_one(s)
        lab = _remap_label_for_task(int(s["failure_label"]), task)
        pos = cid_to_pos.get(lab)
        if pos is None:
            continue
        centroid_sum[pos] += emb
        centroid_counts[pos] += 1
        if (i + 1) % 200 == 0:
            log.info(f"  centroid build {i+1}/{len(train_only_idx)}")
    centroids = torch.zeros_like(centroid_sum)
    for pos in range(k_present):
        if centroid_counts[pos] > 0:
            centroids[pos] = centroid_sum[pos] / centroid_counts[pos]
    centroids = F.normalize(centroids, dim=-1)
    log.info(f"  centroid build time: {time.time() - t_cent:.1f}s  counts="
             f"{ {present_label_names[pos]: centroid_counts.get(pos, 0) for pos in range(k_present)} }")

    # Evaluate
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

    label_names = list(present_label_names)
    predictions: list[str] = []
    ground_truths: list[str] = []
    t_test = time.time()
    for i, idx in enumerate(test_indices):
        s = raw_eval[idx]
        emb = _embed_one(s)
        sims = (emb.unsqueeze(0) @ centroids.T).squeeze(0)
        pred_pos = int(sims.argmax().item())
        gt_cid = _remap_label_for_task(int(s["failure_label"]), task)
        predictions.append(label_names[pred_pos])
        ground_truths.append(label_names[cid_to_pos[gt_cid]])
        if (i + 1) % 200 == 0:
            log.info(f"  test eval {i+1}/{len(test_indices)}")
    eval_time = time.time() - t_test

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"CL-Embed eval-only: acc={m['accuracy']:.4f} "
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
    write_done_flag(run_dir, {
        "method": "cl_embed",
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
        "eval_only": True,
    })
    return blob
    return blob
