"""CL-LLM: contrastive learning over the model's own answer-token logits.

Method (extends Dai et al. 2025 — arxiv:2510.14824):

    binary  : 6 viewpoint images + a yes/no probe ("Did the robot
              successfully complete this task? Answer only yes or no.").
              score = yes_logit - no_logit at the last attended position.
              loss  = asymmetric InfoNCE on the scalar score.
              eval  = score > 0 ⇒ "success", else "failure".

    multiclass (7class / 8class): 6 viewpoint images + a letter-prompt
              ("A=no_grasp, B=slip, ... Answer with one letter only.").
              The set of letters is sized to len(present_classes) so the
              prompt only ever asks about classes that have ≥1 train
              sample (audit P2.1 — wrong_state is dropped on RLBench).
              score = [logit(A), logit(B), ...] at the last attended
              position — a [B, K] tensor, K = num present classes.
              loss  = supervised InfoNCE on the L2-normalized score
              vector (same-class samples are positives).
              eval  = argmax(score) → letter → class name.

The encoder is the full Qwen3.5-4B with QLoRA on ALL layers in every
case. No GradCache here: K is small (<=8) so the contrastive batch
already fits at batch_size=4.
"""

from __future__ import annotations

import math
import os
import sys
import time
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


_BINARY_PROMPT = (
    "Task: {task}\n"
    "Subtask: {subtask}\n"
    "Did the robot successfully complete this task? "
    "Answer only yes or no."
)

_MULTI_PROMPT = (
    "Task: {task}\n"
    "Subtask: {subtask}\n"
    "Classify the robot manipulation outcome.\n"
    "{letter_lines}\n"
    "Answer with one letter only."
)

# Letters used to encode classes in multiclass mode. We slice to
# len(present_classes) so RLBench (where wrong_state is absent) uses
# A..F for 7class and A..G for 8class.
_LETTER_POOL = list("ABCDEFGHIJ")


def _build_user_messages(sample: dict, prompt_text: str) -> list[dict]:
    user_content: list[dict] = [
        {"type": "image", "image": img} for img in sample["images"]
    ]
    user_content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": user_content}]


def _resolve_yes_no_ids(tokenizer) -> tuple[int, int]:
    candidates_yes = ["yes", " yes", "Yes", " Yes"]
    candidates_no = ["no", " no", "No", " No"]
    yes_id = no_id = None
    for s in candidates_yes:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            yes_id = ids[0]; break
    for s in candidates_no:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            no_id = ids[0]; break
    if yes_id is None or no_id is None:
        yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("no", add_special_tokens=False)[0]
    return int(yes_id), int(no_id)


def _resolve_letter_ids(tokenizer, letters: list[str]) -> list[int]:
    """Pick the single-token id for each letter. Tries a few common
    spellings (with/without leading space) and falls back to the first
    sub-token if the letter never resolves to a single token.
    """
    out: list[int] = []
    for letter in letters:
        candidates = [letter, " " + letter]
        chosen: int | None = None
        for s in candidates:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                chosen = int(ids[0]); break
        if chosen is None:
            chosen = int(tokenizer.encode(letter, add_special_tokens=False)[0])
        out.append(chosen)
    if len(set(out)) != len(out):
        raise RuntimeError(
            f"Letter token ids collide ({list(zip(letters, out))}) — "
            f"choose a different LETTER_POOL or a different model."
        )
    return out


def _binary_label(failure_label: int) -> int:
    """0 = success, 1 = failure (any of the 7 failure types)."""
    return 0 if int(failure_label) == 0 else 1


def _remap_label_for_task(failure_label: int, task: str) -> int:
    if task == "8class":
        return int(failure_label)
    if task == "binary":
        return _binary_label(failure_label)
    if task == "7class":
        return int(failure_label) - 1
    raise ValueError(f"Unknown task: {task!r}")


def _filter_idx_for_task(raw, indices, task: str):
    if task != "7class":
        return list(indices)
    return [i for i in indices if int(raw[i]["failure_label"]) >= 1]


def _build_multiclass_prompt(
    sample: dict, *, letters: list[str], class_names: list[str],
) -> str:
    """Build the per-task prompt for multiclass CL-LLM.

    Format (matches the user spec):
        "A=no_grasp, B=slip, ..."
    For 7class we additionally prefix the disambiguation
    'This robot task FAILED. Classify the failure type.' on the line
    before the letter assignments — built into the caller.
    """
    body = ", ".join(f"{l}={n}" for l, n in zip(letters, class_names))
    return _MULTI_PROMPT.format(
        task=sample["task_instruction"],
        subtask=sample["detailed_subtask_name"],
        letter_lines=body,
    )


def _build_7class_prompt(
    sample: dict, *, letters: list[str], class_names: list[str],
) -> str:
    body = ", ".join(f"{l}={n}" for l, n in zip(letters, class_names))
    return (
        f"Task: {sample['task_instruction']}\n"
        f"Subtask: {sample['detailed_subtask_name']}\n"
        f"This robot task FAILED. Classify the failure type.\n"
        f"{body}\n"
        f"Answer with one letter only."
    )


def _infonce_token_loss(scores, labels, temperature: float = 1.0):
    """Asymmetric binary InfoNCE on the scalar (yes-no) score.

    Used for the binary task only. Returns a 0-tensor with grad if either
    class is missing in the batch (the batch_size>=4 guard avoids this in
    practice).
    """
    import torch
    scores = scores / temperature
    success_mask = labels == 0
    failure_mask = labels == 1
    if success_mask.sum() == 0 or failure_mask.sum() == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype,
                           requires_grad=True)
    loss = torch.zeros((), device=scores.device, dtype=scores.dtype)
    n = 0
    for i in success_mask.nonzero(as_tuple=True)[0]:
        pos = scores[i]
        neg = scores[failure_mask]
        all_scores = torch.cat([pos.unsqueeze(0), neg])
        loss = loss - (pos - torch.logsumexp(all_scores, dim=0))
        n += 1
    return loss / max(n, 1)


def _infonce_multiclass(scores, labels, *, temperature: float, infonce_loss):
    """Supervised InfoNCE on the L2-normalized [B, K] letter-logit vector.

    Each sample's K-dim score vector acts as its embedding in the K-d
    space; same-class samples are positives. This is the multiclass
    extension of CL-LLM described in the user spec.
    """
    import torch
    import torch.nn.functional as F
    scaled = scores / temperature
    if not torch.isfinite(scaled).all():
        return torch.zeros((), device=scores.device, dtype=scores.dtype,
                           requires_grad=True)
    embeds = F.normalize(scaled.float(), dim=-1)
    return infonce_loss(embeds, labels=labels)


def train_cl_llm(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "binary"))
    if task not in ("binary", "7class", "8class"):
        raise ValueError(f"Unknown task: {task!r}")

    BATCH_SIZE = int(config.get("batch_size", 4))
    if BATCH_SIZE < 4 and not bool(config.get("allow_degenerate", False)):
        raise AssertionError(
            f"CL-LLM requires batch_size >= 4 to mix classes in the same "
            f"batch (the InfoNCE term needs both). Got {BATCH_SIZE}."
        )

    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(run_dir, name=f"cl_llm.{config.get('exp_id')}.{seed}")

    metrics_path = run_dir / "metrics.json"
    ckpt_dir = run_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== train_cl_llm | exp_id={config.get('exp_id')} task={task} "
             f"seed={seed} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "cl_llm",
        "task": task,
        "seed": seed,
        "config": dict(config),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at": time.time(),
        "environment": capture_environment(),
    }
    save_results_atomically(blob, metrics_path)

    import torch
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
    log.info(f"train: {len(raw_train)} -> {len(train_only_idx)} train + "
             f"{len(val_idx_split)} val (after task='{task}' filter)")

    # Per-task class plumbing.
    if task == "binary":
        present_classes = [0, 1]
        present_label_names = ["success", "failure"]
        cid_to_pos = {0: 0, 1: 1}
    else:
        present_classes, present_label_names, _ = (
            compute_present_classes(raw_train, train_only_idx, task))
        cid_to_pos = {c: i for i, c in enumerate(present_classes)}
    k_present = len(present_classes)
    log.info(f"task '{task}': present classes = {present_classes} "
             f"({k_present}) names={present_label_names}")

    # Resolve answer-token ids and prompt builder per task.
    if task == "binary":
        yes_id, no_id = _resolve_yes_no_ids(tokenizer)
        log.info(f"binary mode: yes_token_id={yes_id} no_token_id={no_id}")
        blob["yes_token_id"] = int(yes_id)
        blob["no_token_id"] = int(no_id)
        letters: list[str] = []
        letter_token_ids: list[int] = []
    else:
        if k_present > len(_LETTER_POOL):
            raise RuntimeError(
                f"too many present classes ({k_present}) for the letter "
                f"pool {_LETTER_POOL}; extend _LETTER_POOL.")
        letters = _LETTER_POOL[:k_present]
        letter_token_ids = _resolve_letter_ids(tokenizer, letters)
        log.info(f"multiclass mode: letters={letters} "
                 f"token_ids={letter_token_ids}")
        blob["letters"] = list(letters)
        blob["letter_token_ids"] = list(letter_token_ids)
        blob["letter_to_class"] = {
            l: present_label_names[i] for i, l in enumerate(letters)
        }

    def _prompt_for(sample: dict) -> str:
        if task == "binary":
            return _BINARY_PROMPT.format(
                task=sample["task_instruction"],
                subtask=sample["detailed_subtask_name"],
            )
        if task == "7class":
            return _build_7class_prompt(
                sample, letters=letters, class_names=present_label_names)
        return _build_multiclass_prompt(
            sample, letters=letters, class_names=present_label_names)

    class CLLLMDS(TorchDataset):
        def __init__(self, ds, idx_list):
            self.ds = ds
            self.idx = list(idx_list)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            sample = self.ds[self.idx[i]]
            cid = _remap_label_for_task(int(sample["failure_label"]), task)
            return {
                "messages": _build_user_messages(sample, _prompt_for(sample)),
                "images": sample["images"],
                "label_pos": cid_to_pos[cid],  # position in present set
            }

    train_ds = CLLLMDS(raw_train, train_only_idx)
    val_ds = CLLLMDS(raw_train, val_idx_split) if val_idx_split else None

    def collate(batch):
        texts = [
            apply_chat_template_safe(processor, b["messages"],
                                     add_generation_prompt=True)
            for b in batch
        ]
        images = [b["images"] for b in batch]
        labels = torch.tensor([b["label_pos"] for b in batch], dtype=torch.long)
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

    # GradCache — when gc_batch_size > batch_size, decouple the contrastive
    # batch from the per-step VRAM by accumulating cached scores from
    # gc_batch_size/batch_size mini-batches and running InfoNCE on the
    # macro-batch (cf. CL-Embed). Set gc_batch_size=0 to fall back to
    # plain gradient accumulation.
    GC_BATCH_SIZE = int(config.get("gc_batch_size", 256))
    USE_GRADCACHE = GC_BATCH_SIZE > BATCH_SIZE
    if USE_GRADCACHE and GC_BATCH_SIZE % BATCH_SIZE != 0:
        raise ValueError(
            f"gc_batch_size ({GC_BATCH_SIZE}) must be a multiple of "
            f"batch_size ({BATCH_SIZE})")
    gc_accumulation = (GC_BATCH_SIZE // BATCH_SIZE) if USE_GRADCACHE else 0

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

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)

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
    multiclass_infonce = InfoNCELoss(temperature=1.0) if task != "binary" else None

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
        log.info(f"Training (legacy GA): epochs={EPOCHS} bs={BATCH_SIZE} "
                 f"ga={GRAD_ACCUM} effective_batch={effective_batch} lr={LR} "
                 f"τ={TEMPERATURE} early_stopping={use_early_stopping} "
                 f"patience={EARLY_STOPPING_PATIENCE}")

    def _last_token_logits(batch_on_dev) -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype):
            out = model(**batch_on_dev)
        logits = out.logits  # [B, T, V]
        attn = batch_on_dev["attention_mask"]
        last_idx = attn.cumsum(dim=1).argmax(dim=1)
        return logits[torch.arange(logits.size(0), device=device), last_idx]

    def _scores_for_batch(batch_on_dev) -> torch.Tensor:
        last_logits = _last_token_logits(batch_on_dev)
        if task == "binary":
            return (last_logits[:, yes_id] - last_logits[:, no_id]).float()
        cols = torch.tensor(letter_token_ids, device=device, dtype=torch.long)
        return last_logits.index_select(1, cols).float()

    def _to_device(cpu_batch: dict) -> dict:
        on_dev: dict = {k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in cpu_batch.items()}
        if isinstance(on_dev.get("pixel_values"), torch.Tensor):
            on_dev["pixel_values"] = on_dev["pixel_values"].to(dtype=dtype)
        return on_dev

    def _save_rng():
        return (torch.get_rng_state(), torch.cuda.get_rng_state())

    def _restore_rng(state) -> None:
        torch.set_rng_state(state[0])
        torch.cuda.set_rng_state(state[1])

    def _step_loss(scores, labels):
        if task == "binary":
            return _infonce_token_loss(scores, labels, temperature=TEMPERATURE)
        return _infonce_multiclass(
            scores, labels, temperature=TEMPERATURE,
            infonce_loss=multiclass_infonce)

    def _compute_val_loss() -> float:
        # Match the training objective: under GradCache, the loss is
        # computed on the full macro-batch, so the val loss aggregates
        # all val scores into a single tensor and runs InfoNCE once.
        model.train(False)
        if USE_GRADCACHE:
            scores_all: list = []; labels_all: list = []
            with torch.no_grad():
                for vbatch in val_loader:
                    lbl = vbatch.pop("labels_cl").to(device)
                    v = _to_device(vbatch)
                    scores = _scores_for_batch(v)
                    scores_all.append(scores); labels_all.append(lbl)
            model.train()
            if not scores_all:
                return float("nan")
            return float(_step_loss(
                torch.cat(scores_all, dim=0),
                torch.cat(labels_all, dim=0)).detach())
        total = 0.0; n = 0
        with torch.no_grad():
            for vbatch in val_loader:
                lbl = vbatch.pop("labels_cl").to(device)
                v = _to_device(vbatch)
                scores = _scores_for_batch(v)
                loss = _step_loss(scores, lbl)
                total += float(loss.detach()); n += 1
        model.train()
        return total / max(1, n)

    epoch_losses: list[float] = []
    epoch_val_losses: list[float] = []
    epoch_grad_norms: list[float] = []
    epoch_zero_pos_batches: list[int] = []
    n_zero_pos_total = 0
    best_val = float("inf")
    epochs_since_improvement = 0
    early_stopped_at = None

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0; n_steps = 0
        zero_pos_this = 0
        grad_norm_sum = 0.0; grad_norm_n = 0
        optimizer.zero_grad(set_to_none=True)

        if USE_GRADCACHE:
            macro_buffer: list[dict] = []
            n_minis = len(train_loader)
            for batch_idx, batch in enumerate(train_loader):
                lbl = batch.pop("labels_cl")
                cpu_inputs = {k: v for k, v in batch.items()}

                # Step 1: no-grad forward, cache scores. RNG snapshot
                # lets Step 3 replay dropout exactly.
                rng_state = _save_rng()
                on_dev = _to_device(cpu_inputs)
                with torch.no_grad():
                    scores_detached = _scores_for_batch(on_dev)
                cached = scores_detached.clone().requires_grad_(True)
                del on_dev, scores_detached

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

                # Step 2: macro-batch InfoNCE → backward onto cached.grad.
                all_scores = torch.cat(
                    [m["cached"] for m in macro_buffer], dim=0)
                all_labels = torch.cat(
                    [m["label"] for m in macro_buffer], dim=0)
                loss = _step_loss(all_scores, all_labels)
                l_val = float(loss.detach())
                running += l_val; n_steps += 1
                if l_val == 0.0:
                    zero_pos_this += 1

                if loss.grad_fn is not None:
                    loss.backward()

                    # Step 3: re-forward each mini-batch WITH grad
                    # (replay RNG so dropout matches Step 1) and chain
                    # the gradient via surrogate = (scores · cached.grad).sum().
                    for m in macro_buffer:
                        cg = m["cached"].grad
                        if cg is None:
                            continue
                        _restore_rng(m["rng"])
                        on_dev = _to_device(m["cpu_inputs"])
                        scores_re = _scores_for_batch(on_dev)
                        surrogate = (scores_re * cg).sum()
                        surrogate.backward()
                        del on_dev, scores_re, surrogate

                    # Step 4: clip + optimizer step.
                    gn = torch.nn.utils.clip_grad_norm_(
                        lora_params, max_norm=1.0)
                    grad_norm_sum += float(gn); grad_norm_n += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    log.warning(
                        f"  [CL-LLM ep {epoch+1:02d}] degenerate macro-batch "
                        f"(no positives / no class mix) — skipping optim step")

                macro_buffer.clear()
        else:
            for batch_idx, batch in enumerate(train_loader):
                lbl = batch.pop("labels_cl").to(device)
                on_dev = _to_device(batch)

                scores = _scores_for_batch(on_dev)
                loss = _step_loss(scores, lbl)
                l_val = float(loss.detach())
                if l_val == 0.0:
                    zero_pos_this += 1
                (loss / GRAD_ACCUM).backward()
                running += l_val; n_steps += 1

                is_boundary = (batch_idx + 1) % GRAD_ACCUM == 0
                is_last = (batch_idx + 1) == len(train_loader)
                if is_boundary or is_last:
                    gn = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                    grad_norm_sum += float(gn); grad_norm_n += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        avg_loss = running / max(1, n_steps)
        avg_grad = grad_norm_sum / max(1, grad_norm_n)
        n_zero_pos_total += zero_pos_this
        epoch_losses.append(avg_loss)
        epoch_grad_norms.append(avg_grad)
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
                f"  [CL-LLM ep {epoch+1:02d}/{EPOCHS}] loss={avg_loss:.4f} "
                f"val={val_loss:.4f} best={best_val:.4f} "
                f"no_improve={epochs_since_improvement} "
                f"grad_norm={avg_grad:.4f} zero_pos={zero_pos_this}"
            )
        else:
            log.info(
                f"  [CL-LLM ep {epoch+1:02d}/{EPOCHS}] loss={avg_loss:.4f} "
                f"grad_norm={avg_grad:.4f} zero_pos={zero_pos_this}"
            )

        blob["train_progress"] = {
            "epochs_done": epoch + 1,
            "epoch_losses": epoch_losses,
            "epoch_val_losses": epoch_val_losses,
            "epoch_grad_norms": epoch_grad_norms,
            "epoch_zero_pos_batches": epoch_zero_pos_batches,
            "best_val_loss": best_val if use_early_stopping else None,
        }
        save_results_atomically(blob, metrics_path)

        if use_early_stopping and epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped_at = epoch + 1
            log.info(f"  [CL-LLM] early stopping at epoch {early_stopped_at}")
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

    # Inference -----------------------------------------------------------
    log.info(f"Eval on {config['dataset_eval']} ({len(raw_eval)} samples) ...")
    model.train(False)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        try:
            model.generation_config.enable_thinking = False
        except Exception:
            pass

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
        log.info(f"  dropped {n_eval_dropped} test samples whose true class "
                 f"is absent from train")

    if task == "binary":
        label_names = ["success", "failure"]
    else:
        label_names = list(present_label_names)

    predictions: list[str] = []
    ground_truths: list[str] = []
    scores_all: list = []
    t_test = time.time()
    with torch.no_grad():
        for n_done, idx in enumerate(test_indices, start=1):
            sample = raw_eval[idx]
            text = apply_chat_template_safe(
                processor, _build_user_messages(sample, _prompt_for(sample)),
                add_generation_prompt=True)
            inputs = processor(
                text=[text], images=[sample["images"]],
                return_tensors="pt", padding=True,
            ).to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
            scores = _scores_for_batch(inputs)
            cid = _remap_label_for_task(int(sample["failure_label"]), task)
            gt_pos = cid_to_pos[cid]
            if task == "binary":
                score = float(scores[0].item())
                scores_all.append(score)
                pred_pos = 0 if score > 0 else 1
            else:
                row = scores[0]  # [K]
                scores_all.append([float(x) for x in row.detach().cpu().tolist()])
                pred_pos = int(row.argmax().item())
            predictions.append(label_names[pred_pos])
            ground_truths.append(label_names[gt_pos])
            if n_done % 200 == 0:
                log.info(f"  test eval {n_done}/{len(test_indices)}")
    eval_time = time.time() - t_test

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"CL-LLM eval ({task}): acc={m['accuracy']:.4f} "
             f"f1_macro={m['f1_macro']:.4f}")
    if m["majority_class_warning"]:
        log.warning(m["majority_class_warning"])

    blob.update({
        "metrics": m,
        "eval_time_sec": round(eval_time, 2),
        "predictions": predictions,
        "ground_truths": ground_truths,
        "scores_first_50": scores_all[:50],
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
        "method": "cl_llm",
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
    })
    return blob


# ---------------------------------------------------------------------------
# Eval-only mode — load LoRA adapter and run the yes/no probe (binary) or
# letter probe (multiclass) on the new --dataset-eval. No training.
# ---------------------------------------------------------------------------

def eval_cl_llm(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    ckpt_dir = Path(config["from_checkpoint"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(
        run_dir, name=f"cl_llm.eval.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    log.info(f"=== eval_cl_llm | from_checkpoint={ckpt_dir} task={task} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "cl_llm",
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
    from unsloth import FastModel
    from data.dataset import GuardianDataset
    from evaluation.metrics import compute_classification_metrics

    log.info("Loading base Qwen3.5-4B (4-bit) + LoRA adapter ...")
    t0 = time.time()
    model, processor = FastModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",
        max_seq_length=2048, load_in_4bit=True,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    adapter_path = ckpt_dir / "checkpoint" / "lora_adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"CL-LLM adapter not found at {adapter_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.train(False)
    blob["load_time_sec"] = round(time.time() - t0, 2)

    device = "cuda"
    dtype = torch.bfloat16

    raw_eval = GuardianDataset(config["dataset_eval"], corruption_name=config.get("corruption"), severity=config.get("severity"))
    test_indices = list(range(len(raw_eval)))
    if task == "7class":
        test_indices = [i for i in test_indices
                        if int(raw_eval[i]["failure_label"]) >= 1]

    # Set up the per-task probe.
    if task == "binary":
        present_classes = [0, 1]
        present_label_names = ["success", "failure"]
        cid_to_pos = {0: 0, 1: 1}
        yes_id, no_id = _resolve_yes_no_ids(tokenizer)
        log.info(f"binary mode: yes_token_id={yes_id} no_token_id={no_id}")
        letters: list[str] = []
        letter_token_ids: list[int] = []
    else:
        # For multiclass, we'd need the same present_classes used at
        # training time. We infer them from the eval split's present
        # labels (fallback) — but for the planned E5.2 sweep we only use
        # binary, so this branch shouldn't fire there.
        present_classes_set = sorted({
            _remap_label_for_task(int(raw_eval[i]["failure_label"]), task)
            for i in test_indices
        })
        present_classes = present_classes_set
        cid_to_pos = {c: i for i, c in enumerate(present_classes)}
        present_label_names = [UNIFIED_LABEL_NAMES[c] for c in present_classes]
        letters = _LETTER_POOL[:len(present_classes)]
        letter_token_ids = _resolve_letter_ids(tokenizer, letters)
        log.info(f"multiclass mode: letters={letters} ids={letter_token_ids}")

    n_eval_before = len(test_indices)
    test_indices = [
        i for i in test_indices
        if _remap_label_for_task(int(raw_eval[i]["failure_label"]), task)
        in cid_to_pos
    ]
    n_eval_dropped = n_eval_before - len(test_indices)
    log.info(f"eval samples: {len(test_indices)} "
             f"(dropped {n_eval_dropped} absent-class)")

    def _prompt_for(sample: dict) -> str:
        if task == "binary":
            return _BINARY_PROMPT.format(
                task=sample["task_instruction"],
                subtask=sample["detailed_subtask_name"],
            )
        if task == "7class":
            return _build_7class_prompt(
                sample, letters=letters, class_names=present_label_names)
        return _build_multiclass_prompt(
            sample, letters=letters, class_names=present_label_names)

    def _scores_for_one(inputs) -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype):
            out = model(**inputs)
        logits = out.logits  # [B, T, V]
        attn = inputs["attention_mask"]
        last_idx = attn.cumsum(dim=1).argmax(dim=1)
        last_logits = logits[
            torch.arange(logits.size(0), device=device), last_idx]
        if task == "binary":
            return (last_logits[:, yes_id] - last_logits[:, no_id]).float()
        cols = torch.tensor(letter_token_ids, device=device, dtype=torch.long)
        return last_logits.index_select(1, cols).float()

    if task == "binary":
        label_names = ["success", "failure"]
    else:
        label_names = list(present_label_names)

    predictions: list[str] = []
    ground_truths: list[str] = []
    scores_all: list = []
    t_test = time.time()
    with torch.no_grad():
        for n_done, idx in enumerate(test_indices, start=1):
            sample = raw_eval[idx]
            text = apply_chat_template_safe(
                processor, _build_user_messages(sample, _prompt_for(sample)),
                add_generation_prompt=True)
            inputs = processor(
                text=[text], images=[sample["images"]],
                return_tensors="pt", padding=True,
            ).to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
            scores = _scores_for_one(inputs)
            cid = _remap_label_for_task(int(sample["failure_label"]), task)
            gt_pos = cid_to_pos[cid]
            if task == "binary":
                score = float(scores[0].item())
                scores_all.append(score)
                pred_pos = 0 if score > 0 else 1
            else:
                row = scores[0]
                scores_all.append([float(x) for x in row.detach().cpu().tolist()])
                pred_pos = int(row.argmax().item())
            predictions.append(label_names[pred_pos])
            ground_truths.append(label_names[gt_pos])
            if n_done % 200 == 0:
                log.info(f"  test eval {n_done}/{len(test_indices)}")
    eval_time = time.time() - t_test

    m = compute_classification_metrics(ground_truths, predictions, label_names)
    log.info(f"CL-LLM eval-only ({task}): acc={m['accuracy']:.4f} "
             f"f1_macro={m['f1_macro']:.4f}")
    if m.get("majority_class_warning"):
        log.warning(m["majority_class_warning"])

    blob.update({
        "metrics": m,
        "eval_time_sec": round(eval_time, 2),
        "predictions": predictions,
        "ground_truths": ground_truths,
        "scores_first_50": scores_all[:50],
        "present_classes": present_classes,
        "present_label_names": present_label_names,
        "n_classes_present": len(present_classes),
        "n_eval_dropped_absent_class": n_eval_dropped,
        "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": True,
    })
    finalize_metrics_schema(blob)
    save_results_atomically(blob, metrics_path)
    write_done_flag(run_dir, {
        "method": "cl_llm",
        "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
        "eval_only": True,
    })
    return blob
