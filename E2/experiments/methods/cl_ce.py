"""CL+CE — joint cross-entropy + supervised-InfoNCE fine-tuning.

NOT the CAFe paper (Yu et al., 2025, arXiv:2503.19900). That paper uses
CLIP-style image-text pair matching for retrieval embeddings. This method
is closer to **SFT + a supervised contrastive regulariser**, drawing on:

- Supervised contrastive (Khosla et al., 2020) for the InfoNCE objective
  (same-class positives within a mini-batch, no pair structure required).
- VLM2Vec (Jiang et al., ICLR 2025) for the [last-attended-token] hidden
  state extraction pattern.
- The *idea* of joint generative + contrastive training on a single tower
  (CAFe; Yu et al., 2025) — adapted here for closed-set classification
  rather than open-vocabulary retrieval.

Design rules (post-mode-collapse rebuild on 2026-05-09):

  CL+CE = SFT + one extra loss term.

  1. Copy SFT's pipeline verbatim — same Unsloth + QLoRA model, same
     UnslothVisionDataCollator, same TRL SFTTrainer scaffolding.
  2. **InfoNCE labels match CE labels (always fine-grained).** Never
     collapse to binary. The earlier audit fix that collapsed binary
     InfoNCE to {success, failure} caused mode collapse — CE trained
     "generate translation/no_grasp/..." while InfoNCE forced all 7
     failure types into a single "failure" centroid. Those objectives
     fight each other.
  3. λ = 0.1 (default). CE must dominate; InfoNCE is a regulariser, not
     the primary objective.
  4. **No GradCache.** Per-mini-batch InfoNCE only. With `batch_size=4`
     the contrastive signal is weak — that's intended. We need it
     small relative to CE so it can't overwrite the LM head.

  Inputs (CE and InfoNCE share):
    User turn   : 6 (or 8 RLBench) viewpoint images + task/subtask prompt
    Assistant   : the fine-grained `failure_mode` string (success / no_grasp
                  / slip / translation / rotation / wrong_object /
                  wrong_sequence / wrong_state)

  Outputs (CE):
    Causal-LM logits over the assistant turn.

  Outputs (InfoNCE):
    Hidden state at the last attended position -> 2-layer projection
    head (hidden_dim -> 512 -> 128) -> L2-normalised embedding ->
    supervised InfoNCE on fine-grained class labels (one centroid per
    present class).

  Eval:
    1) Generative — model.generate() -> parse_failure_mode (identical
       to SFT). This is the headline metric.
    2) Embedding (optional, secondary) — disabled by default; can be
       re-added later if needed for sim-to-real or t-SNE work.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

# Must be set before importing unsloth / transformers.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.methods.common import (  # noqa: E402
    UNIFIED_LABEL_NAMES, apply_chat_template_safe, build_messages,
    capture_environment, finalize_metrics_schema, parse_failure_mode,
    present_label_names_for_task, remap_strings_for_task,
    save_results_atomically, select_indices, set_seed, setup_run_logger,
    write_done_flag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_idx_for_task(raw, indices, task: str):
    """7class task drops success at the dataloader. Binary and 8class
    keep every sample (matching SFT's behaviour)."""
    if task != "7class":
        return list(indices)
    return [i for i in indices if int(raw[i]["failure_label"]) >= 1]


def _build_proj_head(in_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.GELU(),
        nn.Linear(512, 128),
    )


def _discover_hidden(model) -> int:
    """Walk the (possibly PEFT-wrapped) model to find LM hidden size."""
    cur = model
    for _ in range(6):
        cfg = getattr(cur, "config", None)
        if cfg is not None:
            for attr in ("hidden_size",):
                if hasattr(cfg, attr) and isinstance(getattr(cfg, attr), int):
                    return int(getattr(cfg, attr))
            for sub in ("text_config", "language_config"):
                if hasattr(cfg, sub):
                    sub_cfg = getattr(cfg, sub)
                    if hasattr(sub_cfg, "hidden_size"):
                        return int(sub_cfg.hidden_size)
        cur = getattr(cur, "base_model", None) or getattr(cur, "model", None)
        if cur is None:
            break
    raise RuntimeError("Could not discover hidden_size")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_cl_ce(config: dict) -> dict:
    """CL+CE = SFT + λ · InfoNCE on [last-attended-token] hidden state.

    InfoNCE labels are fine-grained class IDs matching what CE generates.
    No binary collapse, no GradCache.
    """
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))

    BATCH_SIZE = int(config.get("batch_size", 4))
    if BATCH_SIZE < 4 and not bool(config.get("allow_degenerate", False)):
        raise AssertionError(
            f"CL+CE requires batch_size >= 4 for the InfoNCE term to find "
            f"positive same-class pairs. Got {BATCH_SIZE}. Pass "
            f"config['allow_degenerate']=True to override (CE-only mode)."
        )

    run_dir = Path(config["run_dir"])
    log = setup_run_logger(run_dir, name=f"cl_ce.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    ckpt_dir = run_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== train_cl_ce | exp_id={config.get('exp_id')} "
             f"seed={seed} task={task} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "cl_ce",
        "task": task,
        "seed": seed,
        "config": dict(config),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at": time.time(),
        "environment": capture_environment(),
    }
    save_results_atomically(blob, metrics_path)

    # ---- Imports (deferred so each method only pays its own import cost) ----
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset as TorchDataset
    from unsloth import FastModel
    from data.dataset import GuardianDataset
    from evaluation.metrics import compute_classification_metrics
    from losses.infonce import InfoNCELoss

    # ---- Model ----
    log.info("Loading Qwen3.5-4B (4-bit) + LoRA r=16 (all layers) ...")
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
    save_results_atomically(blob, metrics_path)

    hidden_dim = _discover_hidden(model)
    log.info(f"hidden_dim={hidden_dim}")
    proj_head = _build_proj_head(hidden_dim).to(device="cuda", dtype=torch.float32)

    # ---- Data ----
    raw_train = GuardianDataset(config["dataset_train"])
    raw_eval = GuardianDataset(config["dataset_eval"], corruption_name=config.get("corruption"), severity=config.get("severity"))

    train_idx = select_indices(
        len(raw_train),
        data_fraction=float(config.get("data_fraction", 1.0)),
        seed=seed,
    )

    # Held-out validation slice off train (NOT the eval set — that would
    # leak the gate metric).
    val_fraction = float(config.get("val_fraction", 0.05))
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    n_val = max(8, int(round(val_fraction * len(train_idx)))) if val_fraction > 0 else 0
    if n_val >= len(train_idx):
        n_val = max(0, len(train_idx) // 10)
    val_idx = [train_idx[int(perm[i])] for i in range(n_val)]
    train_only_idx = [train_idx[int(perm[i])] for i in range(n_val, len(train_idx))]

    # 7class task drops success at the dataloader; binary and 8class don't.
    train_only_idx = _filter_idx_for_task(raw_train, train_only_idx, task)
    val_idx = _filter_idx_for_task(raw_train, val_idx, task)

    # InfoNCE labels are FINE-GRAINED — collect the present fine-grained
    # class IDs from training. wrong_state has 0 RLBench train samples and
    # is automatically excluded; success is excluded only when task=7class.
    present_class_ids = sorted({
        int(raw_train[i]["failure_label"]) for i in train_only_idx
    })
    present_label_names = [UNIFIED_LABEL_NAMES[c] for c in present_class_ids]
    cid_to_pos = {c: i for i, c in enumerate(present_class_ids)}
    log.info(f"task '{task}' fine-grained InfoNCE classes: {present_class_ids} "
             f"({len(present_class_ids)} present) names={present_label_names}")

    log.info(f"train: {len(raw_train)} -> {len(train_only_idx)} train + "
             f"{len(val_idx)} val (after task='{task}' filter); "
             f"eval: {len(raw_eval)}")

    # ---- Datasets + collator (identical shape to SFT) ----
    class CLCEGuardian(TorchDataset):
        """Returns the SFT-style messages dict only — class IDs are
        recovered inside compute_loss by decoding the unmasked
        assistant-turn tokens out of `inputs['labels']`. This sidesteps
        SFTTrainer's collator-replacement behaviour."""
        def __init__(self, ds, idx_list):
            self.ds = ds
            self.idx = list(idx_list)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            sample = self.ds[self.idx[i]]
            return {"messages": build_messages(sample, include_assistant=True)}

    cl_ce_train = CLCEGuardian(raw_train, train_only_idx)
    cl_ce_val = CLCEGuardian(raw_train, val_idx) if val_idx else None

    try:
        from unsloth.trainer import UnslothVisionDataCollator
        collator = UnslothVisionDataCollator(model, processor)
        collator_name = "UnslothVisionDataCollator"
    except Exception as e:
        raise RuntimeError(
            "UnslothVisionDataCollator is required for CL+CE label masking; "
            f"underlying: {e!r}"
        )
    log.info(f"collator: {collator_name}")
    blob["collator"] = collator_name

    # Lookup table: lowercase fine-grained label string -> position in
    # the present-class space. `parse_failure_mode` returns a unified
    # label name; we map back to the cid_to_pos position for InfoNCE.
    label_str_to_pos: dict[str, int] = {
        UNIFIED_LABEL_NAMES[c].lower(): cid_to_pos[c]
        for c in present_class_ids
    }

    # ---- Training hyperparameters ----
    EPOCHS = int(config.get("epochs", 2))
    GRAD_ACCUM = int(config.get("grad_accum", 4))
    LR = float(config.get("lr", 2e-4))
    # Defaults (post-rebuild 2026-05-09):
    #   lambda_c = 0.1 — CE must dominate; InfoNCE is a regulariser, not
    #     the primary signal. λ=1.0 (old default) caused mode collapse
    #     because the InfoNCE term overwhelmed the LM-head supervision.
    #   tau = 1.0 — low temperature (e.g. 0.07) saturates gradients on
    #     small batches (Phase-3 evidence) and amplifies the contrastive
    #     gradient relative to CE, replicating the same overwhelm.
    LAMBDA_C = float(config.get("lambda_contrastive", 0.1))
    TEMPERATURE = float(config.get("temperature", 1.0))
    EARLY_STOPPING_PATIENCE = int(config.get("early_stopping_patience", 2))
    use_early_stopping = cl_ce_val is not None and EARLY_STOPPING_PATIENCE > 0

    # GradCache option remains in the config schema for compatibility but
    # we explicitly do not honour it — see RULE 4 in the module docstring.
    if int(config.get("gc_batch_size", 0) or 0) > BATCH_SIZE:
        log.warning(
            f"gc_batch_size={config.get('gc_batch_size')} ignored — CL+CE "
            f"deliberately uses per-mini-batch InfoNCE."
        )

    # ---- Build the SFT trainer; subclass to add the InfoNCE term ----
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback

    sft_kwargs = dict(
        output_dir=str(ckpt_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=["tensorboard"],
        logging_dir=str(run_dir / "tensorboard"),
        max_seq_length=2048,
        seed=seed,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    if use_early_stopping:
        sft_kwargs["eval_strategy"] = "epoch"
        sft_kwargs["load_best_model_at_end"] = True
        sft_kwargs["metric_for_best_model"] = "eval_loss"
        sft_kwargs["greater_is_better"] = False
    try:
        sft_config = SFTConfig(**sft_kwargs)
    except TypeError:
        for drop in ("dataset_kwargs", "max_seq_length"):
            sft_kwargs.pop(drop, None)
        sft_config = SFTConfig(**sft_kwargs)

    contrastive_loss_fn = InfoNCELoss(temperature=TEMPERATURE)

    class CLCETrainer(SFTTrainer):
        """SFTTrainer + extra InfoNCE term on the last-attended-token hidden
        state. Eval skips the contrastive term (model.training is False)
        so eval_loss = CE only and early stopping is on a clean target.

        Class IDs for InfoNCE are recovered inside compute_loss by decoding
        the unmasked assistant-turn tokens out of `inputs['labels']` and
        looking up the resulting string in `_label_str_to_pos`."""

        def __init__(self, *args, proj_head_, lambda_c, contrastive_fn,
                     tokenizer_, label_str_to_pos, **kw):
            super().__init__(*args, **kw)
            self._proj_head = proj_head_
            self._lambda_c = lambda_c
            self._contrastive_fn = contrastive_fn
            self._tokenizer = tokenizer_
            self._label_str_to_pos = label_str_to_pos
            self._running = {"ce": 0.0, "cl": 0.0, "count": 0}

        def create_optimizer(self):
            super().create_optimizer()
            proj_params = [p for p in self._proj_head.parameters()
                           if p.requires_grad]
            if proj_params:
                self.optimizer.add_param_group({
                    "params": proj_params,
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                })
            return self.optimizer

        def _recover_class_ids(self, label_tok):
            """Decode the unmasked positions of `inputs['labels']` (the
            assistant turn) and map the resulting string to a position in
            the present-class space. Returns a 1-D long tensor on CPU; -1
            for any row whose decoded string isn't in the lookup."""
            class_ids: list[int] = []
            for row in label_tok:
                masked = (row != -100)
                if not bool(masked.any()):
                    class_ids.append(-1); continue
                tok_ids = row[masked].tolist()
                txt = self._tokenizer.decode(
                    tok_ids, skip_special_tokens=True).strip().lower()
                cid = self._label_str_to_pos.get(txt, -1)
                # Fallback: substring match (in case the decoder leaves
                # whitespace/punctuation around the label name).
                if cid < 0:
                    for k, v in self._label_str_to_pos.items():
                        if k in txt:
                            cid = v; break
                class_ids.append(cid)
            return torch.tensor(class_ids, dtype=torch.long)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs, output_hidden_states=True)
            ce = outputs.loss
            cl = ce.new_zeros(())
            if self._lambda_c > 0 and model.training:
                label_tok = inputs.get("labels", None)
                if label_tok is not None:
                    class_labels = self._recover_class_ids(label_tok).to(
                        outputs.hidden_states[-1].device)
                    valid = class_labels >= 0
                    if int(valid.sum()) >= 2:
                        hidden = outputs.hidden_states[-1]
                        attn_mask = inputs["attention_mask"]
                        seq_lens = attn_mask.sum(dim=1) - 1
                        idx = torch.arange(hidden.size(0),
                                           device=hidden.device)
                        eos = hidden[idx, seq_lens]
                        eos_v = eos[valid]
                        proj = F.normalize(
                            self._proj_head(eos_v.float()), dim=-1)
                        cl = self._contrastive_fn(
                            proj, labels=class_labels[valid])
                        self._running["ce"] += float(ce.detach())
                        self._running["cl"] += float(cl.detach())
                        self._running["count"] += 1
            total = ce + self._lambda_c * cl
            return (total, outputs) if return_outputs else total

        def log(self, logs, *args, **kwargs):
            # Surface CL+CE-specific metrics at every `logging_steps` boundary.
            if self._running["count"] > 0:
                n = self._running["count"]
                ce_avg = self._running["ce"] / n
                cl_avg = self._running["cl"] / n
                logs["cl_ce_ce"] = ce_avg
                logs["cl_ce_cl"] = cl_avg
                if ce_avg > 1e-8:
                    logs["cl_ce_ratio"] = cl_avg / ce_avg
                self._running = {"ce": 0.0, "cl": 0.0, "count": 0}
            return super().log(logs, *args, **kwargs)

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=cl_ce_train,
        data_collator=collator,
        processing_class=processor,
        proj_head_=proj_head,
        lambda_c=LAMBDA_C,
        contrastive_fn=contrastive_loss_fn,
        tokenizer_=tokenizer,
        label_str_to_pos=label_str_to_pos,
    )
    if use_early_stopping:
        trainer_kwargs["eval_dataset"] = cl_ce_val
    trainer = CLCETrainer(**trainer_kwargs)
    if use_early_stopping:
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        )
    blob["early_stopping"] = {
        "enabled": use_early_stopping,
        "patience": EARLY_STOPPING_PATIENCE if use_early_stopping else None,
        "n_val": len(val_idx),
    }

    # ---- Train ----
    log.info(f"Training: epochs={EPOCHS} bs={BATCH_SIZE} ga={GRAD_ACCUM} "
             f"lr={LR} lambda_c={LAMBDA_C} tau={TEMPERATURE} "
             f"early_stopping={use_early_stopping}  "
             f"(InfoNCE labels = fine-grained, no GradCache)")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_output = trainer.train()
    blob["train"] = {
        "epochs": EPOCHS, "bs": BATCH_SIZE, "ga": GRAD_ACCUM, "lr": LR,
        "lambda_c": LAMBDA_C, "tau": TEMPERATURE,
        "global_steps": int(train_output.global_step),
        "final_train_loss": float(train_output.training_loss),
        "train_time_sec": round(time.time() - t0, 2),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    save_results_atomically(blob, metrics_path)

    # ---- Eval (generative — identical to SFT) ----
    log.info(f"Generative eval on {config['dataset_eval']} "
             f"({len(raw_eval)} samples) ...")
    model.train(False)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        try:
            model.generation_config.enable_thinking = False
        except Exception:
            pass

    predictions: list[str] = []
    ground_truths: list[str] = []
    raw_decodes: list[str] = []
    t0 = time.time()
    for i in range(len(raw_eval)):
        sample = raw_eval[i]
        msgs = build_messages(sample, include_assistant=False)
        text = apply_chat_template_safe(processor, msgs, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[sample["images"]],
            return_tensors="pt", padding=True,
        ).to("cuda")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                **inputs,
                max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_tokens = out[0][prompt_len:]
        decoded = processor.decode(gen_tokens, skip_special_tokens=True)
        pred = parse_failure_mode(decoded)
        predictions.append(pred)
        ground_truths.append(sample["failure_mode"])
        raw_decodes.append(decoded)
        if (i + 1) % 100 == 0:
            log.info(f"  eval {i+1}/{len(raw_eval)} done")
    eval_time = time.time() - t0

    # Audit P2.1 — restrict label_names to classes present in train so
    # absent classes don't dilute macro-F1.
    present_names = present_label_names_for_task(raw_train, train_only_idx, task)
    gt_mapped, pred_mapped, label_names = remap_strings_for_task(
        ground_truths, predictions, task,
        present_label_names=present_names,
    )
    m = compute_classification_metrics(gt_mapped, pred_mapped, label_names)
    log.info(f"CL+CE (generative): acc={m['accuracy']:.4f} "
             f"f1_macro={m['f1_macro']:.4f}")
    if m.get("majority_class_warning"):
        log.warning(m["majority_class_warning"])

    blob.update({
        "metrics": m,
        "eval_time_sec": round(eval_time, 2),
        "ms_per_sample": round(1000 * eval_time / max(1, len(predictions)), 2),
        "n_unknown": sum(1 for p in predictions if p == "unknown"),
        "predictions": predictions,
        "ground_truths": ground_truths,
        "raw_decodes_first_5": raw_decodes[:5],
        "present_class_ids": present_class_ids,
        "present_label_names": present_label_names,
        "n_classes_present": len(present_class_ids),
        "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": True,
    })
    finalize_metrics_schema(blob)
    save_results_atomically(blob, metrics_path)
    write_done_flag(run_dir, {
        "method": "cl_ce", "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
    })
    return blob


# ---------------------------------------------------------------------------
# Eval-only mode — load LoRA adapter and run generative eval.
# CL+CE's inference is generative-only (the InfoNCE term is training-only;
# we don't ship the proj_head into inference). So eval_cl_ce mirrors eval_sft.
# ---------------------------------------------------------------------------

def _find_latest_lora_checkpoint(ckpt_dir: Path) -> Path:
    inner = ckpt_dir / "checkpoint"
    if not inner.exists():
        raise FileNotFoundError(f"checkpoint/ subdir not found at {inner}")
    subs = sorted(
        [p for p in inner.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not subs:
        raise FileNotFoundError(f"no checkpoint-N/ dirs under {inner}")
    return subs[-1]


def eval_cl_ce(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    ckpt_dir = Path(config["from_checkpoint"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(
        run_dir, name=f"cl_ce.eval.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    log.info(f"=== eval_cl_ce | from_checkpoint={ckpt_dir} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "cl_ce",
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

    log.info("Loading base Qwen3.5-4B (4-bit) ...")
    t0 = time.time()
    model, processor = FastModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    adapter_path = _find_latest_lora_checkpoint(ckpt_dir)
    log.info(f"Loading LoRA adapter from {adapter_path} ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))
    blob["load_time_sec"] = round(time.time() - t0, 2)

    raw_eval = GuardianDataset(config["dataset_eval"], corruption_name=config.get("corruption"), severity=config.get("severity"))
    model.train(False)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        try:
            model.generation_config.enable_thinking = False
        except Exception:
            pass

    predictions: list[str] = []
    ground_truths: list[str] = []
    raw_decodes: list[str] = []
    t0 = time.time()
    for i in range(len(raw_eval)):
        sample = raw_eval[i]
        msgs = build_messages(sample, include_assistant=False)
        text = apply_chat_template_safe(processor, msgs, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[sample["images"]],
            return_tensors="pt", padding=True,
        ).to("cuda")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_tokens = out[0][prompt_len:]
        decoded = processor.decode(gen_tokens, skip_special_tokens=True)
        pred = parse_failure_mode(decoded)
        predictions.append(pred)
        ground_truths.append(sample["failure_mode"])
        raw_decodes.append(decoded)
        if (i + 1) % 100 == 0:
            log.info(f"  eval {i+1}/{len(raw_eval)} done")
    eval_time = time.time() - t0

    # Per-task label space (default). For binary the default is
    # ["success","failure"]; previously this passed UNIFIED_LABEL_NAMES
    # which dragged macro-F1 down across 6 absent classes (BDV2 E5.2
    # reported 0.045 instead of ~0.55).
    gt_mapped, pred_mapped, label_names = remap_strings_for_task(
        ground_truths, predictions, task,
    )
    m = compute_classification_metrics(gt_mapped, pred_mapped, label_names)
    log.info(f"CL+CE eval-only: acc={m['accuracy']:.4f} "
             f"f1_macro={m['f1_macro']:.4f}")
    if m.get("majority_class_warning"):
        log.warning(m["majority_class_warning"])

    blob.update({
        "metrics": m,
        "eval_time_sec": round(eval_time, 2),
        "ms_per_sample": round(1000 * eval_time / max(1, len(predictions)), 2),
        "n_unknown": sum(1 for p in predictions if p == "unknown"),
        "predictions": predictions,
        "ground_truths": ground_truths,
        "raw_decodes_first_5": raw_decodes[:5],
        "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished": True,
    })
    finalize_metrics_schema(blob)
    save_results_atomically(blob, metrics_path)
    write_done_flag(run_dir, {
        "method": "cl_ce", "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
        "eval_only": True,
    })
    return blob
