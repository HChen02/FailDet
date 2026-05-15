"""Supervised Fine-Tuning method (Unsloth + TRL SFTTrainer).

Parameterized version of run_phase2.py. Trains Qwen3.5-4B with QLoRA
4-bit + LoRA r=16 on the chat-formatted (image, prompt → label) task,
evaluates by `model.generate(...)` + label parsing.
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
    capture_environment, finalize_metrics_schema, make_worker_init_fn,
    parse_failure_mode, present_label_names_for_task, remap_strings_for_task,
    save_results_atomically, select_indices, set_seed, setup_run_logger,
    write_done_flag,
)


def train_sft(config: dict) -> dict:
    """Train SFT and evaluate on `config["dataset_eval"]`."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    run_dir = Path(config["run_dir"])
    log = setup_run_logger(run_dir, name=f"sft.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"
    ckpt_dir = run_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== train_sft | exp_id={config.get('exp_id')} seed={seed} "
             f"task={config.get('task','8class')} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "sft",
        "seed": seed,
        "config": dict(config),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_at": time.time(),
        "environment": capture_environment(),
    }
    save_results_atomically(blob, metrics_path)

    import torch
    from torch.utils.data import Dataset as TorchDataset
    from unsloth import FastModel
    from data.dataset import GuardianDataset
    from evaluation.metrics import compute_classification_metrics

    log.info("Loading model + applying LoRA ...")
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
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        random_state=seed,
    )
    blob["load_time_sec"] = round(time.time() - t0, 2)
    save_results_atomically(blob, metrics_path)

    # ---- Data ----
    raw_train = GuardianDataset(config["dataset_train"])
    raw_eval = GuardianDataset(config["dataset_eval"], corruption_name=config.get("corruption"), severity=config.get("severity"))
    train_idx = select_indices(
        len(raw_train),
        data_fraction=float(config.get("data_fraction", 1.0)),
        seed=seed,
    )
    # Carve a small held-out validation slice off training (NOT the test set —
    # that would leak the gate metric back into model selection). Used for
    # eval_loss / early-stopping only; final test eval still uses raw_eval.
    val_fraction = float(config.get("val_fraction", 0.05))
    import numpy as np
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_idx))
    n_val = max(8, int(round(val_fraction * len(train_idx)))) if val_fraction > 0 else 0
    if n_val >= len(train_idx):
        n_val = max(0, len(train_idx) // 10)
    val_idx = [train_idx[int(perm[i])] for i in range(n_val)]
    train_only_idx = [train_idx[int(perm[i])] for i in range(n_val, len(train_idx))]
    log.info(f"train: {len(raw_train)} -> {len(train_only_idx)} train + "
             f"{len(val_idx)} val (val_fraction={val_fraction}); "
             f"eval: {len(raw_eval)}")

    class SFTGuardian(TorchDataset):
        def __init__(self, ds, idx_list):
            self.ds = ds
            self.idx = list(idx_list)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            sample = self.ds[self.idx[i]]
            return {"messages": build_messages(sample, include_assistant=True)}

    sft_train = SFTGuardian(raw_train, train_only_idx)
    sft_val = SFTGuardian(raw_train, val_idx) if val_idx else None

    # Use Unsloth's vision collator (handles label-masking) when available.
    try:
        from unsloth.trainer import UnslothVisionDataCollator
        collator = UnslothVisionDataCollator(model, processor)
        collator_name = "UnslothVisionDataCollator"
    except Exception as e:
        raise RuntimeError(
            "UnslothVisionDataCollator is required for proper assistant-turn "
            f"label masking; manual fallback would silently train on the "
            f"prompt instead of the label. Underlying: {e!r}"
        )
    log.info(f"collator: {collator_name}")
    blob["collator"] = collator_name

    # ---- Build trainer ----
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback

    # Default 5 (was 10) — Phase 2 + E1 dryrun showed loss plateaus by ep 3-5.
    # The callback will cut runs that plateau even earlier.
    epochs = int(config.get("epochs", 5))
    bs = int(config.get("batch_size", 4))
    ga = int(config.get("grad_accum", 4))
    lr = float(config.get("lr", 2e-4))
    early_stopping_patience = int(config.get("early_stopping_patience", 2))
    use_early_stopping = sft_val is not None and early_stopping_patience > 0

    sft_kwargs = dict(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
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
        # save & eval cadence must match for load_best_model_at_end.
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

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=sft_train,
        data_collator=collator,
        processing_class=processor,
    )
    if use_early_stopping:
        trainer_kwargs["eval_dataset"] = sft_val
    trainer = SFTTrainer(**trainer_kwargs)
    if use_early_stopping:
        trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
    blob["early_stopping"] = {
        "enabled": use_early_stopping,
        "patience": early_stopping_patience if use_early_stopping else None,
        "n_val": len(val_idx),
    }

    # ---- Train ----
    log.info(f"Training: epochs={epochs} bs={bs} ga={ga} lr={lr} "
             f"effective_batch={bs*ga}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_output = trainer.train()
    blob["train"] = {
        "epochs": epochs, "bs": bs, "ga": ga, "lr": lr,
        "global_steps": int(train_output.global_step),
        "final_train_loss": float(train_output.training_loss),
        "train_time_sec": round(time.time() - t0, 2),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    save_results_atomically(blob, metrics_path)

    # ---- Eval ----
    log.info(f"Evaluating on {config['dataset_eval']} ({len(raw_eval)} samples) ...")
    model.train(False)
    # CB-4: prevent <think>...</think> preambles from poisoning the parser.
    # parse_failure_mode picks the first matching label name in the decoded
    # string; a think block that mentions a wrong class would win.
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

    task = str(config.get("task", "8class"))
    # Audit P2.1 — restrict label_names to classes actually present in
    # train so wrong_state (0 RLBench train samples) doesn't dilute
    # macro-F1 with a zero-support row.
    present_names = present_label_names_for_task(raw_train, train_only_idx, task)
    gt_mapped, pred_mapped, label_names = remap_strings_for_task(
        ground_truths, predictions, task,
        present_label_names=present_names,
    )
    m = compute_classification_metrics(gt_mapped, pred_mapped, label_names)
    log.info(f"SFT: acc={m['accuracy']:.4f} "
             f"f1_macro={m['f1_macro']:.4f} "
             f"f1_weighted={m['f1_weighted']:.4f}")
    if m["majority_class_warning"]:
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
        "method": "sft", "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
    })
    return blob


# ---------------------------------------------------------------------------
# Eval-only mode — load LoRA adapter from a prior run_dir's checkpoint dir
# and run model.generate() on the new --dataset-eval. No training.
# Added 2026-05-13 for cross-domain transfer (E5.2 etc.).
# ---------------------------------------------------------------------------

def _find_latest_lora_checkpoint(ckpt_dir: Path) -> Path:
    """Find the highest-numbered `checkpoint-N/` subdir under
    `<ckpt_dir>/checkpoint/`. With early stopping + save_total_limit=2 the
    final-epoch adapter is the closest available proxy for the
    'best' in-memory model."""
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


def eval_sft(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    task = str(config.get("task", "8class"))
    ckpt_dir = Path(config["from_checkpoint"])
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_run_logger(
        run_dir, name=f"sft.eval.{config.get('exp_id')}.{seed}")
    metrics_path = run_dir / "metrics.json"

    log.info(f"=== eval_sft | from_checkpoint={ckpt_dir} ===")

    blob: dict = {
        "exp_id": config.get("exp_id"),
        "method": "sft",
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

    # Locate the LoRA adapter and load it.
    adapter_path = _find_latest_lora_checkpoint(ckpt_dir)
    log.info(f"Loading LoRA adapter from {adapter_path} ...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter_path))
    blob["load_time_sec"] = round(time.time() - t0, 2)

    # Eval-only data: just the test split.
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

    # Per-task label space. Don't override with UNIFIED_LABEL_NAMES — that
    # was a bug: for binary the default is ["success","failure"]; passing
    # the 8 unified names made f1_macro average over 6 absent classes and
    # crashed the headline number (BDV2 E5.2 ran at ~0.045 instead of
    # ~0.55). Let remap_strings_for_task pick the correct default per task.
    gt_mapped, pred_mapped, label_names = remap_strings_for_task(
        ground_truths, predictions, task,
    )
    m = compute_classification_metrics(gt_mapped, pred_mapped, label_names)
    log.info(f"SFT eval-only: acc={m['accuracy']:.4f} "
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
        "method": "sft", "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
        "eval_only": True,
    })
    return blob
