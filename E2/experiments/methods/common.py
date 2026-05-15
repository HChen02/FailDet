"""Shared utilities for the per-method training functions.

Lives separate from the phase scripts so the dispatch path doesn't have
to import them as modules (the phase scripts run as `__main__` and do
file-side-effecty things at import time — log file creation, dir creation,
etc.).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 8-class unified label vocabulary. Keep in lock-step with
# data/dataset.py:LABEL_TO_ID.
UNIFIED_LABEL_NAMES = [
    "success", "no_grasp", "slip", "translation",
    "rotation", "wrong_object", "wrong_sequence", "wrong_state",
]
NUM_CLASSES = 8

# VQA prompt — identical to run_phase2.py / run_phase4.py / run_e0_zeroshot.py.
PROMPT_TEMPLATE = (
    "Task: {task}\n"
    "Subtask: {subtask}\n\n"
    "Look at these 6 multi-view images showing the robot before and after "
    "executing this subtask. Classify the outcome as exactly one of: "
    "success, no_grasp, slip, translation, rotation, wrong_object, "
    "wrong_sequence, wrong_state"
)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed all RNGs used by torch / numpy / random / cuda."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(seed: int):
    """Build a deterministic DataLoader worker_init_fn for the given seed."""
    def _init(worker_id: int) -> None:
        import random
        import numpy as np
        s = (seed + worker_id) % (2**32)
        np.random.seed(s)
        random.seed(s)
    return _init


def capture_environment() -> dict:
    """Snapshot the runtime environment for the results JSON.

    Includes: pip freeze, GPU model, CUDA / torch versions, git HEAD.
    Every field is best-effort; failures are captured as their string
    representation rather than crashing the run.
    """
    env: dict = {}
    try:
        env["pip_freeze"] = subprocess.check_output(
            ["pip", "freeze"], stderr=subprocess.STDOUT,
        ).decode()
    except Exception as e:
        env["pip_freeze_error"] = repr(e)
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_version"] = torch.version.cuda
        env["gpu_name"] = (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        )
        if torch.cuda.is_available():
            env["gpu_total_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 3,
            )
    except Exception as e:
        env["torch_error"] = repr(e)
    try:
        env["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.STDOUT,
        ).decode().strip()
    except Exception as e:
        env["git_hash_error"] = repr(e)
    env["python_version"] = sys.version
    env["host"] = os.uname().nodename
    return env


def setup_run_logger(run_dir: Path, name: str) -> logging.Logger:
    """File + stdout logger writing to run_dir/train.log."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def save_results_atomically(results: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(path)


def write_done_flag(run_dir: Path, payload: Optional[dict] = None) -> None:
    """Mark a run as complete. The orchestrator skips runs whose run_dir
    already contains done.flag."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    flag = run_dir / "done.flag"
    body = {"finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if payload:
        body.update(payload)
    with open(flag, "w") as f:
        json.dump(body, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Chat helpers (same prompt as Phase 2 / 4 / E0)
# ---------------------------------------------------------------------------

def build_messages(sample: dict, *, include_assistant: bool) -> list[dict]:
    """Convert a GuardianDataset sample into Qwen3.5 chat format."""
    user_content: list[dict] = [
        {"type": "image", "image": img} for img in sample["images"]
    ]
    user_content.append({
        "type": "text",
        "text": PROMPT_TEMPLATE.format(
            task=sample["task_instruction"],
            subtask=sample["detailed_subtask_name"],
        ),
    })
    messages = [{"role": "user", "content": user_content}]
    if include_assistant:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["failure_mode"]}],
        })
    return messages


def apply_chat_template_safe(processor, messages, *, add_generation_prompt: bool):
    """Wrap apply_chat_template with `enable_thinking=False`; retry without
    it if the installed processor version doesn't accept that kwarg."""
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def parse_failure_mode(generated_text: str) -> str:
    """Pick the first unified label name that appears in the generation,
    longest-first to avoid 'no_grasp' matching inside a longer label."""
    text_lower = generated_text.lower()
    for name in sorted(UNIFIED_LABEL_NAMES, key=len, reverse=True):
        if name in text_lower:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Data subset selection (E2 low-data + E6 holdout)
# ---------------------------------------------------------------------------

def select_indices(
    n_total: int,
    *,
    data_fraction: float,
    seed: int,
    low_data_splits_path: Optional[Path] = None,
) -> list[int]:
    """Return the indices the runner should train on.

    For ``data_fraction == 1.0`` returns the full range. For smaller
    fractions, look up the precomputed stratified split in
    ``data/low_data_splits.json`` keyed by ``f"{int(round(100*pct))}pct_seedN"``;
    fall back to a random subset of the same size if the key is missing.
    """
    if data_fraction == 1.0 or data_fraction is None:
        return list(range(n_total))
    if low_data_splits_path is None:
        low_data_splits_path = PROJECT_ROOT / "data" / "low_data_splits.json"
    pct_key = f"{int(round(100 * float(data_fraction)))}pct_seed{int(seed)}"
    if low_data_splits_path.exists():
        with open(low_data_splits_path) as f:
            splits = json.load(f)
        if pct_key in splits:
            return list(splits[pct_key])
    # Fallback: random subset.
    import numpy as np
    rng = np.random.default_rng(seed)
    n_pick = max(1, int(round(data_fraction * n_total)))
    idx = rng.choice(n_total, size=n_pick, replace=False)
    return sorted(int(i) for i in idx)


def present_label_names_for_task(
    raw_train, indices: list[int], task: str,
) -> list[str]:
    """Convenience wrapper around compute_present_classes that returns
    only the per-task label-name list, filtered to the classes that
    actually appear in the training subset.

    Used by SFT / CL+CE generative metric computation to keep the label
    space comparable to the centroid-based methods (audit P2.1).
    """
    _, names, _ = compute_present_classes(raw_train, indices, task)
    return names


def remap_strings_for_task(
    ground_truths: list[str], predictions: list[str], task: str,
    *, present_label_names: Optional[list[str]] = None,
) -> tuple[list[str], list[str], list[str]]:
    """Map free-form 8-class label strings into the per-task label space
    for sklearn metric computation.

    Returns ``(ground_truths_out, predictions_out, label_names)``.

    - ``binary``: any failure name (no_grasp, slip, ...) collapses to
      ``"failure"``; ``"success"`` stays. label_names = ["success","failure"].
      Without this remap, ``f1_macro`` would average over all 8 unified
      classes and the 6 absent classes contribute F1=0, dragging the
      macro to a misleading low value (the 0.250 on a perfect SFT seen
      in the quick test).
    - ``7class``: drop every sample whose ground truth is "success" so the
      accuracy denominator matches across method families (CL methods
      already filter at the dataloader level; SFT/CL+CE go through
      generative inference and don't). label_names = the 7 failure-type
      names. (CB-2 fix from results/audit_synthesis.md.)
    - ``8class``: pass-through; label_names = the full unified 8.
    """
    if task == "8class":
        gt_out, pred_out = list(ground_truths), list(predictions)
        default_names = list(UNIFIED_LABEL_NAMES)
    elif task == "7class":
        gt_out = []
        pred_out = []
        for g, p in zip(ground_truths, predictions):
            if g == "success":
                continue
            gt_out.append(g)
            pred_out.append(p)
        default_names = list(UNIFIED_LABEL_NAMES[1:])
    elif task == "binary":
        def _to_bin(name: str) -> str:
            if name == "success":
                return "success"
            if name in UNIFIED_LABEL_NAMES:
                return "failure"
            # "unknown" or anything else stays as-is so sklearn counts it
            # as a wrong prediction (not in label_names → never a TP).
            return name
        gt_out = [_to_bin(g) for g in ground_truths]
        pred_out = [_to_bin(p) for p in predictions]
        default_names = ["success", "failure"]
    else:
        raise ValueError(f"Unknown task: {task!r}")

    # Audit P2.1 — let the caller substitute a present-classes-only
    # label_names list so absent classes (e.g. wrong_state on RLBench)
    # don't dilute macro-F1 with a 0/0 row.
    label_names = (list(present_label_names)
                   if present_label_names is not None
                   else default_names)
    return gt_out, pred_out, label_names


def restrict_labels_for_task(
    labels: list[int], *, task: str,
) -> tuple[list[int], list[str]]:
    """Map the 8-class labels to the requested task's label space.

    Args:
        task: "8class" — pass-through.
              "binary" — collapse to 0=success, 1=any failure.
              "7class" — drop success samples and re-index 1..7 → 0..6.

    Returns: (new_labels, new_label_names)
    """
    if task == "8class":
        return labels, list(UNIFIED_LABEL_NAMES)
    if task == "binary":
        return [0 if int(l) == 0 else 1 for l in labels], ["success", "failure"]
    if task == "7class":
        # Caller is responsible for filtering out success samples; we just
        # remap the surviving labels.
        return [int(l) - 1 for l in labels if int(l) != 0], list(UNIFIED_LABEL_NAMES[1:])
    raise ValueError(f"Unknown task: {task!r}")


def task_label_names_full(task: str) -> list[str]:
    """The full per-task label space (no per-dataset filtering applied).

    binary  -> ["success", "failure"]
    7class  -> 7 failure-type names (UNIFIED_LABEL_NAMES[1:])
    8class  -> all 8 unified names
    """
    if task == "binary":
        return ["success", "failure"]
    if task == "7class":
        return list(UNIFIED_LABEL_NAMES[1:])
    if task == "8class":
        return list(UNIFIED_LABEL_NAMES)
    raise ValueError(f"Unknown task: {task!r}")


def remap_label_for_task(failure_label: int, task: str) -> int:
    """Map a raw 8-class failure_label (0..7) into the per-task index space.

    For 7class success (0) maps to -1 — caller is responsible for filtering
    success samples at the dataloader level before invoking this remap.
    """
    fl = int(failure_label)
    if task == "8class":
        return fl
    if task == "binary":
        return 0 if fl == 0 else 1
    if task == "7class":
        return -1 if fl == 0 else fl - 1
    raise ValueError(f"Unknown task: {task!r}")


def compute_present_classes(
    raw_dataset, indices: list[int], task: str,
) -> tuple[list[int], list[str], list[int]]:
    """Determine which per-task class ids actually have samples in the
    training subset, and remap labels to that subset.

    Background — RLBench-Fail train has 0 ``wrong_state`` samples, so any
    method building a centroid set for that class would include a zero
    vector that randomly attracts test points (the failure mode that
    surfaced in the 2026-05 CL-FT sweep). Filtering to *present* classes
    only is the audit P2.1 fix.

    Returns:
        present_classes:  sorted list of per-task class ids with >=1
                          sample in `indices`. For ``7class`` on RLBench
                          this is [0, 1, 2, 3, 4, 5] (wrong_state, id 6,
                          excluded). For ``8class`` on RLBench this is
                          [0, 1, 2, 3, 4, 5, 6] (wrong_state, id 7,
                          excluded).
        present_label_names:  the corresponding label-name list, suitable
                              for ``compute_classification_metrics``.
        per_index_labels:  list parallel to ``indices``, holding the
                           remapped per-task class id for each kept
                           sample (or -1 for samples that should be
                           dropped — only happens for ``7class`` success
                           if the caller didn't pre-filter).
    """
    full_names = task_label_names_full(task)
    per_index_labels: list[int] = []
    counts: dict[int, int] = {}
    for i in indices:
        sample = raw_dataset[i]
        lab = remap_label_for_task(int(sample["failure_label"]), task)
        per_index_labels.append(lab)
        if lab < 0:
            continue
        counts[lab] = counts.get(lab, 0) + 1
    present_classes = sorted(c for c, n in counts.items() if n > 0)
    present_label_names = [full_names[c] for c in present_classes]
    return present_classes, present_label_names, per_index_labels


def majority_class_warn(predictions: list[str], logger: logging.Logger) -> Optional[str]:
    """Log a loud warning if a single class dominates predictions and return
    the warning string (None if no warning)."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from evaluation.metrics import majority_class_check
    msg = majority_class_check(predictions)
    if msg:
        logger.warning(msg)
    return msg


# ---------------------------------------------------------------------------
# Cross-method metrics schema
# ---------------------------------------------------------------------------
# Every method's final metrics.json should expose the same top-level keys
# so the dryrun summary, run_sweep, and analysis scripts don't have to
# special-case each method. The nested `metrics.*` block is left intact for
# backward compatibility with code that already reads it.
TOP_LEVEL_METRICS_KEYS = (
    "accuracy", "f1_macro", "f1_weighted", "per_class_f1",
    "confusion_matrix", "n_total", "n_unknown",
    "peak_gpu_mem_gb", "train_time_sec",
)


def finalize_metrics_schema(blob: dict) -> dict:
    """Promote a method's metrics into a uniform top-level schema.

    Reads from blob["metrics"], blob["train"|"cl_train"|"cl_ce_train"], and
    blob["predictions"], and writes flat top-level keys onto blob:
        accuracy, f1_macro, f1_weighted, per_class_f1, confusion_matrix,
        n_total, n_unknown, peak_gpu_mem_gb, train_time_sec.
    Existing top-level values win — we only fill missing keys.
    """
    m = blob.get("metrics") or {}
    blob.setdefault("accuracy", m.get("accuracy"))
    blob.setdefault("f1_macro", m.get("f1_macro"))
    blob.setdefault("f1_weighted", m.get("f1_weighted"))

    per_class = m.get("per_class") or m.get("per_class_f1") or {}
    if isinstance(per_class, dict):
        # Both shapes appear in the wild: {label: {f1: x, ...}} (sklearn-ish)
        # and {label: x} (already-flat). Coerce to the flat one.
        flat: dict = {}
        for k, v in per_class.items():
            flat[k] = float(v["f1"]) if isinstance(v, dict) and "f1" in v else (
                float(v) if isinstance(v, (int, float)) else None)
        blob.setdefault("per_class_f1", flat)
    else:
        blob.setdefault("per_class_f1", {})

    blob.setdefault("confusion_matrix", m.get("confusion_matrix"))

    n_total = m.get("n_samples")
    if n_total is None:
        preds = blob.get("predictions") or []
        n_total = len(preds) if preds else None
    blob.setdefault("n_total", n_total)

    if blob.get("n_unknown") is None:
        preds = blob.get("predictions") or []
        blob["n_unknown"] = sum(1 for p in preds if p == "unknown")

    # Older runs wrote `cafe_train`; the rebuilt CL+CE writes `train`.
    # Keep cafe_train as a legacy fallback so old metrics.json files still
    # parse cleanly.
    train_block = (blob.get("train") or blob.get("cl_train")
                   or blob.get("cl_ce_train") or blob.get("cafe_train") or {})
    if blob.get("peak_gpu_mem_gb") is None:
        blob["peak_gpu_mem_gb"] = train_block.get("peak_gpu_mem_gb")
    if blob.get("train_time_sec") is None:
        blob["train_time_sec"] = train_block.get("train_time_sec")

    return blob
