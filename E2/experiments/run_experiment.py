"""Single-experiment dispatch entry point.

Loads a config either (a) from `configs/experiments.yaml` via `--from-config`,
(b) from CLI flags only, or (c) a mix (CLI overrides yaml). Resolves it to
the appropriate method runner and writes results to
``results/{exp_id}_seed{seed}/metrics.json`` plus a ``done.flag``.

Examples
--------
# Single config from yaml + a specific seed:
  python experiments/run_experiment.py \
      --from-config E1.3_SFT_8class --seed 42

# Fully CLI-driven (no yaml):
  python experiments/run_experiment.py \
      --exp-id E1.3 --method sft --task 8class \
      --dataset-train paulpacaud/rlbenchfail_train_dataset \
      --dataset-eval  paulpacaud/rlbenchfail_test_dataset \
      --seed 42 --epochs 10 --batch-size 4 --grad-accum 4 --lr 2e-4

# Resume / skip if already done:
  Re-running the same command: skips because results/<id>_seed<seed>/done.flag
  exists. Pass --force to re-run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REGISTRY_PATH = PROJECT_ROOT / "configs" / "experiments.yaml"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


METHOD_DISPATCH = {
    # Lazy imports inside _import_method so each method only pays the
    # transformers / peft / Unsloth import cost it actually needs.
    "sft":       ("experiments.methods.sft",      "train_sft"),
    "cl_embed":  ("experiments.methods.cl_embed", "train_cl_embed"),
    "cl_llm":    ("experiments.methods.cl_llm",   "train_cl_llm"),
    "cl_ce":     ("experiments.methods.cl_ce",    "train_cl_ce"),
    "zero_shot": ("experiments.methods.zero_shot", "eval_zero_shot"),
    # DINOv2-based dedicated-encoder baselines (no Unsloth, no Qwen).
    "dino_ce":      ("experiments.methods.dino_ce",      "train_dino_ce"),
    "dino_ce_attn": ("experiments.methods.dino_ce_attn", "train_dino_ce_attn"),
    "dino_cl":      ("experiments.methods.dino_cl",      "train_dino_cl"),
    "dino_cl_attn": ("experiments.methods.dino_cl_attn", "train_dino_cl_attn"),
    "dino_clip":    ("experiments.methods.dino_clip",    "train_dino_clip"),
    # DA3-as-preprocessor methods: depth maps are produced once by DA3,
    # cached to data/depth_cache/, then DINOv2 + CNN + AttentionPool +
    # head trains on the depth maps. DA3 itself is never loaded during
    # training/eval. Added 2026-05-15 (replaces the earlier dual-encoder
    # fusion variant).
    "da3_dino_ce": ("experiments.methods.da3_dino_ce", "train_da3_dino_ce"),
    "da3_dino_cl": ("experiments.methods.da3_dino_cl", "train_da3_dino_cl"),
    # Note: cl_ft (vision-only LoRA + InfoNCE) was retired after the
    # 2026-05 sweep documented its negative result. v6 numbers live at
    # results/E1_cl_ft_*_seed42.
}


# Eval-only dispatch (added 2026-05-13 to support cross-domain transfer
# experiments without retraining). Each entry maps a method to an
# `eval_<method>(config)` function that:
#   1. Loads weights from config["from_checkpoint"]  (path to a prior run_dir)
#   2. Runs only the eval block on config["dataset_eval"]
#   3. Writes metrics.json + done.flag to config["run_dir"]
METHOD_EVAL_DISPATCH = {
    "sft":          ("experiments.methods.sft",          "eval_sft"),
    "cl_embed":     ("experiments.methods.cl_embed",     "eval_cl_embed"),
    "cl_llm":       ("experiments.methods.cl_llm",       "eval_cl_llm"),
    "cl_ce":        ("experiments.methods.cl_ce",        "eval_cl_ce"),
    "dino_ce":      ("experiments.methods.dino_ce",      "eval_dino_ce"),
    "dino_ce_attn": ("experiments.methods.dino_ce_attn", "eval_dino_ce_attn"),
    "dino_cl":      ("experiments.methods.dino_cl",      "eval_dino_cl"),
    "dino_cl_attn": ("experiments.methods.dino_cl_attn", "eval_dino_cl_attn"),
    "dino_clip":    ("experiments.methods.dino_clip",    "eval_dino_clip"),
    "da3_dino_ce": ("experiments.methods.da3_dino_ce", "eval_da3_dino_ce"),
    "da3_dino_cl": ("experiments.methods.da3_dino_cl", "eval_da3_dino_cl"),
}


def _import_method(method: str, eval_only: bool = False):
    table = METHOD_EVAL_DISPATCH if eval_only else METHOD_DISPATCH
    if method not in table:
        raise SystemExit(
            f"Unknown method: {method!r} "
            f"(eval_only={eval_only}). "
            f"Allowed: {sorted(table.keys())}"
        )
    mod_name, fn_name = table[method]
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def _load_yaml() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-config",
                   help="experiment id from configs/experiments.yaml")
    p.add_argument("--exp-id",
                   help="logical experiment id (e.g. E1.3) — required when "
                        "--from-config is omitted")
    p.add_argument("--method",
                   choices=sorted(METHOD_DISPATCH.keys()),
                   help="method to dispatch to")
    p.add_argument("--task", default="8class",
                   choices=["binary", "7class", "8class"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-fraction", type=float, default=1.0)
    p.add_argument("--dataset-train", default=None,
                   help="HF repo id of the train split (omit for zero_shot)")
    p.add_argument("--dataset-eval", default=None,
                   help="HF repo id of the eval split")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lambda-contrastive", type=float, default=None,
                   help="CL+CE λ — weight on the InfoNCE term")
    p.add_argument("--temperature", type=float, default=None,
                   help="SINCERE temperature τ")
    p.add_argument("--gc-batch-size", type=int, default=None,
                   help="CL-Embed GradCache effective contrastive batch "
                        "size (must be a multiple of --batch-size). "
                        "Set to 0 to disable GradCache and use plain "
                        "gradient accumulation. Default 256 inside "
                        "cl_embed.py when unset.")
    p.add_argument("--variant", choices=["frozen", "lora"], default=None,
                   help="DINOv2 method variant: 'frozen' freezes the "
                        "encoder and trains only the head(s); 'lora' "
                        "adds LoRA r=16 on DINOv2 attention. Default "
                        "'frozen' inside each dino_*.py runner.")
    p.add_argument("--lr-encoder", type=float, default=None,
                   help="DINOv2-LoRA encoder learning rate (default "
                        "1e-4 inside the dino_*.py runners).")
    p.add_argument("--lr-proj", type=float, default=None,
                   help="DINOv2-CL / DINOv2-CLIP projection-head LR "
                        "(default 1e-2 inside the runners).")
    p.add_argument("--early-stopping-patience", type=int, default=None,
                   help="patience (in epochs) for early stopping. Set to 0 "
                        "to disable. Default depends on the method (typically 2).")
    p.add_argument("--corruption", default=None,
                   choices=["gaussian_noise", "gaussian_blur",
                            "jpeg", "contrast", "brightness"],
                   help="E7 corruption robustness: apply this corruption to "
                        "every loaded eval-set image before passing to the "
                        "model. Has no effect during training.")
    p.add_argument("--severity", type=int, default=None,
                   choices=[1, 2, 3, 4, 5],
                   help="E7 corruption severity (1=mild .. 5=heavy). Required "
                        "with --corruption.")
    p.add_argument("--from-checkpoint", default=None,
                   help="Path to a prior run_dir whose trained model "
                        "should be loaded and re-evaluated on the new "
                        "--dataset-eval (eval-only mode; skips training). "
                        "The method's eval_<method>(config) is called "
                        "instead of train_<method>(config).")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--force", action="store_true",
                   help="ignore done.flag and re-run")
    p.add_argument("--dry-run", action="store_true",
                   help="print resolved config and exit")
    p.add_argument("--allow-degenerate", action="store_true",
                   help="allow CL+CE with batch_size<4 (CE-only mode)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Compose the per-run config dict from yaml + CLI flags."""
    cfg: dict[str, Any] = {}
    if args.from_config:
        registry = _load_yaml()
        if args.from_config not in registry:
            raise SystemExit(
                f"Unknown config id {args.from_config!r}; "
                f"see {REGISTRY_PATH}"
            )
        cfg.update(registry[args.from_config])
        cfg.setdefault("exp_id", args.from_config)
        # The yaml `seeds: [...]` list is for the sweep; a single dispatch
        # uses `--seed` as the active value.
        cfg.pop("seeds", None)

    # CLI overrides yaml for any explicitly set value.
    overrides = {
        "exp_id": args.exp_id, "method": args.method, "task": args.task,
        "seed": args.seed, "data_fraction": args.data_fraction,
        "dataset_train": args.dataset_train, "dataset_eval": args.dataset_eval,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "grad_accum": args.grad_accum, "lr": args.lr,
        "lambda_contrastive": args.lambda_contrastive,
        "temperature": args.temperature,
        "gc_batch_size": args.gc_batch_size,
        "variant": args.variant,
        "lr_encoder": args.lr_encoder,
        "lr_proj": args.lr_proj,
        "early_stopping_patience": args.early_stopping_patience,
        "corruption": args.corruption,
        "severity": args.severity,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if args.allow_degenerate:
        cfg["allow_degenerate"] = True
    if args.from_checkpoint:
        cfg["from_checkpoint"] = args.from_checkpoint

    if "exp_id" not in cfg:
        raise SystemExit("--exp-id (or --from-config) is required")
    if "method" not in cfg:
        raise SystemExit("--method (or a --from-config that names one) is required")
    if "dataset_eval" not in cfg or cfg["dataset_eval"] is None:
        raise SystemExit("--dataset-eval is required")

    cfg["seed"] = int(cfg.get("seed", 42))
    return cfg


def make_run_dir(cfg: dict, results_root: Path) -> Path:
    """Disk layout: results/{exp_id}_{method}[_{task}][_eval-...][_f{pct}][_holdout-...]_seed{seed}/

    Includes whatever fields are needed to disambiguate the run from
    sibling configs (multiple methods at the same exp_id, different eval
    splits at the same exp_id like the two E0 entries, low-data fractions,
    held-out classes).
    """
    parts = [str(cfg["exp_id"]), str(cfg["method"])]
    task = str(cfg.get("task", "8class"))
    if task != "8class":
        parts.append(task)
    eval_repo = cfg.get("dataset_eval")
    if eval_repo:
        eval_short = (
            str(eval_repo).rsplit("/", 1)[-1].replace("_dataset", "")
        )
        parts.append(f"eval-{eval_short}")
    df = cfg.get("data_fraction", 1.0)
    if df is not None and float(df) < 1.0:
        parts.append(f"f{int(round(100*float(df)))}pct")
    holdout = cfg.get("holdout_class")
    if holdout:
        parts.append(f"holdout-{holdout}")
    corr = cfg.get("corruption")
    sev = cfg.get("severity")
    if corr and sev:
        parts.append(f"corr-{corr}-sev{int(sev)}")
    parts.append(f"seed{int(cfg['seed'])}")
    return results_root / "_".join(parts)


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    results_root = Path(args.results_dir)
    run_dir = make_run_dir(cfg, results_root)
    cfg["run_dir"] = str(run_dir)

    if args.dry_run:
        print(f"run_dir = {run_dir}")
        for k, v in sorted(cfg.items()):
            print(f"  {k} = {v!r}")
        return 0

    done_flag = run_dir / "done.flag"
    if done_flag.exists() and not args.force:
        print(f"[skip] {run_dir.name}: done.flag present "
              f"(use --force to re-run)")
        return 0

    eval_only = bool(cfg.get("from_checkpoint"))
    method_fn = _import_method(cfg["method"], eval_only=eval_only)
    if eval_only:
        print(f"[eval-only] loading from {cfg['from_checkpoint']}")

    started = time.time()
    try:
        method_fn(cfg)
    except Exception as e:
        # Surface the failure in the run dir so the orchestrator can see it.
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "FAILED.txt", "w") as f:
            f.write(f"{type(e).__name__}: {e}\n")
        raise
    elapsed = time.time() - started
    print(f"[ok] {run_dir.name}: completed in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
