"""Method-specific training functions for the experiment-runner dispatch.

Each method exposes a single entry point:

    train_<method>(config: dict) -> dict

where `config` is the resolved per-run configuration (built by
`experiments/run_experiment.py` from CLI args + configs/experiments.yaml)
and the return value is the metrics dict that will be written to
`{run_dir}/metrics.json`.

Methods (FINAL 4-method VLM line-up + DINOv2 baselines + zero-shot):
    sft           — supervised fine-tuning (Unsloth + TRL SFTTrainer; CE)
    cl_embed      — full-VLM last-token contrastive (Unsloth + QLoRA;
                    supervised InfoNCE + nearest-centroid eval)
    cl_llm        — yes/no token-logit contrastive (binary only;
                    asymmetric InfoNCE on yes_logit - no_logit)
    cl_ce         — joint CE + supervised InfoNCE on [EOS] (CL+CE)
    dino_ce       — frozen DINOv2-L + mean-pool + CE classifier head
    dino_ce_attn  — frozen DINOv2-L + attention-pool + CE classifier head
    dino_cl       — frozen DINOv2-L + mean-pool + supervised InfoNCE
    dino_cl_attn  — frozen DINOv2-L + attention-pool + supervised InfoNCE
    dino_clip     — frozen DINOv2-L + frozen MiniLM + CLIP-style InfoNCE
    zero_shot     — eval-only baseline (no training)

CL-FT (vision-only LoRA + InfoNCE) was archived as a documented negative
result after the 2026-05 sweep showed loss flat at log(B) regardless of
class-balanced sampling. Its v6 numbers live in
results/E1_cl_ft_*_seed42 and are referenced from the paper.
"""
