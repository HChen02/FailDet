#!/usr/bin/env bash
# =============================================================================
#  run_e2_data_efficiency.sh
# =============================================================================
#  E2 — Data Efficiency Sweep on RLBench-Fail (binary task)
#
#  Hypothesis:
#    Contrastive methods (CL-Embed, DINOv2-CL, etc.) reach competitive F1
#    with less training data than supervised cross-entropy methods (SFT,
#    DINOv2-CE).  Curve: F1 vs % of training data.
#
#  Sweep design:
#    fractions × methods × seeds × {train + eval} cells.
#    fractions: 1, 5, 10, 25, 50, 100 percent of RLBench-Fail train.
#    100% reuses the existing E1 binary checkpoints if present, otherwise
#    retrains.  Smaller fractions use the pre-computed stratified splits
#    in data/low_data_splits.json (same per-seed splits as the literature
#    baselines).
#
#  Output:
#    results/E2_<method>_binary_eval-rlbenchfail_test_f<pct>pct_seed<seed>/
#      metrics.json, head.pt, done.flag
#
#  Resumable:
#    Each cell drops a done.flag. Re-running this script skips finished
#    cells. Safe to interrupt and restart.
#
# -----------------------------------------------------------------------------
#  PREREQUISITES (on the target server)
# -----------------------------------------------------------------------------
#  1. The project repository is cloned and CWD-able. This script auto-detects
#     the project root from its own location; no hardcoded paths.
#  2. A conda / venv environment with the project's training stack:
#       torch >= 2.10, transformers, datasets, peft, unsloth, bitsandbytes,
#       accelerate, scikit-learn, einops, omegaconf, addict, plyfile,
#       moviepy, decorator, proglog
#     The `depth-anything-3` package (editable install) is needed for the
#     da3_dino_* methods; skip those from METHODS if you don't have it.
#  3. NVIDIA GPU with bf16 support (RTX 5090 / A100 / H100). Approximate
#     VRAM peaks: DINOv2 + DA3 methods ~3 GB, VLM methods ~14 GB.
#  4. HuggingFace cache directory writable. First run downloads the
#     `paulpacaud/rlbenchfail_*` datasets + `facebook/dinov2-large` +
#     `depth-anything/DA3-BASE` (~3 GB total).
#  5. For da3_dino_* methods the depth_cache must exist. If absent the
#     methods themselves will raise — run experiments/preprocess_da3.py
#     first OR drop them from METHODS.
#
# -----------------------------------------------------------------------------
#  HOW TO RUN (after `git clone <repo>` on the target server)
# -----------------------------------------------------------------------------
#  Default sweep (fast methods only, ~15h on a single 5090):
#
#    cd <repo_root>
#    chmod +x run_e2_data_efficiency.sh
#    nohup ./run_e2_data_efficiency.sh \
#        > results/logs/e2_master.log 2>&1 < /dev/null &
#    echo $! > results/logs/e2.pid
#
#  Override defaults via env vars (e.g. add the VLM methods):
#
#    METHODS="sft cl_embed cl_llm cl_ce dino_ce dino_cl da3_dino_ce da3_dino_cl" \
#    FRACTIONS="1 5 10 25 50" \
#    SEEDS="42 123 456" \
#    nohup ./run_e2_data_efficiency.sh > e2.log 2>&1 < /dev/null &
#
#  Different python interpreter or repo location:
#
#    PYTHON=/path/to/python PROJECT_ROOT=/abs/path/to/repo \
#    nohup ./run_e2_data_efficiency.sh > e2.log 2>&1 < /dev/null &
#
# -----------------------------------------------------------------------------

set -uo pipefail

# Project root = directory containing this script.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_ROOT}"

# Python interpreter: respect $PYTHON env var, else fall back to `python`.
PYTHON="${PYTHON:-python}"
if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} cannot import torch. Set PYTHON env var to your env's python."
    exit 1
fi

# Unsloth quirks (required for VLM methods, harmless otherwise).
export UNSLOTH_COMPILE_DISABLE=1
export UNSLOTH_DISABLE_FAST_GENERATION=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

# ----------------------------------------------------------------------
# Configurable knobs
# ----------------------------------------------------------------------
# Default methods: 7 "fast" methods (DINOv2 + DA3). VLM methods (sft,
# cl_embed, cl_llm, cl_ce) are excluded by default because each cell
# trains for ~3 h on a 5090 — 4 VLM methods × 6 fractions × 3 seeds =
# 72 cells × ~3 h = ~9 days. Add them via METHODS=... env var only if
# you have multi-day cluster time.
DEFAULT_METHODS="dino_ce dino_ce_attn dino_cl dino_cl_attn dino_clip da3_dino_ce da3_dino_cl"
read -r -a METHODS <<< "${METHODS:-${DEFAULT_METHODS}}"

# Fractions: 1, 5, 10, 25, 50, 100 (percent). 100 reuses E1 if available.
read -r -a FRACTIONS <<< "${FRACTIONS:-1 5 10 25 50 100}"

# Seeds.
read -r -a SEEDS <<< "${SEEDS:-42 123 456}"

# Task + datasets — binary on RLBench-Fail.
TASK="binary"
DATASET_TRAIN="paulpacaud/rlbenchfail_train_dataset"
DATASET_EVAL="paulpacaud/rlbenchfail_test_dataset"

# Per-method training defaults (mirrors v7 plan).
ce_epochs=30;  ce_lr=1e-3            # SFT, dino_ce, dino_ce_attn, da3_dino_ce
cl_epochs=30;  cl_lr=1e-2            # CL-Embed, CL-LLM, CL+CE, dino_cl*, dino_clip, da3_dino_cl
vlm_bs=4;      vlm_ga=4              # SFT, CL-Embed, CL-LLM, CL+CE
dino_bs=32                            # DINOv2 and DA3 methods
BATCH_SIZE_DEFAULT=32

# Total cells = fractions × methods × seeds.
TOTAL=$(( ${#FRACTIONS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))

# Output dirs.
RESULTS_ROOT="${PROJECT_ROOT}/results"
LOG_DIR="${RESULTS_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
already_done() { [[ -f "$1/done.flag" ]]; }

# Resolve epochs + LR for the given method.
hp_for() {
    case "$1" in
        sft|dino_ce|dino_ce_attn|da3_dino_ce|cl_llm)
            echo "${ce_epochs} ${ce_lr}" ;;
        cl_embed|cl_ce|dino_cl|dino_cl_attn|dino_clip|da3_dino_cl)
            echo "${cl_epochs} ${cl_lr}" ;;
        *)
            echo "${ce_epochs} ${ce_lr}" ;;
    esac
}

bs_for() {
    case "$1" in
        sft|cl_embed|cl_llm|cl_ce) echo "${vlm_bs} ${vlm_ga}" ;;
        *)                          echo "${dino_bs} 1" ;;
    esac
}

# run_dir = results/E2_<method>_binary_eval-<eval_short>_f<pct>pct_seed<seed>
#  (matches the convention in run_experiment.py::make_run_dir)
rd_for() {
    local method="$1"; local frac="$2"; local seed="$3"
    local eval_short; eval_short="${DATASET_EVAL##*/}"
    eval_short="${eval_short%_dataset}"
    if [[ "${frac}" == "100" ]]; then
        # 100% has no fX_pct suffix (data_fraction=1.0 default).
        echo "${RESULTS_ROOT}/E2_${method}_${TASK}_eval-${eval_short}_seed${seed}"
    else
        echo "${RESULTS_ROOT}/E2_${method}_${TASK}_eval-${eval_short}_f${frac}pct_seed${seed}"
    fi
}

run_cell() {
    local method="$1"; local frac="$2"; local seed="$3"
    local rd; rd=$(rd_for "${method}" "${frac}" "${seed}")
    if already_done "${rd}"; then
        echo "[skip] E2 ${method}/f${frac}pct/seed=${seed}"
        return 0
    fi
    local hp; read -r -a hp <<< "$(hp_for "${method}")"
    local bs_ga; read -r -a bs_ga <<< "$(bs_for "${method}")"
    local frac_dec
    # Convert integer percent → float (e.g. 1 → 0.01, 100 → 1.0).
    frac_dec=$(awk "BEGIN { printf \"%.4f\", ${frac} / 100.0 }")
    echo ""
    echo "================================================================"
    echo " TRAIN  E2 ${method}/f${frac}pct/seed=${seed}   $(date +%H:%M:%S)"
    echo " run_dir=${rd}"
    echo "================================================================"
    local t0; t0=$(date +%s)
    "${PYTHON}" experiments/run_experiment.py \
        --method "${method}" \
        --exp-id "E2" \
        --task "${TASK}" \
        --dataset-train "${DATASET_TRAIN}" \
        --dataset-eval  "${DATASET_EVAL}" \
        --seed "${seed}" \
        --data-fraction "${frac_dec}" \
        --epochs "${hp[0]}" \
        --lr "${hp[1]}" \
        --batch-size "${bs_ga[0]}" \
        --grad-accum "${bs_ga[1]}" \
        --early-stopping-patience 0
    local rc=$?
    local el=$(( $(date +%s) - t0 ))
    echo "[done] E2 ${method}/f${frac}pct/seed=${seed} rc=${rc} elapsed=${el}s"
}

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
T0=$(date +%s)
echo "================================================================"
echo " E2 — Data Efficiency Sweep on RLBench-Fail (binary)"
echo " project root: ${PROJECT_ROOT}"
echo " python:       ${PYTHON}"
echo " methods (${#METHODS[@]}): ${METHODS[*]}"
echo " fractions:    ${FRACTIONS[*]} (percent)"
echo " seeds:        ${SEEDS[*]}"
echo " total cells:  ${TOTAL}"
echo " started at $(date)"
echo "================================================================"

# Order: fractions outermost so partial completion still gives a full
# curve for the cheapest fractions first.
for FRAC in "${FRACTIONS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_cell "${METHOD}" "${FRAC}" "${SEED}"
        done
    done
done

ELAPSED=$(( $(date +%s) - T0 ))
echo ""
echo "================================================================"
echo " E2 sweep -- DONE"
echo " total runtime: ${ELAPSED}s ($((ELAPSED/3600))h $((ELAPSED%3600/60))m)"
echo " $(date)"
echo "================================================================"
