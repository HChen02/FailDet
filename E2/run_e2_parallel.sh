#!/usr/bin/env bash
# =============================================================================
#  run_e2_parallel.sh
# =============================================================================
#  Parallel version of run_e2_data_efficiency.sh — runs multiple cells
#  concurrently to saturate the GPU.
#
#  Same coverage:
#    fractions × methods × seeds × {train + eval}.
#    Defaults: 6 fractions × 7 methods × 3 seeds = 126 cells.
#
#  Concurrency:
#    PARALLEL env var (default 4) caps simultaneous jobs.
#    Per-cell VRAM peaks:
#      DA3 / DINOv2 methods: ~1.3 GB  → A100-80GB can fit 30+; cap on
#                                       I/O contention rather than VRAM.
#      VLM methods (sft / cl_*):  ~14 GB → A100-80GB fits 4-5 in parallel.
#    Each backgrounded cell writes to results/logs/parallel/<cell>.log.
#
#  Resumable: done.flag pattern. Restarts skip finished cells.
#
#  Usage (after `git clone` + cd E2):
#    chmod +x run_e2_parallel.sh
#    export PYTHON=$(which python)
#    mkdir -p results/logs
#    PARALLEL=8 nohup ./run_e2_parallel.sh \
#        > results/logs/e2_parallel_master.log 2>&1 < /dev/null &
#    echo $! > results/logs/e2_parallel.pid
#
#  Suggested PARALLEL by GPU:
#    A100 80 GB, fast methods only:        PARALLEL=8-12
#    A100 80 GB, includes VLM methods:     PARALLEL=4
#    RTX 5090 32 GB, fast methods only:    PARALLEL=4-6
#    RTX 5090 32 GB, includes VLM methods: PARALLEL=1-2
# =============================================================================

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python}"
if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} cannot import torch. Set PYTHON env var to your env's python."
    exit 1
fi

export UNSLOTH_COMPILE_DISABLE=1
export UNSLOTH_DISABLE_FAST_GENERATION=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

# ---- Concurrency ----
MAX_PARALLEL="${PARALLEL:-4}"

# ---- Scope ----
DEFAULT_METHODS="dino_ce dino_ce_attn dino_cl dino_cl_attn dino_clip da3_dino_ce da3_dino_cl"
read -r -a METHODS <<< "${METHODS:-${DEFAULT_METHODS}}"
read -r -a FRACTIONS <<< "${FRACTIONS:-1 5 10 25 50 100}"
read -r -a SEEDS <<< "${SEEDS:-42 123 456}"

TASK="binary"
DATASET_TRAIN="paulpacaud/rlbenchfail_train_dataset"
DATASET_EVAL="paulpacaud/rlbenchfail_test_dataset"

# Per-method LR.
lr_for() {
    case "$1" in
        sft|dino_ce|dino_ce_attn|da3_dino_ce|cl_llm) echo "1e-3" ;;
        cl_embed|cl_ce|dino_cl|dino_cl_attn|dino_clip|da3_dino_cl) echo "1e-2" ;;
        *) echo "1e-3" ;;
    esac
}
# Per-method batch size + grad accum.
bs_for() {
    case "$1" in
        sft|cl_embed|cl_llm|cl_ce) echo "4 4" ;;
        *)                          echo "32 1" ;;
    esac
}

EPOCHS=30
EARLY_STOP=0

TOTAL=$(( ${#FRACTIONS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))

RESULTS_ROOT="${PROJECT_ROOT}/results"
LOG_DIR="${RESULTS_ROOT}/logs"
PAR_LOG_DIR="${LOG_DIR}/parallel"
mkdir -p "${LOG_DIR}" "${PAR_LOG_DIR}"

already_done() { [[ -f "$1/done.flag" ]]; }

# Mirror run_experiment.py::make_run_dir for E2 cells.
rd_for() {
    local method="$1"; local frac="$2"; local seed="$3"
    local eval_short; eval_short="${DATASET_EVAL##*/}"; eval_short="${eval_short%_dataset}"
    if [[ "${frac}" == "100" ]]; then
        echo "${RESULTS_ROOT}/E2_${method}_${TASK}_eval-${eval_short}_seed${seed}"
    else
        echo "${RESULTS_ROOT}/E2_${method}_${TASK}_eval-${eval_short}_f${frac}pct_seed${seed}"
    fi
}

# ---- Bash job pool: cap at MAX_PARALLEL concurrent background jobs. ----
wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        wait -n 2>/dev/null || sleep 5
    done
}

spawn_cell() {
    local method="$1"; local frac="$2"; local seed="$3"
    local rd; rd=$(rd_for "${method}" "${frac}" "${seed}")
    if already_done "${rd}"; then
        echo "[skip] E2 ${method}/f${frac}pct/seed=${seed}"
        return 0
    fi
    wait_for_slot
    local lr; lr=$(lr_for "${method}")
    local bs_ga; read -r -a bs_ga <<< "$(bs_for "${method}")"
    local frac_dec; frac_dec=$(awk "BEGIN { printf \"%.4f\", ${frac} / 100.0 }")
    local log_file="${PAR_LOG_DIR}/E2_${method}_f${frac}pct_seed${seed}.log"
    echo "[spawn $(date +%H:%M:%S)] E2 ${method}/f${frac}pct/seed=${seed} lr=${lr} bs=${bs_ga[0]} → ${log_file}"
    (
        "${PYTHON}" experiments/run_experiment.py \
            --method "${method}" \
            --exp-id "E2" \
            --task "${TASK}" \
            --dataset-train "${DATASET_TRAIN}" \
            --dataset-eval  "${DATASET_EVAL}" \
            --seed "${seed}" \
            --data-fraction "${frac_dec}" \
            --epochs "${EPOCHS}" \
            --lr "${lr}" \
            --batch-size "${bs_ga[0]}" \
            --grad-accum "${bs_ga[1]}" \
            --early-stopping-patience "${EARLY_STOP}" \
            > "${log_file}" 2>&1
        local rc=$?
        echo "[done $(date +%H:%M:%S)] E2 ${method}/f${frac}pct/seed=${seed} rc=${rc}"
    ) &
}

T0=$(date +%s)
echo "================================================================"
echo " E2 — Parallel Data-Efficiency Sweep on RLBench-Fail (binary)"
echo " project root: ${PROJECT_ROOT}"
echo " python:       ${PYTHON}"
echo " methods (${#METHODS[@]}): ${METHODS[*]}"
echo " fractions:    ${FRACTIONS[*]} (percent)"
echo " seeds:        ${SEEDS[*]}"
echo " concurrency:  PARALLEL=${MAX_PARALLEL}"
echo " total cells:  ${TOTAL}"
echo " started at $(date)"
echo "================================================================"

# Order: fractions outermost (cheap-first), methods middle, seeds innermost.
for FRAC in "${FRACTIONS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            spawn_cell "${METHOD}" "${FRAC}" "${SEED}"
        done
    done
done

echo "[main] all cells spawned, waiting for the pool to drain ..."
wait

ELAPSED=$(( $(date +%s) - T0 ))
echo ""
echo "================================================================"
echo " E2 parallel sweep -- DONE"
echo " total runtime: ${ELAPSED}s ($((ELAPSED/3600))h $((ELAPSED%3600/60))m)"
echo " $(date)"
echo "================================================================"
