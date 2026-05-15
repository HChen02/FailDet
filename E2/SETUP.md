# E2 — Setup on a fresh target server

This bundle is self-contained except for two things you must install: the
Python deps in `requirements.txt`, and (only if you want the `da3_dino_*`
methods) the Depth-Anything-3 package.

## 1. Python environment

Python 3.10 strongly recommended. The project's existing env was built
against torch 2.10 + CUDA 12.8.

### Option A — conda (recommended)

```bash
conda create -n faildet python=3.10 -y
conda activate faildet
pip install -r requirements.txt
```

### Option B — venv

```bash
python3.10 -m venv ~/faildet-env
source ~/faildet-env/bin/activate
pip install -r requirements.txt
```

After install, sanity-check with:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import unsloth; print('unsloth ok')"
```

## 2. Optional — install Depth-Anything-3 for the `da3_dino_*` methods

The two depth-based methods (`da3_dino_ce`, `da3_dino_cl`) require the
official DA3 source. We install it `--no-deps` to avoid clobbering our
numpy version (DA3's requirements pin `numpy<2`, which would break
transformers).

```bash
# Anywhere outside the E2/ folder:
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git ~/Depth-Anything-3
pip install --no-deps -e ~/Depth-Anything-3
```

Verify:
```bash
python -c "from depth_anything_3.model.dinov2.dinov2 import DinoV2; print('DA3 backbone ok')"
```

If you do not need DA3 methods, **skip this entire step** and drop them
from the sweep with:

```bash
METHODS="dino_ce dino_ce_attn dino_cl dino_cl_attn dino_clip" \
    nohup ./run_e2_data_efficiency.sh > results/logs/e2_master.log 2>&1 < /dev/null &
```

## 3. Pre-built depth cache (only if running DA3 methods)

The `da3_dino_*` methods read fp16 depth maps from
`data/depth_cache/<dataset>/<sample_idx>/frame_<i>.npy`. These are NOT
shipped in the bundle (~14 GB for RLBench alone). The preprocessor script
`experiments/preprocess_da3.py` is included; generate the cache before
launching the sweep:

```bash
cd E2/
mkdir -p results/logs
# Build depth cache for the splits E2 needs (RLBench train + test).
# Takes ~10 min on an A100 80 GB.
python experiments/preprocess_da3.py --splits rlbench_train rlbench_test
```

If you'd rather skip DA3 entirely, drop those methods from the sweep:

```bash
METHODS="dino_ce dino_ce_attn dino_cl dino_cl_attn dino_clip" \
    nohup ./run_e2_parallel.sh > results/logs/e2_master.log 2>&1 < /dev/null &
```

## 4. GPU

- NVIDIA GPU with bf16 support (RTX 5090 / A100 / H100).
- Peak VRAM per cell: DINOv2 + DA3 methods ~3 GB; VLM methods ~14 GB.
- Multi-GPU: not needed — every cell is a single-GPU job.

## 5. HuggingFace cache

First run downloads:
- `paulpacaud/rlbenchfail_train_dataset` (~1 GB tarball)
- `paulpacaud/rlbenchfail_test_dataset` (~80 MB)
- `facebook/dinov2-large` (~1.2 GB)
- `depth-anything/DA3-BASE` (~500 MB, only if DA3 methods enabled)

Set `HF_HOME=/path/to/cache` if you need a custom location.

## 6. Run

```bash
cd E2/
chmod +x run_e2_data_efficiency.sh
export PYTHON=$(which python)   # point the launcher at the right interpreter
nohup ./run_e2_data_efficiency.sh \
    > results/logs/e2_master.log 2>&1 < /dev/null &
echo $! > results/logs/e2.pid
```

Monitor:
```bash
tail -f results/logs/e2_master.log
ls results/E2_*/done.flag | wc -l   # target 126 with default scope
```

## 7. Output

Per cell: `results/E2_<method>_binary_eval-rlbenchfail_test_f<pct>pct_seed<seed>/metrics.json`

The 100% cells omit the `f<pct>pct` suffix:
`results/E2_<method>_binary_eval-rlbenchfail_test_seed<seed>/metrics.json`

Each metrics.json contains the full predictions/ground_truths arrays, F1,
accuracy, per-class breakdowns, train history, and a `head.pt` checkpoint
in the same directory.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: depth_anything_3` | Either install per step 2, or drop `da3_dino_ce`, `da3_dino_cl` from `METHODS` |
| `pyarrow.lib.ArrowInvalid` parsing BDV2/UR5-train JSONL | Not an issue for E2 (RLBench only). Project's `data/dataset.py` falls back to pandas automatically. |
| `RuntimeError: CUBLAS_STATUS_INTERNAL_ERROR` mid-training | Transient cublas error from VRAM contention. Re-run; `done.flag` resumes from where it stopped. |
| `Cache miss: data/depth_cache/...` | DA3 method invoked without pre-built depth cache. See step 3. |
| Sweep killed mid-cell | `done.flag` was not written for that cell; re-launching the script will redo it. |
