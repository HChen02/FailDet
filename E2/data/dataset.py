"""
PyTorch Dataset for the Guardian failure-detection benchmark.

Each Guardian split lives in its own HF dataset repo. The repo holds:
  * a metadata table (parquet / JSONL) with one row per episode-step. The
    `images` column is a list of *relative* paths like
    'records/<taskvar>/<episode>/<step>_img_viewpoint_<view>.png'.
  * a `records.tar.gz` archive that bundles the actual PNGs.

Two complications this module handles:

1. **Heterogeneous label vocabulary.** The 9 Guardian splits use 16 distinct
   raw `failure_mode` strings (RLBench uses 'no_close', 'translation', 'slip',
   ...; UR5 train uses verbose labels like 'imprecise grasping/pushing';
   UR5 val uses snake_case like 'no_grasp', 'translation_object', ...).
   We collapse them into a single 8-class scheme via UNIFIED_LABELS.

2. **BridgeDataV2's `failure_reason` column type oscillates** between int
   (e.g. 0) and string in the underlying JSONL, which trips pyarrow's
   schema inference. We force the schema with explicit Features, and fall
   back to a pandas JSONL load if that still fails.
"""

from __future__ import annotations

import json
import tarfile
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from datasets import Dataset as HFDataset
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

# Where the on-disk image cache lives by default. See
# experiments/run_phase5.py for the build script that populates it.
_DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parent / "image_cache"
)


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

UNIFIED_LABELS: dict[str, str] = {
    # Class 0: success
    "ground_truth": "success",

    # Class 1: no_grasp
    "no_close": "no_grasp",
    "no_grasp": "no_grasp",
    "no gripper close": "no_grasp",
    "no progress": "no_grasp",

    # Class 2: slip
    "slip": "slip",

    # Class 3: translation
    "translation": "translation",
    "translation_object": "translation",
    "translation_target": "translation",
    "imprecise grasping/pushing": "translation",

    # Class 4: rotation
    "rotation": "rotation",

    # Class 5: wrong_object
    "wrong_object": "wrong_object",
    "wrong object manipulated": "wrong_object",

    # Class 6: wrong_sequence
    "wrong_sequence": "wrong_sequence",

    # Class 7: wrong_state
    "wrong_target": "wrong_state",
    "wrong object state or placement": "wrong_state",
}

LABEL_TO_ID: dict[str, int] = {
    "success": 0,
    "no_grasp": 1,
    "slip": 2,
    "translation": 3,
    "rotation": 4,
    "wrong_object": 5,
    "wrong_sequence": 6,
    "wrong_state": 7,
}
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABEL_TO_ID)

UNKNOWN_LABEL = "unknown"
UNKNOWN_ID = -1


def normalize_label(raw: str) -> str:
    """Map a raw `failure_mode` string to the unified 8-class vocabulary.

    Emits a warning and returns "unknown" if `raw` is not registered, so
    that pipelines never crash on a stray label — the validator below is
    responsible for catching such cases at dataset-prep time.
    """
    if raw in UNIFIED_LABELS:
        return UNIFIED_LABELS[raw]
    warnings.warn(
        f"Unrecognized raw failure_mode: {raw!r}. Mapping to {UNKNOWN_LABEL!r}.",
        stacklevel=2,
    )
    return UNKNOWN_LABEL


# ---------------------------------------------------------------------------
# Metadata loading (with pandas fallback for BDV2's mixed-type column)
# ---------------------------------------------------------------------------

# Splits whose JSONL has a column (`failure_reason`) whose JSON type
# oscillates row-to-row between number and string. `load_dataset` (via
# pyarrow) always fails on these — and the failure path spams "Generating
# train split: 0 examples [00:00] Failed to load JSON ..." into the log.
# Skip the doomed attempt entirely and load via pandas right away.
_KNOWN_TYPE_UNSTABLE_REPOS = {
    "paulpacaud/bdv2fail_train_dataset",
    "paulpacaud/bdv2fail_val_dataset",
    "paulpacaud/bdv2fail_test_dataset",
    "paulpacaud/ur5fail_train_dataset",
}


def _load_via_pandas(repo_id: str, *, expected: bool = False) -> HFDataset:
    """Pandas-based JSONL loader that coerces the failure_reason column to
    string. Used for the known-broken splits AND as a fallback if any
    other split happens to fail load_dataset.
    """
    jsonl_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="metadata_execution.jsonl",
    )
    df = pd.read_json(jsonl_path, lines=True)
    if "failure_reason" in df.columns:
        df["failure_reason"] = df["failure_reason"].astype("string").fillna("")
    else:
        df["failure_reason"] = ""
    defaults: dict[str, Any] = {
        "task_instruction": "", "detailed_subtask_name": "",
        "failure_mode": "", "execution_reward": 0,
        "reward": 0, "planning_reward": 0, "episode_id": 0,
        "images": [], "plan": [], "taskvar": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = [default] * len(df) if isinstance(default, list) else default
    if not expected:
        warnings.warn(
            f"plain load_dataset failed for {repo_id}; recovered via pandas "
            f"(n={len(df)}).", stacklevel=2,
        )
    return HFDataset.from_pandas(df, preserve_index=False)


def _load_metadata_table(repo_id: str, split: str = "train") -> HFDataset:
    """Return a HuggingFace Dataset for the given Guardian split.

    Strategy: known-broken splits (BDV2-all, ur5fail_train) skip
    load_dataset and go straight to pandas — silently. Other splits try
    load_dataset first and fall back to pandas if pyarrow chokes.
    """
    if repo_id in _KNOWN_TYPE_UNSTABLE_REPOS:
        return _load_via_pandas(repo_id, expected=True)
    try:
        return load_dataset(repo_id, split=split)
    except Exception:
        return _load_via_pandas(repo_id, expected=False)


def resolve_records_path(records_root: Path, rel_path: str) -> Optional[Path]:
    """Map a Guardian metadata image path to an actual file under
    ``records_root``, returning None if no candidate exists.

    RLBench and UR5 metadata paths begin with ``records/...`` and resolve
    directly. BDV2 metadata paths are prefixed with
    ``data/failure_forge/data/<split>/records/...``; the tarball still
    extracts to ``records/...``, so we strip everything before the first
    ``records/`` segment and re-anchor under ``records_root``.
    """
    primary = records_root / rel_path
    if primary.exists():
        return primary
    marker = "records/"
    idx = rel_path.find(marker)
    if idx > 0:
        candidate = records_root / rel_path[idx:]
        if candidate.exists():
            return candidate
    return None


def _ensure_records_extracted(repo_id: str) -> Path:
    """Download `records.tar.gz` for `repo_id` and extract once."""
    archive_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="records.tar.gz",
        )
    )
    extract_root = archive_path.parent / "extracted_records"
    sentinel = extract_root / ".extracted_ok"
    if not sentinel.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_root)
        sentinel.touch()
    return extract_root


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class GuardianDataset(Dataset):
    """One Guardian HF split, with normalized 8-class labels.

    __getitem__ returns:
        {
          "images":                list[PIL.Image] (typically length 6),
          "failure_mode_raw":      str  (original, not normalized),
          "failure_mode":          str  (normalized: success/no_grasp/.../wrong_state/unknown),
          "failure_label":         int  (0..7, or -1 for unknown),
          "binary_label":          int  (0=success, 1=failure),
          "task_instruction":      str,
          "detailed_subtask_name": str,
          "execution_reward":      int,
        }

    Args:
        repo_id: HF dataset id, e.g. "paulpacaud/ur5fail_val_dataset".
        split: HF split name (Guardian repos all ship a single "train" split).
        load_images: when False, skips PIL.Image.open for fast schema/label-only
            iteration (useful for class-balance checks).
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        load_images: bool = True,
        *,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
        corruption_name: Optional[str] = None,
        severity: Optional[int] = None,
    ) -> None:
        self.repo_id = repo_id
        self.split = split
        self.load_images = load_images
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        # E7 corruption robustness: when set, every loaded image is passed
        # through experiments.corruptions.apply before being returned.
        self.corruption_name = corruption_name
        self.severity = int(severity) if severity is not None else None

        self._table = _load_metadata_table(repo_id, split)
        # Tarball is only needed when load_images=True, but extracting is cheap
        # and idempotent so we always do it for consistency.
        self._records_root = (
            _ensure_records_extracted(repo_id) if load_images else None
        )
        # Optional on-disk image cache (built by experiments/run_phase5.py).
        # Falls back transparently to the tarball-extracted layout if the
        # cache index is absent.
        self._cache_index: Optional[dict[str, list[str]]] = (
            self._maybe_load_cache_index() if use_cache and load_images
            else None
        )

    def __len__(self) -> int:
        return len(self._table)

    @staticmethod
    def normalize_label(raw: str) -> str:
        return normalize_label(raw)

    def split_short_name(self) -> str:
        """Short identifier used by the on-disk cache layout (e.g.
        'paulpacaud/ur5fail_val_dataset' -> 'ur5fail_val')."""
        return self.repo_id.split("/")[-1].replace("_dataset", "")

    def _maybe_load_cache_index(self) -> Optional[dict[str, list[str]]]:
        """Load `cache_dir/<split_name>/index.json` if it exists.

        The index maps a string-form sample index to a list of 6 absolute
        image paths. Missing or malformed index → cache disabled for
        this instance and we fall back to the tarball-extracted layout.
        """
        idx_path = self.cache_dir / self.split_short_name() / "index.json"
        if not idx_path.exists():
            return None
        try:
            with open(idx_path) as f:
                idx = json.load(f)
            if not isinstance(idx, dict):
                return None
            return idx
        except Exception as e:
            warnings.warn(
                f"failed to load image cache index at {idx_path}: {e!r}; "
                "falling back to tarball-extracted layout",
                stacklevel=2,
            )
            return None

    def _resolve_image(self, rel_path: str) -> Image.Image:
        assert self._records_root is not None
        full_path = self._resolve_image_path(rel_path)
        if full_path is None:
            raise FileNotFoundError(
                f"Image referenced by {self.repo_id} not found after extraction: "
                f"{self._records_root / rel_path}"
            )
        return Image.open(full_path).convert("RGB")

    def _resolve_image_path(self, rel_path: str) -> Optional[Path]:
        """Thin wrapper around the module-level resolver."""
        assert self._records_root is not None
        return resolve_records_path(self._records_root, rel_path)

    def _load_images_for_idx(self, idx: int, rel_paths: list[str]) -> list[Image.Image]:
        """Use the on-disk cache when present, fall back to the tarball-
        extracted layout otherwise. Cache lookup is keyed by `str(idx)`
        because JSON dict keys are strings."""
        if self._cache_index is not None:
            cached = self._cache_index.get(str(idx))
            if cached is not None and len(cached) == len(rel_paths):
                try:
                    return [Image.open(p).convert("RGB") for p in cached]
                except Exception as e:
                    warnings.warn(
                        f"image cache miss at idx={idx} ({e!r}); "
                        "falling back to tarball-extracted layout",
                        stacklevel=2,
                    )
        return [self._resolve_image(p) for p in rel_paths]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._table[idx]

        raw = row["failure_mode"]
        norm = self.normalize_label(raw)
        label_id = LABEL_TO_ID.get(norm, UNKNOWN_ID)
        binary = 0 if norm == "success" else 1

        images = (
            self._load_images_for_idx(idx, row["images"])
            if self.load_images
            else []
        )

        # E7 corruption hook (no-op unless corruption_name + severity were
        # passed to the constructor). Per-sample-index rng_seed keeps the
        # corruption deterministic across runs.
        if self.corruption_name and self.severity and images:
            from experiments.corruptions import apply as _corrupt
            images = [_corrupt(im, self.corruption_name, self.severity,
                               rng_seed=int(idx))
                      for im in images]

        return {
            "images": images,
            "failure_mode_raw": raw,
            "failure_mode": norm,
            "failure_label": label_id,
            "binary_label": binary,
            "task_instruction": row["task_instruction"],
            "detailed_subtask_name": row["detailed_subtask_name"],
            "execution_reward": int(row["execution_reward"]),
        }


# ---------------------------------------------------------------------------
# Validation / smoke test
# ---------------------------------------------------------------------------

ALL_GUARDIAN_SPLITS: list[str] = [
    "paulpacaud/rlbenchfail_train_dataset",
    "paulpacaud/rlbenchfail_val_dataset",
    "paulpacaud/rlbenchfail_test_dataset",
    "paulpacaud/bdv2fail_train_dataset",
    "paulpacaud/bdv2fail_val_dataset",
    "paulpacaud/bdv2fail_test_dataset",
    "paulpacaud/ur5fail_train_dataset",
    "paulpacaud/ur5fail_val_dataset",
    "paulpacaud/ur5fail_test_dataset",
]


def _validate_all_splits() -> int:
    """Load every Guardian split (metadata only — no tarballs), normalize
    every failure_mode, and report coverage.

    Returns: count of unknown raw labels encountered (0 = full coverage).
    """
    from collections import Counter

    print("=" * 80)
    print(" Validation: normalize every failure_mode in all 9 Guardian splits")
    print("=" * 80)

    global_norm_counts: Counter = Counter()
    all_unknowns: dict[str, int] = {}
    per_split_summaries: list[tuple[str, int, set, set, int]] = []
    failed_loads: list[tuple[str, str]] = []

    for repo in ALL_GUARDIAN_SPLITS:
        try:
            table = _load_metadata_table(repo)
        except Exception as e:
            failed_loads.append((repo, f"{type(e).__name__}: {e}"))
            print(f"\n[{repo}] LOAD FAILED: {type(e).__name__}: {e}")
            continue

        n = len(table)
        raw_set: set = set()
        norm_set: set = set()
        unknown_count = 0
        for raw in table["failure_mode"]:
            raw_set.add(raw)
            if raw in UNIFIED_LABELS:
                norm = UNIFIED_LABELS[raw]
            else:
                norm = UNKNOWN_LABEL
                unknown_count += 1
                all_unknowns[raw] = all_unknowns.get(raw, 0) + 1
            norm_set.add(norm)
            global_norm_counts[norm] += 1
        per_split_summaries.append((repo, n, raw_set, norm_set, unknown_count))

        print(f"\n[{repo}]  n={n}")
        print(f"  raw labels ({len(raw_set)}):  {sorted(raw_set)}")
        print(f"  norm labels ({len(norm_set)}): {sorted(norm_set)}")
        if unknown_count:
            print(f"  ** {unknown_count} samples mapped to 'unknown' **")

    # ---- Summary table
    print("\n" + "=" * 80)
    print(" Per-split summary")
    print("=" * 80)
    print(f"{'split':<48} {'n':>7} {'raw':>5} {'norm':>5} {'unknown':>8}")
    for repo, n, raw, norm, unk in per_split_summaries:
        print(f"{repo:<48} {n:>7} {len(raw):>5} {len(norm):>5} {unk:>8}")

    # ---- Global counts
    total = sum(global_norm_counts.values())
    print("\n" + "=" * 80)
    print(f" Global normalized label distribution (n={total})")
    print("=" * 80)
    for label_id in range(NUM_CLASSES):
        name = ID_TO_LABEL[label_id]
        c = global_norm_counts.get(name, 0)
        pct = (100.0 * c / total) if total else 0.0
        print(f"  {label_id} {name:<16} {c:>7} ({pct:5.2f}%)")
    if global_norm_counts.get(UNKNOWN_LABEL, 0):
        c = global_norm_counts[UNKNOWN_LABEL]
        pct = 100.0 * c / total
        print(f"  - {UNKNOWN_LABEL:<16} {c:>7} ({pct:5.2f}%)")

    # ---- Unknown / failed report
    print("\n" + "=" * 80)
    print(" Issues")
    print("=" * 80)
    if failed_loads:
        print(f"  Splits that failed to load: {len(failed_loads)}")
        for repo, err in failed_loads:
            print(f"    - {repo}: {err}")
    else:
        print("  All 9 splits loaded.")

    if all_unknowns:
        print(f"  Raw labels NOT covered by UNIFIED_LABELS: {len(all_unknowns)}")
        for k, c in sorted(all_unknowns.items(), key=lambda kv: -kv[1]):
            print(f"    - {k!r}: {c}")
    else:
        print("  All raw labels covered by UNIFIED_LABELS.")

    return sum(all_unknowns.values())


def _smoke_test_ur5_val() -> None:
    """Quick end-to-end check on ur5fail_val (loads tarball + decodes images)."""
    print("\n" + "=" * 80)
    print(" Smoke test: GuardianDataset('paulpacaud/ur5fail_val_dataset')")
    print("=" * 80)
    ds = GuardianDataset("paulpacaud/ur5fail_val_dataset")
    print(f"len(ds) = {len(ds)}")
    sample = ds[0]
    print(f"keys           : {list(sample.keys())}")
    print(f"num images     : {len(sample['images'])} (size={sample['images'][0].size})")
    print(f"failure_mode_raw: {sample['failure_mode_raw']!r}")
    print(f"failure_mode    : {sample['failure_mode']!r}")
    print(f"failure_label   : {sample['failure_label']}")
    print(f"binary_label    : {sample['binary_label']}")
    print(f"task_instruction: {sample['task_instruction']}")
    print(f"detailed_subtask_name: {sample['detailed_subtask_name']}")
    print(f"execution_reward: {sample['execution_reward']}")


if __name__ == "__main__":
    n_unknown = _validate_all_splits()
    _smoke_test_ur5_val()
    if n_unknown:
        raise SystemExit(
            f"\nFAIL: {n_unknown} sample(s) had raw failure_mode strings outside "
            f"UNIFIED_LABELS. Extend the mapping before training."
        )
    print("\nOK: all raw labels covered by UNIFIED_LABELS.")
