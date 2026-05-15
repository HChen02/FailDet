"""
Inspect the schema of the Guardian dataset before building the full Dataset class.

Downloads ONE small split (paulpacaud/ur5fail_val_dataset, ~180 samples) and prints:
  - Column names
  - Column dtypes
  - One full sample with all field types
  - How images are stored (PIL, paths, bytes, dict, etc.)
  - The list of unique failure_mode values across the full split

We deliberately avoid downloading all 9 splits until the schema is understood.
"""

from __future__ import annotations

import io
import sys
from collections import Counter
from typing import Any

from datasets import load_dataset


SMALLEST_SPLIT = "paulpacaud/ur5fail_val_dataset"


def describe_value(v: Any, max_len: int = 200) -> str:
    """Render a short, type-aware description of a single field value."""
    t = type(v).__name__
    if v is None:
        return f"<{t}> None"

    # PIL Image
    try:
        from PIL import Image as PILImage
        if isinstance(v, PILImage.Image):
            return f"<PIL.Image> mode={v.mode} size={v.size}"
    except ImportError:
        pass

    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        head = b[:8]
        return f"<{t}> len={len(b)} head_hex={head.hex()}"

    if isinstance(v, dict):
        return f"<dict> keys={list(v.keys())}"

    if isinstance(v, list):
        sample = v[:3]
        return f"<list> len={len(v)} first={[type(x).__name__ for x in sample]}"

    s = repr(v)
    if len(s) > max_len:
        s = s[:max_len] + "...[truncated]"
    return f"<{t}> {s}"


def inspect_image_field(v: Any) -> str:
    """Detailed description of an image-like field."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        PILImage = None

    if v is None:
        return "None"
    if PILImage is not None and isinstance(v, PILImage.Image):
        return f"PIL.Image mode={v.mode} size={v.size} format={v.format}"
    if isinstance(v, dict):
        # HF datasets often store images as {'bytes': ..., 'path': ...}
        keys = list(v.keys())
        if "bytes" in v and v["bytes"] is not None:
            b = v["bytes"]
            head = bytes(b)[:8].hex()
            try:
                img = PILImage.open(io.BytesIO(bytes(b))) if PILImage else None
                size = img.size if img else None
                mode = img.mode if img else None
            except Exception:
                size, mode = None, None
            return f"dict(keys={keys}) bytes_len={len(b)} head_hex={head} size={size} mode={mode}"
        return f"dict(keys={keys}) -> {v}"
    if isinstance(v, str):
        return f"str(path?)={v[:200]}"
    if isinstance(v, (bytes, bytearray)):
        return f"bytes len={len(v)}"
    return f"<{type(v).__name__}> {repr(v)[:200]}"


def main() -> None:
    print(f"=== Inspecting Guardian split: {SMALLEST_SPLIT} ===\n")

    # 1) Streaming first to see schema without downloading whole split.
    print("[1] Loading in STREAMING mode for first-sample inspection...")
    ds_stream = load_dataset(SMALLEST_SPLIT, split="train", streaming=True)
    iter_ds = iter(ds_stream)
    first = next(iter_ds)
    columns = list(first.keys())
    print(f"\n[2] Columns ({len(columns)}): {columns}\n")

    print("[3] One full sample, field-by-field:")
    for k in columns:
        print(f"    - {k}: {describe_value(first[k])}")
    print()

    # 2) Image storage details: print every field that looks image-ish.
    print("[4] Image-like fields (deeper look):")
    image_keywords = ("image", "img", "view", "obs", "rgb", "frame", "pixel")
    image_fields = [k for k in columns if any(kw in k.lower() for kw in image_keywords)]
    if not image_fields:
        # If no obvious name match, also include any PIL Image / dict-with-bytes field.
        for k in columns:
            v = first[k]
            try:
                from PIL import Image as PILImage
                if isinstance(v, PILImage.Image):
                    image_fields.append(k)
                    continue
            except ImportError:
                pass
            if isinstance(v, dict) and ("bytes" in v or "path" in v):
                image_fields.append(k)
    print(f"    detected image-like fields: {image_fields}")
    for k in image_fields:
        print(f"    - {k}: {inspect_image_field(first[k])}")
    print()

    # 3) Now download the full (small) split non-streaming to enumerate failure modes.
    print(f"[5] Downloading full split (non-streaming) to enumerate failure_mode values...")
    ds_full = load_dataset(SMALLEST_SPLIT, split="train")
    print(f"    total rows: {len(ds_full)}")
    print(f"    features: {ds_full.features}")
    print()

    # 4) Find the failure-mode column heuristically
    label_candidates = [
        c for c in ds_full.column_names
        if "failure" in c.lower() or c.lower() in {"label", "class", "mode", "outcome"}
    ]
    print(f"[6] Possible failure-mode columns: {label_candidates}")
    for col in label_candidates:
        try:
            vals = ds_full[col]
            counts = Counter(vals)
            print(f"    {col}: {len(counts)} unique values")
            for k, c in counts.most_common():
                print(f"      {k!r}: {c}")
        except Exception as e:
            print(f"    {col}: could not enumerate ({e})")
    print()

    print("=== Done. Use this output to design data/dataset.py. ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        raise
