"""Stratified low-data splits — fixed once, reused by every method.

For E2 (data-efficiency curves) we sample subsets of the training set at
1%, 5%, 10%, 25%, 50% with three seeds each. Splits MUST be the same
across methods or the comparison is confounded by sample selection.

Output JSON layout:
    {
      "1pct_seed42":  [0, 5, 17, ...],
      "5pct_seed42":  [...],
      ...,
      "50pct_seed456":[...]
    }
Indices are 0-based and reference the dataset whose labels were passed
to `create_low_data_splits`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def create_low_data_splits(
    labels: Iterable[int],
    percentages: Iterable[float],
    seeds: Iterable[int],
) -> dict[str, list[int]]:
    """Return stratified subsets at each (percentage, seed) pair.

    If a percentage is too small for stratification (e.g. would leave a
    rare class with zero samples), fall back to a forced-coverage random
    selection: pick one of each class first, then fill the rest by
    uniform random sampling.
    """
    arr = np.asarray([int(c) for c in labels])
    n = len(arr)
    out: dict[str, list[int]] = {}
    for pct in percentages:
        if not (0.0 < pct <= 1.0):
            raise ValueError(f"percentage must be in (0, 1]; got {pct}")
        n_select = max(1, int(round(pct * n)))
        for seed in seeds:
            key = f"{int(round(pct * 100))}pct_seed{int(seed)}"
            if pct == 1.0:
                out[key] = list(range(n))
                continue
            try:
                splitter = StratifiedShuffleSplit(
                    n_splits=1, train_size=n_select, random_state=seed
                )
                train_idx, _ = next(splitter.split(np.zeros(n), arr))
                out[key] = sorted(int(i) for i in train_idx)
            except ValueError:
                rng = np.random.default_rng(seed)
                forced: list[int] = []
                for c in np.unique(arr):
                    cls_idx = np.flatnonzero(arr == c)
                    if cls_idx.size:
                        forced.append(int(rng.choice(cls_idx)))
                remaining = max(0, n_select - len(forced))
                pool = list(set(range(n)) - set(forced))
                rng.shuffle(pool)
                out[key] = sorted(forced + pool[:remaining])
    return out


def class_distribution(indices: list[int], labels: list[int]) -> dict[int, int]:
    """Class histogram over a subset, keyed by integer class id."""
    arr = np.asarray([int(labels[i]) for i in indices])
    if arr.size == 0:
        return {}
    uniq, cnt = np.unique(arr, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, cnt)}


def save_splits(splits: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
