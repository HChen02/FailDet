"""ClassBalancedBatchSampler — every batch contains all classes present.

For SupCon/SINCERE-style contrastive training, batches that miss a class
contribute zero positive pairs for that class. This sampler avoids that
by stratifying per-batch.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Optional

from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """Yield batches such that every class present in `labels` appears at
    least once per batch. Remaining batch slots are filled by uniform
    random sampling (with replacement) across the whole dataset.

    Args:
        labels: integer class id per dataset sample. Pass the full list
            of labels — caller extracts this from the metadata table to
            avoid loading images.
        batch_size: number of indices per batch. Must be >= number of
            distinct classes.
        num_batches: how many batches per epoch. Defaults to
            len(labels) // batch_size.
        seed: RNG seed for reproducibility.

    Use as `DataLoader(ds, batch_sampler=ClassBalancedBatchSampler(...))`.
    """

    def __init__(
        self,
        labels: Iterable[int],
        batch_size: int,
        num_batches: Optional[int] = None,
        seed: int = 42,
    ):
        labels = [int(c) for c in labels]
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        by_class: dict[int, list[int]] = {}
        for i, c in enumerate(labels):
            if c < 0:
                continue  # skip "unknown" labels
            by_class.setdefault(c, []).append(i)
        if not by_class:
            raise ValueError("no labeled samples — every label was negative.")
        self._classes = sorted(by_class.keys())
        if batch_size < len(self._classes):
            raise ValueError(
                f"batch_size={batch_size} < num_classes={len(self._classes)}; "
                "cannot place at least one sample per class per batch."
            )
        self._by_class = by_class
        self._all_indices = [i for c in self._classes for i in by_class[c]]
        self._batch_size = batch_size
        self._num_batches = (
            num_batches if num_batches is not None
            else max(1, len(labels) // batch_size)
        )
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self._num_batches):
            batch: list[int] = []
            # Stage 1: one sample per class.
            for c in self._classes:
                batch.append(self._rng.choice(self._by_class[c]))
            # Stage 2: fill the rest by uniform sampling with replacement.
            for _ in range(self._batch_size - len(batch)):
                batch.append(self._rng.choice(self._all_indices))
            self._rng.shuffle(batch)
            yield batch

    @property
    def classes(self) -> list[int]:
        return list(self._classes)
