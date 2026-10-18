"""Deterministic stratified sampling (pure stdlib, ported verbatim).

Kept dependency-free (only ``random``) so it is unit-testable without pandas and
identical run-to-run for a given ``(size, seed)``.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List


def stratified_sample_indices(strata: List[Any], size: int, seed: int) -> List[int]:
    """Return row indices for a deterministic stratified sample.

    Allocates ``size`` slots proportionally across the distinct ``strata`` values
    (largest-remainder method), then picks rows within each stratum using a
    fixed-seed shuffle. Returns all indices when ``size >= len(strata)``. The
    final index list is sorted for stable ordering.

    Args:
        strata: Stratum key per row (e.g. shard id), aligned to row order.
        size: Desired total sample size.
        seed: RNG seed for reproducibility.
    """
    n = len(strata)
    if size <= 0:
        return []
    if size >= n:
        return list(range(n))

    groups: Dict[Any, List[int]] = {}
    for idx, key in enumerate(strata):
        groups.setdefault(key, []).append(idx)

    keys = sorted(groups, key=lambda k: (str(k)))
    sizes = {k: len(groups[k]) for k in keys}
    raw_alloc = {k: size * sizes[k] / n for k in keys}
    base = {k: int(raw_alloc[k]) for k in keys}
    allocated = sum(base.values())
    remainder = size - allocated
    frac_order = sorted(
        keys, key=lambda k: (raw_alloc[k] - base[k], str(k)), reverse=True
    )
    for k in frac_order[: max(0, remainder)]:
        base[k] += 1

    rng = random.Random(seed)
    chosen: List[int] = []
    for k in keys:
        members = list(groups[k])
        rng.shuffle(members)
        take = min(base[k], len(members))
        chosen.extend(members[:take])

    chosen.sort()
    return chosen


__all__ = ["stratified_sample_indices"]
