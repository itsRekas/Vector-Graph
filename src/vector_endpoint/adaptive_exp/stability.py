"""Helpers for adaptive-k escalation (ladder, Jaccard, stability check)."""

from __future__ import annotations

from typing import Iterable, Optional

from vector_endpoint.auto_k import milvus_safe_k


def build_k_ladder(
    seed_k: int,
    *,
    multipliers: Iterable[int] = (1, 10, 100, 1000),
    milvus_max_topk: int = 200000,
) -> list[int]:
    """Return the ascending list of k values to try for a single query.

    Each rung is ``seed_k * m`` capped at ``milvus_max_topk``. Consecutive
    duplicates from the cap are removed so the ladder ends at the maximum.
    """
    if seed_k <= 0:
        seed_k = 1
    ladder: list[int] = []
    for m in multipliers:
        raw = seed_k * int(m)
        capped = milvus_safe_k(raw, milvus_max_topk=milvus_max_topk)
        if ladder and ladder[-1] == capped:
            continue
        ladder.append(capped)
    return ladder


def jaccard(a: set[int], b: set[int]) -> float:
    """Plain Jaccard index. Returns 0.0 when both sets are empty."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def is_stable(
    prev: Optional[set[int]],
    curr: set[int],
    *,
    jaccard_threshold: float = 0.99,
) -> bool:
    """Decide whether escalation can stop after observing `curr`."""
    if prev is None:
        return False
    if not prev and not curr:
        return False
    if len(prev) != len(curr):
        return False
    if prev == curr:
        return True
    return jaccard(prev, curr) >= jaccard_threshold
