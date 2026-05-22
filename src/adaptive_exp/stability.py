"""Pure helpers for the adaptive-k escalation strategy.

- `build_k_ladder` produces the per-query k schedule (seed, seed*10, ...)
  capped at the Milvus top-k limit with consecutive duplicates removed.
- `jaccard` is a plain Jaccard index over two sets. Callers are responsible
  for deciding what the "both empty" case means.
- `is_stable` applies the stability rule described in the plan:
    * `prev is None` (first round) -> not stable
    * both empty                   -> not stable (never stop on empty-empty)
    * sizes differ                 -> not stable
    * exact non-empty equality     -> stable
    * sizes match & jaccard >= thr -> stable
"""

from __future__ import annotations

from typing import Iterable, Optional

from auto_k import milvus_safe_k


def build_k_ladder(
    seed_k: int,
    *,
    multipliers: Iterable[int] = (1, 10, 100, 1000),
    milvus_max_topk: int = 16384,
) -> list[int]:
    """Return the ascending list of k values to try for a single query.

    Each rung is `seed_k * m` capped at `milvus_max_topk`. Consecutive
    duplicates produced by the cap are removed so the ladder naturally ends
    once we hit the maximum.
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
