"""Adaptive k-escalation orchestrator.

Runs a per-query k-ladder against a batched vector database. Each round,
active queries are grouped by their current k for one `vdb.search` per group.

The stability rule lives in `stability.is_stable` and is applied to the
post-filtered Milvus id set returned by `filter_fn`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Optional

from .stability import build_k_ladder, jaccard


_StopReason = str  # includes "stable_below_floor" when stability holds but |S| < catalog floor


def _classify(
    prev: Optional[set[int]],
    curr: set[int],
    *,
    jaccard_threshold: float,
) -> tuple[bool, _StopReason]:
    """Decide whether to stop escalation for a query and label the reason."""
    if prev is None:
        return False, "first_round"
    if not prev and not curr:
        return False, "both_empty"
    if len(prev) != len(curr):
        return False, "unstable"
    if prev == curr:
        return True, "exact"
    if jaccard(prev, curr) >= jaccard_threshold:
        return True, "near"
    return False, "unstable"


def adaptive_batch_search(
    *,
    vdb,
    collection_name: str,
    search_queries: list[dict],
    seed_ks: list[int],
    filter_fn: Callable[[list[dict], int], tuple[list[dict], set[int]]],
    multipliers: Iterable[int] = (1, 10, 100, 1000),
    jaccard_threshold: float = 0.99,
    output_fields: Iterable[str] = ("text",),
    milvus_max_topk: int = 200000,
    log: bool = False,
    stability_count_floors: Optional[list[Optional[int]]] = None,
) -> list[list[dict]]:
    """Run per-query k-escalation with stability-based early stopping.

    Args:
        vdb: VectorDataBase-like object exposing a `search(...)` method.
        collection_name: Milvus collection name.
        search_queries: list of query dicts (one per value_row) already built
            by the caller. They are passed as `query_texts` to `vdb.search`.
        seed_ks: list of seed k values, one per `search_queries` entry.
        filter_fn: callable `(matches, query_index) -> (rows, filtered_ids)`.
            The caller provides this as a closure bound to the per-row
            validation context.
        multipliers: ladder multipliers applied to each seed_k.
        jaccard_threshold: near-stability threshold (default 0.99).
        output_fields: fields requested from `vdb.search`.
        milvus_max_topk: cap used when building ladders.
        log: when True, prints per-round stability diagnostics.
        stability_count_floors: optional per-query catalog lower bound on the
            post-filtered id set size. Escalation continues while below the
            floor even if Jaccard is stable. Per-query ``None`` skips the floor.
            Omit the argument entirely to disable floors for all queries.

    Returns:
        `final_rows[i]` is the list of result rows for query `i` at the round
        where it stopped (or the last ladder rung if it never stabilized).
    """
    n = len(search_queries)
    if n == 0:
        return []
    if len(seed_ks) != n:
        raise ValueError(f"seed_ks length {len(seed_ks)} != search_queries length {n}")
    if stability_count_floors is not None and len(stability_count_floors) != n:
        raise ValueError(
            f"stability_count_floors length {len(stability_count_floors)} "
            f"!= search_queries length {n}"
        )

    ladders = [
        build_k_ladder(k, multipliers=multipliers, milvus_max_topk=milvus_max_topk)
        for k in seed_ks
    ]
    max_rounds = max(len(ladder) for ladder in ladders)

    prev_ids: list[Optional[set[int]]] = [None] * n
    final_rows: list[list[dict]] = [[] for _ in range(n)]
    active: set[int] = set(range(n))

    out_fields = list(output_fields)

    for round_idx in range(max_rounds):
        if not active:
            break

        groups: dict[int, list[int]] = defaultdict(list)
        for i in list(active):
            if round_idx >= len(ladders[i]):
                active.discard(i)
                continue
            groups[ladders[i][round_idx]].append(i)

        if not groups:
            break

        for k, indices in groups.items():
            batch = [search_queries[i] for i in indices]
            try:
                results = vdb.search(
                    collection_name=collection_name,
                    query_texts=batch,
                    limit=k,
                    output_fields=out_fields,
                    log=log,
                )
            except Exception:  # noqa: BLE001
                raise

            for local_idx, i in enumerate(indices):
                if local_idx >= len(results):
                    continue
                matches = results[local_idx].get("matches", [])
                rows, ids = filter_fn(matches, i)
                final_rows[i] = rows

                stable, _reason = _classify(
                    prev_ids[i], ids, jaccard_threshold=jaccard_threshold
                )
                floor: Optional[int] = None
                if stability_count_floors is not None:
                    floor = stability_count_floors[i]
                below_floor = floor is not None and len(ids) < floor
                if stable and below_floor:
                    stopped = False
                else:
                    stopped = stable
                prev_ids[i] = ids
                if stopped:
                    active.discard(i)

    return final_rows


__all__ = ["adaptive_batch_search"]
