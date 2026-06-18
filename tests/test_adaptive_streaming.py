"""Tests for incremental adaptive finalization."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vector_endpoint.adaptive_exp import (  # noqa: E402
    adaptive_batch_search,
    iter_adaptive_batch_search,
)


class FakeVDB:
    """Minimal VectorDataBase stand-in: returns ``limit`` matches per query."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def search(self, *, collection_name, query_texts, limit, output_fields, log):
        self.calls.append((limit, len(query_texts)))
        # One result per query in the batch, each with `limit` synthetic matches.
        return [
            {"matches": [{"id": j} for j in range(limit)]}
            for _ in query_texts
        ]


def make_filter_fn():
    """Query 0 is selective (stable id set); query 1 grows with k (never stable)."""

    def filter_fn(matches, query_index):
        k = len(matches)
        if query_index == 0:
            return [{"q": 0, "k": k}], {0}
        return [{"q": 1, "k": k}], set(range(k))

    return filter_fn


COMMON_KWARGS = dict(
    collection_name="version_5",
    seed_ks=[1, 1],
    multipliers=(1, 2, 4, 8),
    jaccard_threshold=0.99,
)


class TestAdaptiveStreaming(unittest.TestCase):
    def test_selective_query_finalizes_before_non_selective(self):
        vdb = FakeVDB()
        search_queries = [{"text": "easy"}, {"text": "hard"}]
        emitted = list(
            iter_adaptive_batch_search(
                vdb=vdb,
                search_queries=search_queries,
                filter_fn=make_filter_fn(),
                **COMMON_KWARGS,
            )
        )
        order = [idx for idx, _rows in emitted]
        # Selective query (0) finalizes before non-selective (1).
        self.assertEqual(order, [0, 1])
        rows_by_idx = {idx: rows for idx, rows in emitted}
        self.assertEqual(rows_by_idx[0], [{"q": 0, "k": 2}])
        self.assertEqual(rows_by_idx[1], [{"q": 1, "k": 8}])

    def test_wrapper_returns_same_final_rows_and_calls_callback(self):
        vdb_iter = FakeVDB()
        search_queries = [{"text": "easy"}, {"text": "hard"}]
        iter_rows = {
            idx: rows
            for idx, rows in iter_adaptive_batch_search(
                vdb=vdb_iter,
                search_queries=search_queries,
                filter_fn=make_filter_fn(),
                **COMMON_KWARGS,
            )
        }
        expected = [iter_rows[0], iter_rows[1]]

        vdb_wrap = FakeVDB()
        callback_calls: list[tuple[int, list]] = []
        final_rows = adaptive_batch_search(
            vdb=vdb_wrap,
            search_queries=search_queries,
            filter_fn=make_filter_fn(),
            on_query_finalized=lambda i, rows: callback_calls.append((i, rows)),
            **COMMON_KWARGS,
        )
        self.assertEqual(final_rows, expected)
        # One callback per query, in finalization order.
        self.assertEqual([i for i, _ in callback_calls], [0, 1])
        self.assertEqual([rows for _, rows in callback_calls], expected)

    def test_empty_query_set_returns_empty(self):
        vdb = FakeVDB()
        self.assertEqual(
            list(
                iter_adaptive_batch_search(
                    vdb=vdb,
                    search_queries=[],
                    seed_ks=[],
                    filter_fn=make_filter_fn(),
                    collection_name="version_5",
                )
            ),
            [],
        )
        self.assertEqual(
            adaptive_batch_search(
                vdb=vdb,
                search_queries=[],
                seed_ks=[],
                filter_fn=make_filter_fn(),
                collection_name="version_5",
            ),
            [],
        )
        self.assertEqual(vdb.calls, [])

    def test_mismatched_seed_ks_raises(self):
        vdb = FakeVDB()
        with self.assertRaises(ValueError):
            list(
                iter_adaptive_batch_search(
                    vdb=vdb,
                    search_queries=[{"text": "a"}],
                    seed_ks=[1, 1],
                    filter_fn=make_filter_fn(),
                    collection_name="version_5",
                )
            )


if __name__ == "__main__":
    unittest.main()
