"""Tests for pagination sessions and limit resolution."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vector_endpoint.auto_k import CatalogKResolver, resolve_pagination_limit  # noqa: E402
from vector_endpoint.db.VectorDataBase import SearchIteratorHandle  # noqa: E402
from vector_endpoint.pagination_sessions import (  # noqa: E402
    PAGINATION_SESSION_STORE,
    PaginationPageNotCached,
    PaginationSessionNotFound,
    resolve_session_id,
)
from vector_endpoint.pattern_query import PatternQueryInput  # noqa: E402

NO_CATALOG = CatalogKResolver(catalog_path=None)


class FakeIterator:
    def __init__(self, pages: list[list[dict]]) -> None:
        self._pages = list(pages)
        self.closed = False

    def next(self):
        if not self._pages:
            return None
        page = self._pages.pop(0)
        if not page:
            return None

        class _Page:
            def ids(self_inner):
                return [h["id"] for h in page]

            def distances(self_inner):
                return [h.get("distance", 0.0) for h in page]

            def __getitem__(self_inner, idx):
                hit = page[idx]

                class _Hit:
                    id = hit["id"]
                    distance = hit.get("distance", 0.0)

                    class entity:
                        @staticmethod
                        def get(field):
                            return hit.get(field)

                return _Hit()

        return _Page()

    def close(self) -> None:
        self.closed = True


class FakeVDB:
    def __init__(self, pages: list[list[dict]]) -> None:
        self.pages = pages
        self.open_calls: list[tuple[int, int]] = []

    def open_search_iterator(
        self,
        collection_name,
        query_text,
        *,
        batch_size,
        limit,
        output_fields=None,
        log=False,
    ):
        self.open_calls.append((batch_size, limit))
        return SearchIteratorHandle(
            FakeIterator(self.pages),
            output_fields=output_fields or ["text"],
            metric_type="COSINE",
        )


class TestResolvePaginationLimit(unittest.TestCase):
    def test_default_doubles_catalog_k(self):
        self.assertEqual(resolve_pagination_limit(500, catalog_k=600, explicit_limit=None), 1200)

    def test_bumps_limit_to_k_when_larger_than_default(self):
        self.assertEqual(resolve_pagination_limit(1500, catalog_k=600, explicit_limit=None), 1500)

    def test_explicit_limit_rejects_oversized_k(self):
        with self.assertRaises(ValueError):
            resolve_pagination_limit(1500, catalog_k=600, explicit_limit=1000)

    def test_no_catalog_uses_2x_k(self):
        self.assertEqual(resolve_pagination_limit(500, catalog_k=None, explicit_limit=None), 1000)


class TestPaginationFlow(unittest.TestCase):
    def setUp(self) -> None:
        PAGINATION_SESSION_STORE._sessions.clear()

    def test_start_and_next_until_done(self):
        from vector_endpoint.pagination_search import resolve_pagination_page, start_pagination_page

        pages = [
            [{"id": 1, "distance": 0.1, "text": "a"}],
            [{"id": 2, "distance": 0.2, "text": "b"}],
            [],
        ]
        vdb = FakeVDB(pages)
        query_input = PatternQueryInput.from_json(
            {
                "k_mode": "pagination",
                "k": 1,
                "limit": 2,
                "pattern": {
                    "subject": {"type": "iri", "value": "http://example.org/s"},
                    "predicate": {"type": "iri", "value": "http://example.org/p"},
                    "object": {"type": "literal", "value": "o"},
                },
                "vars": [],
                "values": [{}],
            }
        )
        first = start_pagination_page(
            query_input,
            collection_name="version_5",
            database=vdb,
            resolver=NO_CATALOG,
        )
        self.assertFalse(first.pagination.done)
        self.assertIsNotNone(first.pagination.cursor)
        pagination_dict = first.pagination.to_dict()
        self.assertIn("session", pagination_dict)
        self.assertNotIn("cursor", pagination_dict)

        session_id = first.pagination.cursor
        second = resolve_pagination_page(session_id)
        self.assertTrue(second.pagination.done)
        self.assertIsNone(second.pagination.cursor)
        self.assertEqual(second.pagination.milvus_hits_total, 2)

    def test_unknown_session_raises(self):
        from vector_endpoint.pagination_search import resolve_pagination_page

        with self.assertRaises(PaginationSessionNotFound):
            resolve_pagination_page("missing-session")

    def test_page_cache_hit_and_miss(self):
        from vector_endpoint.pagination_search import resolve_pagination_page, start_pagination_page

        pages = [
            [{"id": 1, "distance": 0.1, "text": "a"}],
            [{"id": 2, "distance": 0.2, "text": "b"}],
            [{"id": 3, "distance": 0.3, "text": "c"}],
            [],
        ]
        vdb = FakeVDB(pages)
        query_input = PatternQueryInput.from_json(
            {
                "k_mode": "pagination",
                "k": 1,
                "limit": 3,
                "pattern": {
                    "subject": {"type": "iri", "value": "http://example.org/s"},
                    "predicate": {"type": "iri", "value": "http://example.org/p"},
                    "object": {"type": "literal", "value": "o"},
                },
                "vars": [],
                "values": [{}],
            }
        )
        first = start_pagination_page(
            query_input,
            collection_name="version_5",
            database=vdb,
            resolver=NO_CATALOG,
        )
        session_id = first.pagination.cursor
        self.assertEqual(len(PAGINATION_SESSION_STORE.get(session_id).page_cache), 1)

        resolve_pagination_page(session_id)
        self.assertEqual(len(PAGINATION_SESSION_STORE.get(session_id).page_cache), 2)

        cached = resolve_pagination_page(session_id, page=1)
        self.assertTrue(cached.pagination.from_cache)
        self.assertEqual(cached.pagination.page_index, 1)
        self.assertEqual(len(vdb.open_calls), 1)

        with self.assertRaises(PaginationPageNotCached):
            resolve_pagination_page(session_id, page=99)

    def test_legacy_cursor_alias(self):
        self.assertEqual(
            resolve_session_id({"cursor": "abc-123"}),
            "abc-123",
        )
        self.assertEqual(
            resolve_session_id({"session": "def-456"}),
            "def-456",
        )

    def test_close_clears_page_cache(self):
        from vector_endpoint.pagination_search import start_pagination_page

        pages = [[{"id": 1, "distance": 0.1, "text": "a"}], []]
        vdb = FakeVDB(pages)
        query_input = PatternQueryInput.from_json(
            {
                "k_mode": "pagination",
                "k": 1,
                "limit": 1,
                "pattern": {
                    "subject": {"type": "iri", "value": "http://example.org/s"},
                    "predicate": {"type": "iri", "value": "http://example.org/p"},
                    "object": {"type": "literal", "value": "o"},
                },
                "vars": [],
                "values": [{}],
            }
        )
        first = start_pagination_page(
            query_input,
            collection_name="version_5",
            database=vdb,
            resolver=NO_CATALOG,
        )
        session_id = first.pagination.cursor
        PAGINATION_SESSION_STORE.close(session_id)
        self.assertNotIn(session_id, PAGINATION_SESSION_STORE._sessions)

    def test_collect_pagination_pages_in_process(self):
        from vector_endpoint.pagination_search import collect_pagination_pages

        pages = [[{"id": i, "distance": 0.1, "text": "t"} for i in range(3)], []]
        vdb = FakeVDB(pages)
        query_input = PatternQueryInput.from_json(
            {
                "k_mode": "pagination",
                "k": 3,
                "limit": 3,
                "pattern": {
                    "subject": "s",
                    "predicate": {"type": "iri", "value": "http://example.org/p"},
                    "object": {"type": "literal", "value": "o"},
                },
                "vars": [],
                "values": [{}],
                "include_raw_hits": True,
            }
        )
        rows, last, raw = collect_pagination_pages(
            query_input,
            collection_name="version_5",
            database=vdb,
            resolver=NO_CATALOG,
        )
        self.assertEqual(len(raw), 3)
        self.assertTrue(last.pagination.done)


if __name__ == "__main__":
    unittest.main()
