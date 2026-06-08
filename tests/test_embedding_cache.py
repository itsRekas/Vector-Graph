"""Tests for the LRU embedding component cache.

Runnable without pytest:  ``.venv/bin/python -m unittest discover -s tests``
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vector_endpoint.db.VectorDataBase import _LruEmbeddingCache  # noqa: E402


class TestLruEmbeddingCache(unittest.TestCase):
    def test_put_and_get_round_trip(self):
        cache = _LruEmbeddingCache(4)
        vec = np.array([1.0, 2.0, 3.0])
        key = _LruEmbeddingCache.make_key("http://ex/p", normalize=True)
        cache.put(key, vec)
        got = cache.get(key)
        self.assertIsNotNone(got)
        np.testing.assert_array_equal(got, vec)
        # Stored copy is independent of the source array.
        vec[0] = 99.0
        np.testing.assert_array_equal(cache.get(key), np.array([1.0, 2.0, 3.0]))

    def test_norm_and_raw_keys_differ(self):
        text = "http://ex/p"
        self.assertNotEqual(
            _LruEmbeddingCache.make_key(text, normalize=True),
            _LruEmbeddingCache.make_key(text, normalize=False),
        )

    def test_lru_evicts_least_recently_used_not_insertion_order(self):
        cache = _LruEmbeddingCache(2)
        key_a = "a"
        key_b = "b"
        key_c = "c"
        cache.put(key_a, np.array([1.0]))
        cache.put(key_b, np.array([2.0]))
        # Touch a so b becomes LRU.
        self.assertIsNotNone(cache.get(key_a))
        cache.put(key_c, np.array([3.0]))
        self.assertIsNotNone(cache.get(key_a))
        self.assertIsNone(cache.get(key_b))
        self.assertIsNotNone(cache.get(key_c))

    def test_put_updates_existing_key_without_growing(self):
        cache = _LruEmbeddingCache(2)
        key = "hot"
        cache.put(key, np.array([1.0]))
        cache.put("other", np.array([2.0]))
        cache.put(key, np.array([9.0]))
        self.assertEqual(len(cache), 2)
        np.testing.assert_array_equal(cache.get(key), np.array([9.0]))

    def test_clear_returns_previous_size(self):
        cache = _LruEmbeddingCache(4)
        cache.put("x", np.array([1.0]))
        cache.put("y", np.array([2.0]))
        self.assertEqual(cache.clear(), 2)
        self.assertEqual(len(cache), 0)

    def test_invalid_max_size_raises(self):
        with self.assertRaises(ValueError):
            _LruEmbeddingCache(0)


if __name__ == "__main__":
    unittest.main()
