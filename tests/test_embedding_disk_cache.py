"""Tests for disk-backed component embedding cache."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vector_endpoint.embedding_disk_cache import (  # noqa: E402
    DiskEmbeddingCache,
    adjust_component,
    build_meta,
    collect_component_texts_from_triple,
    embed_triple_batch,
    load_cache,
    save_cache,
    validate_meta,
)


class TestComponentTextKeys(unittest.TestCase):
    def test_literal_object_key(self):
        triple = {
            "subject": "<http://ex/s>",
            "predicate": "<http://ex/p>",
            "object": "hello",
            "object_type": "literal",
        }
        subj, pred, obj = collect_component_texts_from_triple(triple)
        self.assertEqual(subj, "<http://ex/s>")
        self.assertEqual(pred, "<http://ex/p>")
        self.assertEqual(obj, "literal:hello")

    def test_uri_object_key(self):
        triple = {
            "subject": "<http://ex/s>",
            "predicate": "<http://ex/p>",
            "object": "<http://ex/o>",
            "object_type": "uri",
        }
        _, _, obj = collect_component_texts_from_triple(triple)
        self.assertEqual(obj, "<http://ex/o>")


class TestAdjustComponent(unittest.TestCase):
    def test_truncate_and_normalize_matches_manual(self):
        raw = np.array([3.0, 4.0, 0.0, 0.0], dtype=float)
        got = adjust_component(raw, target_dim=2, normalize=True)
        expected = np.array([0.6, 0.8], dtype=float)
        np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_zero_vector_stays_zero(self):
        raw = np.zeros(4, dtype=float)
        got = adjust_component(raw, target_dim=2, normalize=True)
        np.testing.assert_array_equal(got, np.zeros(2))


class TestDiskEmbeddingCache(unittest.TestCase):
    def test_put_and_lookup(self):
        cache = DiskEmbeddingCache.from_dict({})
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.put("a", vec)
        np.testing.assert_array_equal(cache.lookup_raw("a"), vec)

    def test_embed_triple_batch_uses_cache(self):
        cache = DiskEmbeddingCache.from_dict(
            {
                "s": np.array([1.0, 0.0], dtype=np.float32),
                "p": np.array([0.0, 1.0], dtype=np.float32),
                "literal:o": np.array([1.0, 1.0], dtype=np.float32),
            }
        )
        triple = {
            "subject": "s",
            "predicate": "p",
            "object": "o",
            "object_type": "literal",
        }
        got = embed_triple_batch([triple], cache, target_dim=2, normalize=False)
        self.assertEqual(got.shape, (1, 6))
        np.testing.assert_array_equal(got[0, :2], np.array([1.0, 0.0]))
        np.testing.assert_array_equal(got[0, 2:4], np.array([0.0, 1.0]))
        np.testing.assert_array_equal(got[0, 4:6], np.array([1.0, 1.0]))

    def test_embed_triple_batch_hadamard_uses_cache(self):
        cache = DiskEmbeddingCache.from_dict(
            {
                "s": np.array([2.0, 0.0], dtype=np.float32),
                "p": np.array([3.0, 0.0], dtype=np.float32),
                "literal:o": np.array([0.0, 4.0], dtype=np.float32),
            }
        )
        triple = {
            "subject": "s",
            "predicate": "p",
            "object": "o",
            "object_type": "literal",
        }
        got = embed_triple_batch(
            [triple], cache, target_dim=2, normalize=False, fusion="hadamard"
        )
        self.assertEqual(got.shape, (1, 2))
        np.testing.assert_array_equal(got[0], np.array([0.0, 0.0]))


class TestCachePersistence(unittest.TestCase):
    def test_save_load_round_trip(self):
        cache = DiskEmbeddingCache.from_dict(
            {
                "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "y": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            }
        )
        meta = {
            "input_file": "/tmp/data.nt",
            "input_file_size": 123,
            "input_file_mtime": 1.0,
            "embedding_model": "all-MiniLM-L6-v2",
            "dim_adjustment": "truncate",
            "cache_full_dim": 3,
            "unique_components": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "cache.npz"
            save_cache(npz_path, cache, meta)
            loaded, loaded_meta = load_cache(npz_path)
            self.assertEqual(loaded_meta["embedding_model"], "all-MiniLM-L6-v2")
            np.testing.assert_array_equal(loaded.lookup_raw("x"), cache.lookup_raw("x"))
            np.testing.assert_array_equal(loaded.lookup_raw("y"), cache.lookup_raw("y"))


class TestValidateMeta(unittest.TestCase):
    def test_rejects_model_mismatch(self):
        with tempfile.NamedTemporaryFile(suffix=".nt", delete=False) as handle:
            handle.write(b"<a> <b> <c> .\n")
            nt_path = Path(handle.name)
        try:
            meta = build_meta(
                nt_path=nt_path,
                embedding_model="all-MiniLM-L6-v2",
                dim_adjustment="truncate",
                cache_full_dim=384,
                unique_components=1,
            )
            with self.assertRaises(ValueError):
                validate_meta(
                    meta,
                    nt_path=nt_path,
                    embedding_model="other-model",
                    dim_adjustment="truncate",
                )
        finally:
            nt_path.unlink(missing_ok=True)

    def test_rejects_stale_mtime(self):
        with tempfile.NamedTemporaryFile(suffix=".nt", delete=False) as handle:
            handle.write(b"<a> <b> <c> .\n")
            nt_path = Path(handle.name)
        try:
            meta = build_meta(
                nt_path=nt_path,
                embedding_model="all-MiniLM-L6-v2",
                dim_adjustment="truncate",
                cache_full_dim=384,
                unique_components=1,
            )
            meta["input_file_mtime"] = 0.0
            with self.assertRaises(ValueError):
                validate_meta(
                    meta,
                    nt_path=nt_path,
                    embedding_model="all-MiniLM-L6-v2",
                    dim_adjustment="truncate",
                )
        finally:
            nt_path.unlink(missing_ok=True)


@unittest.skipUnless(
    os.environ.get("RUN_EMBEDDING_CACHE_INTEGRATION") == "1",
    "set RUN_EMBEDDING_CACHE_INTEGRATION=1 to run model parity test",
)
class TestEmbeddingParityIntegration(unittest.TestCase):
    def test_cache_embed_matches_live_encode(self):
        from vector_endpoint.db.VectorDataBase import VectorDataBase
        from vector_endpoint.embedding_disk_cache import embed_triple_batch

        triples = [
            {
                "subject": "<http://www.Department0.University0.edu/FullProfessor0>",
                "predicate": "<http://www.lehigh.edu/~zhp16/ubm/ontologies/University0.owl#name>",
                "object": "FullProfessor0",
                "object_type": "literal",
            },
            {
                "subject": "<http://www.Department0.University0.edu>",
                "predicate": "<http://www.lehigh.edu/~zhp16/ubm/ontologies/University0.owl#memberOf>",
                "object": "<http://www.University0.edu>",
                "object_type": "uri",
            },
        ]

        vdb_full = VectorDataBase(
            database_name="test",
            host="localhost",
            port=19530,
            embedding_model="all-MiniLM-L6-v2",
            target_embedding_dim=384,
            dim_adjustment="truncate",
        )
        cache = DiskEmbeddingCache.from_dict({})
        for triple in triples:
            for text in collect_component_texts_from_triple(triple):
                if text and cache.lookup_raw(text) is None:
                    raw = vdb_full._encode_text_batch([text], normalize=False)[0]
                    cache.put(text, raw)

        vdb_small = VectorDataBase(
            database_name="test",
            host="localhost",
            port=19530,
            embedding_model="all-MiniLM-L6-v2",
            target_embedding_dim=8,
            dim_adjustment="truncate",
        )
        expected = vdb_small._embed_triple_batch(triples, normalize=True)
        got = embed_triple_batch(triples, cache, target_dim=8, normalize=True)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_cache_embed_hadamard_matches_live_encode(self):
        from vector_endpoint.db.VectorDataBase import VectorDataBase
        from vector_endpoint.embedding_disk_cache import embed_triple_batch

        triples = [
            {
                "subject": "<http://www.Department0.University0.edu/FullProfessor0>",
                "predicate": "<http://www.lehigh.edu/~zhp16/ubm/ontologies/University0.owl#name>",
                "object": "FullProfessor0",
                "object_type": "literal",
            },
        ]

        vdb_full = VectorDataBase(
            database_name="test",
            host="localhost",
            port=19530,
            embedding_model="all-MiniLM-L6-v2",
            target_embedding_dim=384,
            dim_adjustment="truncate",
            component_fusion="hadamard",
        )
        cache = DiskEmbeddingCache.from_dict({})
        for triple in triples:
            for text in collect_component_texts_from_triple(triple):
                if text and cache.lookup_raw(text) is None:
                    raw = vdb_full._encode_text_batch([text], normalize=False)[0]
                    cache.put(text, raw)

        vdb_small = VectorDataBase(
            database_name="test",
            host="localhost",
            port=19530,
            embedding_model="all-MiniLM-L6-v2",
            target_embedding_dim=8,
            dim_adjustment="truncate",
            component_fusion="hadamard",
        )
        expected = vdb_small._embed_triple_batch(triples, normalize=True)
        got = embed_triple_batch(
            triples, cache, target_dim=8, normalize=True, fusion="hadamard"
        )
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
