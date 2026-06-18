"""Tests for S|P|O component fusion (concat vs Hadamard)."""

from __future__ import annotations

import unittest

import numpy as np

from vector_endpoint.component_fusion import (
    fuse_component_batch,
    parse_component_fusion,
    stored_embedding_dim,
)


class TestStoredEmbeddingDim(unittest.TestCase):
    def test_concat(self) -> None:
        self.assertEqual(stored_embedding_dim(384, "concat"), 1152)

    def test_hadamard(self) -> None:
        self.assertEqual(stored_embedding_dim(384, "hadamard"), 384)


class TestParseComponentFusion(unittest.TestCase):
    def test_defaults_and_aliases(self) -> None:
        self.assertEqual(parse_component_fusion("concat"), "concat")
        self.assertEqual(parse_component_fusion("HADAMARD"), "hadamard")

    def test_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_component_fusion("sum")


class TestFuseComponentBatch(unittest.TestCase):
    def test_concat_matches_manual(self) -> None:
        components = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=float)
        present = np.array([[True, True, True]], dtype=bool)
        got = fuse_component_batch(components, present, fusion="concat", normalize=False)
        np.testing.assert_allclose(got, np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))

    def test_hadamard_full_triple(self) -> None:
        s = np.array([2.0, 0.0])
        p = np.array([3.0, 0.0])
        o = np.array([0.0, 4.0])
        components = np.stack([s, p, o], axis=0).reshape(1, 3, 2)
        present = np.array([[True, True, True]], dtype=bool)
        got = fuse_component_batch(components, present, fusion="hadamard", normalize=False)
        np.testing.assert_allclose(got, np.array([[0.0, 0.0]]))

    def test_hadamard_partial_sp_pattern_nonzero(self) -> None:
        """Bound S+P with missing O (sp* query) must not collapse to zero."""
        s = np.array([2.0, 1.0])
        p = np.array([3.0, 2.0])
        components = np.zeros((1, 3, 2), dtype=float)
        components[0, 0, :] = s
        components[0, 1, :] = p
        present = np.array([[True, True, False]], dtype=bool)
        got = fuse_component_batch(components, present, fusion="hadamard", normalize=False)
        np.testing.assert_allclose(got, np.array([[6.0, 2.0]]))
        self.assertFalse(np.allclose(got, 0.0))

    def test_hadamard_partial_po_pattern_nonzero(self) -> None:
        p = np.array([2.0, 3.0])
        o = np.array([4.0, 5.0])
        components = np.zeros((1, 3, 2), dtype=float)
        components[0, 1, :] = p
        components[0, 2, :] = o
        present = np.array([[False, True, True]], dtype=bool)
        got = fuse_component_batch(components, present, fusion="hadamard", normalize=False)
        np.testing.assert_allclose(got, np.array([[8.0, 15.0]]))

    def test_hadamard_all_missing_zeros(self) -> None:
        components = np.ones((1, 3, 2), dtype=float)
        present = np.array([[False, False, False]], dtype=bool)
        got = fuse_component_batch(components, present, fusion="hadamard", normalize=True)
        np.testing.assert_allclose(got, np.zeros((1, 2)))

    def test_hadamard_normalize(self) -> None:
        components = np.array([[[2.0, 0.0], [2.0, 0.0], [1.0, 0.0]]], dtype=float)
        present = np.array([[True, True, True]], dtype=bool)
        got = fuse_component_batch(components, present, fusion="hadamard", normalize=True)
        self.assertAlmostEqual(float(np.linalg.norm(got[0])), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
