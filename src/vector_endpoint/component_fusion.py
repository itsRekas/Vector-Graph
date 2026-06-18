"""Fuse per-component S|P|O embeddings into a single stored/query vector."""

from __future__ import annotations

from typing import Literal

import numpy as np

ComponentFusion = Literal["concat", "hadamard"]

_VALID_FUSIONS = frozenset({"concat", "hadamard"})


def parse_component_fusion(value: str) -> ComponentFusion:
    normalized = value.strip().lower()
    if normalized not in _VALID_FUSIONS:
        raise ValueError(
            f"Invalid component_fusion={value!r}; expected one of {sorted(_VALID_FUSIONS)}"
        )
    return normalized  # type: ignore[return-value]


def stored_embedding_dim(component_dim: int, fusion: ComponentFusion) -> int:
    if component_dim <= 0:
        raise ValueError(f"component_dim must be > 0, got {component_dim}")
    if fusion == "concat":
        return component_dim * 3
    return component_dim


def fuse_component_batch(
    components: np.ndarray,
    present: np.ndarray,
    *,
    fusion: ComponentFusion,
    normalize: bool,
) -> np.ndarray:
    """Fuse batched component vectors.

    Args:
        components: shape (B, 3, d) — S, P, O slot order
        present: shape (B, 3) bool — True where slot is bound/present
        fusion: ``concat`` or ``hadamard``
        normalize: L2-normalize each output row (Hadamard only; concat ignores)

    Returns:
        shape (B, stored_embedding_dim(d, fusion))
    """
    if components.ndim != 3:
        raise ValueError(f"components must be (B, 3, d), got shape {components.shape}")
    if present.shape != components.shape[:2]:
        raise ValueError(
            f"present shape {present.shape} must match components batch/slots {(components.shape[0], 3)}"
        )

    batch_size, _slots, dim = components.shape
    if batch_size == 0:
        return np.zeros((0, stored_embedding_dim(dim, fusion)), dtype=float)

    if fusion == "concat":
        return components.reshape(batch_size, dim * 3)

    # Hadamard: identity (ones) for missing slots; product over present only.
    result = np.ones((batch_size, dim), dtype=float)
    any_present = present.any(axis=1)
    for slot in range(3):
        slot_present = present[:, slot]
        if not np.any(slot_present):
            continue
        result[slot_present] *= components[slot_present, slot, :]

    result[~any_present] = 0.0

    if normalize:
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        safe = np.where(norms > 0.0, norms, 1.0)
        result = result / safe
        result[~any_present] = 0.0

    return result
