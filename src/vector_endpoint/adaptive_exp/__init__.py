"""Adaptive k-escalation with post-filter stability over VectorDataBase search."""

from .stability import build_k_ladder, jaccard, is_stable
from .filter import filter_matches_to_rows
from .adaptive_search import adaptive_batch_search, iter_adaptive_batch_search

__all__ = [
    "build_k_ladder",
    "jaccard",
    "is_stable",
    "filter_matches_to_rows",
    "adaptive_batch_search",
    "iter_adaptive_batch_search",
]
