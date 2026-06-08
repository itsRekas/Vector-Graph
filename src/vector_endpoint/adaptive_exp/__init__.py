"""Adaptive k-escalation search experiments.

Provides per-query k-escalation with a post-filter stability criterion on top
of the existing VectorDataBase batch search.
"""

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
