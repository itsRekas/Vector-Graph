from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

from vector_endpoint.catalog import Catalog


def _normalize_uri(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("<") and raw.endswith(">"):
        return raw
    if raw.startswith("http://") or raw.startswith("https://"):
        return f"<{raw}>"
    return raw


def _normalize_literal(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    match = re.match(r'^"(.*)"(?:\^\^.+|@[A-Za-z0-9-]+)?$', raw)
    if match:
        return match.group(1)
    return raw


def normalize_catalog_terms(
    *,
    subject: Optional[str],
    predicate: Optional[str],
    object_value: Optional[str],
    object_type: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    normalized_subject = _normalize_uri(subject)
    normalized_predicate = _normalize_uri(predicate)

    normalized_object: Optional[str]
    if object_value is None:
        normalized_object = None
    elif object_type == "uri":
        normalized_object = _normalize_uri(object_value)
    else:
        normalized_object = _normalize_literal(object_value)

    return normalized_subject, normalized_predicate, normalized_object


def scaled_k_from_count(count: int, *, scale: float = 1.2, min_k: int = 10) -> int:
    return max(min_k, int(math.ceil(max(0, count) * scale)))


def milvus_safe_k(k: int, *, milvus_max_topk: int = 16384) -> int:
    if k <= milvus_max_topk:
        return k
    return milvus_max_topk


class CatalogKResolver:
    def __init__(
        self,
        *,
        catalog_path: Optional[Path],
        scale: float = 1.2,
        min_k: int = 10,
    ) -> None:
        self.scale = scale
        self.min_k = min_k
        self.catalog_path = catalog_path
        self.catalog: Optional[Catalog] = None
        self.error: Optional[str] = None

        if catalog_path is None:
            self.error = "catalog path not set"
            return
        try:
            self.catalog = Catalog.load_pickle(catalog_path)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self.catalog is not None

    def catalog_match_count(
        self,
        *,
        subject: Optional[str],
        predicate: Optional[str],
        object_value: Optional[str],
        object_type: Optional[str] = None,
    ) -> Optional[int]:
        """Raw pattern match count from the catalog (used as a lower bound on hits)."""
        if self.catalog is None:
            return None
        s_norm, p_norm, o_norm = normalize_catalog_terms(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            object_type=object_type,
        )
        return self.catalog.count(subject=s_norm, predicate=p_norm, object_value=o_norm)

    def auto_k_for_pattern(
        self,
        *,
        subject: Optional[str],
        predicate: Optional[str],
        object_value: Optional[str],
        object_type: Optional[str] = None,
    ) -> Optional[int]:
        count = self.catalog_match_count(
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            object_type=object_type,
        )
        if count is None:
            return None
        return scaled_k_from_count(count, scale=self.scale, min_k=self.min_k)
