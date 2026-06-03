from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import pickle
import re
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union


TripleKey = Tuple[str, str, str]
PairKey = Tuple[str, str]


class _CompatUnpickler(pickle.Unpickler):
    """Unpickler that loads catalogs saved before the package move.

    Older ``catalog.pkl`` files reference ``catalog.Catalog`` (top-level module).
    Remaps that module path to ``vector_endpoint.catalog`` on load.
    """

    def find_class(self, module: str, name: str):
        if module == "catalog":
            module = "vector_endpoint.catalog"
        return super().find_class(module, name)


@dataclass(frozen=True)
class TripleRecord:
    subject: str
    predicate: str
    object_value: str


def parse_nt_triple_line(line: Optional[str]) -> Optional[TripleRecord]:
    """
    Parse a single NT-style triple line.

    Representation is intentionally aligned with existing project code:
    - subject: '<uri>'
    - predicate: '<uri>'
    - object: '<uri>' for URI objects, raw literal text for literals
    """
    if not line:
        return None

    triple_pattern = r'<([^>]+)>\s+<([^>]+)>\s+(?:"([^"]+)"|<([^>]+)>)\s*\.?'
    match = re.match(triple_pattern, line.strip())
    if not match:
        return None

    subject = f"<{match.group(1)}>"
    predicate = f"<{match.group(2)}>"
    object_literal = match.group(3)
    object_uri = match.group(4)
    object_value = object_literal if object_literal is not None else f"<{object_uri}>"
    return TripleRecord(subject=subject, predicate=predicate, object_value=object_value)


class Catalog:
    """
    Pickle-friendly cardinality catalog for triple-pattern selectivity.

    Index families:
    - single: s, p, o
    - pair: sp, po, so
    - full: spo
    """

    def __init__(self, track_spo: bool = False) -> None:
        self.track_spo: bool = track_spo
        self.total_triples: int = 0
        self.s_counts: Dict[str, int] = {}
        self.p_counts: Dict[str, int] = {}
        self.o_counts: Dict[str, int] = {}
        self.sp_counts: Dict[PairKey, int] = {}
        self.po_counts: Dict[PairKey, int] = {}
        self.so_counts: Dict[PairKey, int] = {}
        self.spo_counts: Dict[TripleKey, int] = {}

    @staticmethod
    def _inc(store: Dict[Union[str, PairKey, TripleKey], int], key: Union[str, PairKey, TripleKey]) -> None:
        store[key] = store.get(key, 0) + 1

    def add_triple(self, subject: str, predicate: str, object_value: str) -> None:
        self.total_triples += 1
        self._inc(self.s_counts, subject)
        self._inc(self.p_counts, predicate)
        self._inc(self.o_counts, object_value)
        self._inc(self.sp_counts, (subject, predicate))
        self._inc(self.po_counts, (predicate, object_value))
        self._inc(self.so_counts, (subject, object_value))
        if self.track_spo:
            self._inc(self.spo_counts, (subject, predicate, object_value))

    def add_batch(self, triples: Iterable[Union[TripleRecord, Tuple[str, str, str], Dict[str, str]]]) -> int:
        added = 0
        for triple in triples:
            subject: Optional[str] = None
            predicate: Optional[str] = None
            object_value: Optional[str] = None

            if isinstance(triple, TripleRecord):
                subject, predicate, object_value = triple.subject, triple.predicate, triple.object_value
            elif isinstance(triple, tuple) and len(triple) == 3:
                subject, predicate, object_value = triple
            elif isinstance(triple, dict):
                subject = triple.get("subject")
                predicate = triple.get("predicate")
                object_value = triple.get("object_value", triple.get("object"))

            if subject and predicate and object_value:
                self.add_triple(subject, predicate, object_value)
                added += 1
        return added

    def count(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_value: Optional[str] = None,
    ) -> int:
        # Most specific -> least specific, using available pair/single stats.
        # If all three terms are bound and SPO tracking is disabled, we estimate
        # from the tightest available pair count.
        if subject is not None and predicate is not None and object_value is not None:
            if self.track_spo:
                return self.spo_counts.get((subject, predicate, object_value), 0)
            candidates = [
                self.sp_counts.get((subject, predicate)),
                self.po_counts.get((predicate, object_value)),
                self.so_counts.get((subject, object_value)),
            ]
            known = [value for value in candidates if value is not None]
            if known:
                return min(known)
            return 0

        if subject is not None and predicate is not None:
            return self.sp_counts.get((subject, predicate), 0)

        if predicate is not None and object_value is not None:
            return self.po_counts.get((predicate, object_value), 0)

        if subject is not None and object_value is not None:
            return self.so_counts.get((subject, object_value), 0)

        if subject is not None:
            return self.s_counts.get(subject, 0)

        if predicate is not None:
            return self.p_counts.get(predicate, 0)

        if object_value is not None:
            return self.o_counts.get(object_value, 0)

        return self.total_triples

    def summary(self) -> Dict[str, int]:
        return {
            "total_triples": self.total_triples,
            "unique_s": len(self.s_counts),
            "unique_p": len(self.p_counts),
            "unique_o": len(self.o_counts),
            "unique_sp": len(self.sp_counts),
            "unique_po": len(self.po_counts),
            "unique_so": len(self.so_counts),
            "unique_spo": len(self.spo_counts),
        }

    def to_bytes(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Catalog":
        loaded = _CompatUnpickler(io.BytesIO(data)).load()
        if isinstance(loaded, cls):
            # Backward compatibility for older pickle payloads.
            if not hasattr(loaded, "track_spo"):
                loaded.track_spo = True
            return loaded
        raise TypeError(f"Unsupported catalog payload type: {type(loaded)!r}")

    def save_pickle(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        target.write_bytes(self.to_bytes())
        return target

    @classmethod
    def load_pickle(cls, path: Union[str, Path]) -> "Catalog":
        source = Path(path)
        return cls.from_bytes(source.read_bytes())

    @classmethod
    def build_from_nt(cls, path: Union[str, Path]) -> "Catalog":
        catalog = cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                triple = parse_nt_triple_line(line)
                if triple:
                    catalog.add_triple(triple.subject, triple.predicate, triple.object_value)
        return catalog

    @staticmethod
    def iter_nt_records(path: Union[str, Path]) -> Iterator[TripleRecord]:
        with Path(path).open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                triple = parse_nt_triple_line(line)
                if triple:
                    yield triple


# Alias for UK spelling.
Catalogue = Catalog

