"""Disk-backed cache of raw component embeddings for multi-dim benchmark loads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from vector_endpoint.catalog import Catalog, parse_nt_triple_line
from vector_endpoint.component_fusion import (
    ComponentFusion,
    fuse_component_batch,
    stored_embedding_dim,
)


def meta_path_for_npz(npz_path: Path) -> Path:
    return Path(npz_path).with_name(f"{Path(npz_path).stem}_meta.json")


def component_text_for_object(obj: Optional[str], object_type: Optional[str]) -> Optional[str]:
    if not obj or obj == "":
        return None
    if object_type == "literal":
        return f"literal:{obj}"
    return obj


def collect_component_texts_from_triple(triple: Mapping[str, object]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return embed keys for subject, predicate, object (matches VectorDataBase._embed_triple_batch)."""
    subject = triple.get("subject")
    predicate = triple.get("predicate")
    obj = triple.get("object")
    object_type = triple.get("object_type")

    subj_text = subject if subject and subject != "" else None
    pred_text = predicate if predicate and predicate != "" else None
    obj_text = component_text_for_object(
        obj if isinstance(obj, str) else None,
        object_type if isinstance(object_type, str) else None,
    )
    return subj_text, pred_text, obj_text


def collect_component_texts_from_triples(triples: Sequence[Mapping[str, object]]) -> List[str]:
    """Collect unique component texts from triple dicts in stable order."""
    seen: set[str] = set()
    ordered: List[str] = []
    for triple in triples:
        for text in collect_component_texts_from_triple(triple):
            if text is not None and text not in seen:
                seen.add(text)
                ordered.append(text)
    return ordered


def iter_nt_lines(file_path: Path, max_lines: Optional[int] = None) -> Iterator[str]:
    yielded = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield line
            yielded += 1
            if max_lines is not None and yielded >= max_lines:
                return


def file_fingerprint(nt_path: Path) -> Dict[str, object]:
    resolved = nt_path.resolve()
    stat = resolved.stat()
    return {
        "input_file": str(resolved),
        "input_file_size": stat.st_size,
        "input_file_mtime": stat.st_mtime,
    }


def truncate_component(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.shape[0] < target_dim:
        raise ValueError(
            f"Cannot truncate cached vector dim {vec.shape[0]} to larger target dim {target_dim}"
        )
    return vec[:target_dim]


def normalize_component(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def adjust_component(vec: np.ndarray, target_dim: int, normalize: bool) -> np.ndarray:
    truncated = truncate_component(vec, target_dim)
    if normalize:
        return normalize_component(truncated)
    return truncated


@dataclass
class DiskEmbeddingCache:
    """Mapping from component text to raw (pre-truncate, pre-normalize) model vectors."""

    texts: List[str]
    embeddings: np.ndarray
    full_dim: int
    _index: Dict[str, int]

    @classmethod
    def from_dict(cls, text_to_vec: Mapping[str, np.ndarray]) -> "DiskEmbeddingCache":
        if not text_to_vec:
            return cls(texts=[], embeddings=np.zeros((0, 0), dtype=np.float32), full_dim=0, _index={})
        texts = sorted(text_to_vec.keys())
        sample = next(iter(text_to_vec.values()))
        full_dim = int(sample.shape[0])
        embeddings = np.stack([np.asarray(text_to_vec[t], dtype=np.float32) for t in texts], axis=0)
        index = {text: i for i, text in enumerate(texts)}
        return cls(texts=texts, embeddings=embeddings, full_dim=full_dim, _index=index)

    def lookup_raw(self, text: Optional[str]) -> Optional[np.ndarray]:
        if text is None or text == "":
            return None
        idx = self._index.get(text)
        if idx is None:
            return None
        return self.embeddings[idx]

    def put(self, text: str, raw_vec: np.ndarray) -> None:
        if text in self._index:
            return
        vec = np.asarray(raw_vec, dtype=np.float32).reshape(-1)
        if self.embeddings.size == 0:
            self.full_dim = int(vec.shape[0])
            self.texts = [text]
            self.embeddings = vec.reshape(1, -1)
            self._index = {text: 0}
            return
        if int(vec.shape[0]) != self.full_dim:
            raise ValueError(
                f"Cache full_dim={self.full_dim} but got vector with dim {vec.shape[0]} for {text!r}"
            )
        idx = len(self.texts)
        self.texts.append(text)
        self.embeddings = np.vstack([self.embeddings, vec.reshape(1, -1)])
        self._index[text] = idx

    def __len__(self) -> int:
        return len(self.texts)


def embed_triple_batch(
    triples: Sequence[Mapping[str, object]],
    cache: DiskEmbeddingCache,
    target_dim: int,
    normalize: bool = True,
    fusion: ComponentFusion = "concat",
) -> np.ndarray:
    """Build fused triple embeddings from cached raw component vectors."""
    out_dim = stored_embedding_dim(target_dim, fusion)
    if not triples:
        return np.zeros((0, out_dim), dtype=float)

    batch_size = len(triples)
    components = np.zeros((batch_size, 3, target_dim), dtype=float)
    present = np.zeros((batch_size, 3), dtype=bool)

    for i, triple in enumerate(triples):
        subj_text, pred_text, obj_text = collect_component_texts_from_triple(triple)
        for slot, text in enumerate((subj_text, pred_text, obj_text)):
            if text is None:
                continue
            raw = cache.lookup_raw(text)
            if raw is None:
                continue
            components[i, slot, :] = adjust_component(raw, target_dim, normalize)
            present[i, slot] = True

    fuse_normalize = normalize and fusion == "hadamard"
    return fuse_component_batch(
        components,
        present,
        fusion=fusion,
        normalize=fuse_normalize,
    )


def build_meta(
    *,
    nt_path: Path,
    embedding_model: str,
    dim_adjustment: str,
    cache_full_dim: int,
    unique_components: int,
) -> Dict[str, object]:
    meta = file_fingerprint(nt_path)
    meta.update(
        {
            "embedding_model": embedding_model,
            "dim_adjustment": dim_adjustment,
            "cache_full_dim": cache_full_dim,
            "unique_components": unique_components,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return meta


def validate_meta(
    meta: Mapping[str, object],
    *,
    nt_path: Path,
    embedding_model: str,
    dim_adjustment: str,
) -> None:
    resolved = str(nt_path.resolve())
    expected_file = meta.get("input_file")
    if expected_file != resolved:
        raise ValueError(
            f"Embedding cache input_file mismatch: cache={expected_file!r}, current={resolved!r}"
        )

    stat = nt_path.stat()
    expected_size = meta.get("input_file_size")
    if expected_size is not None and int(expected_size) != stat.st_size:
        raise ValueError(
            f"Embedding cache stale: NT file size changed ({expected_size} -> {stat.st_size})"
        )

    expected_mtime = meta.get("input_file_mtime")
    if expected_mtime is not None and float(expected_mtime) != stat.st_mtime:
        raise ValueError(
            f"Embedding cache stale: NT file mtime changed ({expected_mtime} -> {stat.st_mtime})"
        )

    if meta.get("embedding_model") != embedding_model:
        raise ValueError(
            f"Embedding cache model mismatch: cache={meta.get('embedding_model')!r}, "
            f"current={embedding_model!r}"
        )

    if meta.get("dim_adjustment") != dim_adjustment:
        raise ValueError(
            f"Embedding cache dim_adjustment mismatch: cache={meta.get('dim_adjustment')!r}, "
            f"current={dim_adjustment!r}"
        )


def save_cache(npz_path: Path, cache: DiskEmbeddingCache, meta: Mapping[str, object]) -> Path:
    npz_path = Path(npz_path)
    meta_path = meta_path_for_npz(npz_path)
    if npz_path.parent and not npz_path.parent.exists():
        npz_path.parent.mkdir(parents=True, exist_ok=True)

    texts_array = np.asarray(cache.texts, dtype=object)
    np.savez_compressed(
        npz_path,
        texts=texts_array,
        embeddings=np.asarray(cache.embeddings, dtype=np.float32),
        full_dim=np.int32(cache.full_dim),
    )
    meta_path.write_text(json.dumps(dict(meta), indent=2), encoding="utf-8")
    return meta_path


def load_cache(npz_path: Path, meta_path: Optional[Path] = None) -> Tuple[DiskEmbeddingCache, Dict[str, object]]:
    npz_path = Path(npz_path)
    if meta_path is None:
        meta_path = meta_path_for_npz(npz_path)
    else:
        meta_path = Path(meta_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Embedding cache meta not found: {meta_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        texts = [str(t) for t in data["texts"].tolist()]
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        full_dim = int(data["full_dim"])

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    index = {text: i for i, text in enumerate(texts)}
    cache = DiskEmbeddingCache(texts=texts, embeddings=embeddings, full_dim=full_dim, _index=index)
    return cache, meta


def collect_unique_component_texts_from_nt(
    nt_path: Path,
    max_lines: Optional[int] = None,
) -> Tuple[List[str], Catalog]:
    """Scan NT file and return unique component texts plus a built catalog."""
    from vector_endpoint.db.VectorDataBase import VectorDataBase

    catalog = Catalog(track_spo=False)
    seen: set[str] = set()
    ordered: List[str] = []

    for line in iter_nt_lines(nt_path, max_lines=max_lines):
        parsed = VectorDataBase._parse_triple_line(line)
        if parsed:
            catalog.add_batch([(parsed["subject"], parsed["predicate"], parsed["object"])])
            for text in collect_component_texts_from_triple(parsed):
                if text is not None and text not in seen:
                    seen.add(text)
                    ordered.append(text)
        else:
            triple_record = parse_nt_triple_line(line)
            if triple_record:
                catalog.add_batch(
                    [(triple_record.subject, triple_record.predicate, triple_record.object_value)]
                )

    return ordered, catalog


def encode_unique_components(vdb, unique_texts: Sequence[str], batch_size: int = 32) -> DiskEmbeddingCache:
    """Encode unique component strings at full model dim without normalization."""
    cache = DiskEmbeddingCache.from_dict({})
    if not unique_texts:
        return cache

    for start in range(0, len(unique_texts), batch_size):
        batch = list(unique_texts[start : start + batch_size])
        raw = vdb._encode_text_batch(batch, normalize=False)
        for text, vec in zip(batch, raw):
            cache.put(text, vec)
    return cache


def build_cache_from_nt(
    nt_path: Path,
    vdb,
    *,
    max_lines: Optional[int] = None,
    embedding_model: str,
    dim_adjustment: str,
) -> Tuple[DiskEmbeddingCache, Catalog, Dict[str, object]]:
    unique_texts, catalog = collect_unique_component_texts_from_nt(nt_path, max_lines=max_lines)
    cache = encode_unique_components(vdb, unique_texts)
    meta = build_meta(
        nt_path=nt_path,
        embedding_model=embedding_model,
        dim_adjustment=dim_adjustment,
        cache_full_dim=cache.full_dim,
        unique_components=len(cache),
    )
    return cache, catalog, meta
