#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running this script from string_match_benchmark/ while importing project src modules.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_k import CatalogKResolver, milvus_safe_k
from catalog import parse_nt_triple_line
from vector_endpoint.db.VectorDataBase import VectorDataBase


@dataclass(frozen=True)
class QueryPattern:
    subject: Optional[str]
    predicate: Optional[str]
    object_value: Optional[str]
    object_type: Optional[str]  # "literal", "uri", or None


@dataclass(frozen=True)
class CandidateTriple:
    subject: str
    predicate: str
    object_value: str
    object_type: str  # "literal" or "uri"
    text: str


DEFAULT_QUERY = (
    "SELECT ?X WHERE {"
    " ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
    " <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> }"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark post-filter timing: string part-match vs embedding part-match."
    )
    parser.add_argument("--collection", default="version_5", help="Milvus collection name")
    parser.add_argument("--database-name", default="lubm_db", help="Database label for VectorDataBase")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--target-embedding-dim", type=int, default=8, help="Model component dimension")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Simple SPARQL query to benchmark")
    parser.add_argument("--k", type=int, default=None, help="Top-k candidates retrieved before post-filter (default: catalog auto-k)")
    parser.add_argument(
        "--catalog-path",
        default=str(Path(__file__).resolve().parents[1] / "catalog.pkl"),
        help="Catalog pickle path used for auto-k when --k is omitted.",
    )
    parser.add_argument("--runs", type=int, default=30, help="Repetitions per matcher")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.999,
        help="Cosine threshold for embedding part-equality checks",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory for raw benchmark outputs",
    )
    parser.add_argument("--log", action="store_true", help="Verbose logs")
    return parser.parse_args()


def _extract_where_body(query: str) -> str:
    upper = query.upper()
    where_idx = upper.find("WHERE")
    if where_idx < 0:
        raise ValueError("Query must contain WHERE clause.")

    open_idx = query.find("{", where_idx)
    close_idx = query.rfind("}")
    if open_idx < 0 or close_idx < 0 or close_idx <= open_idx:
        raise ValueError("Query WHERE clause must include {...}.")

    return query[open_idx + 1 : close_idx].strip()


def _tokenize_triple_pattern(where_body: str) -> List[str]:
    # Supports ?vars, <URI>, and "literal".
    import re

    token_pattern = r'(<[^>]+>|"[^"]*"|\?[A-Za-z_][A-Za-z0-9_]*)'
    return re.findall(token_pattern, where_body)


def parse_query_pattern(query: str) -> QueryPattern:
    where_body = _extract_where_body(query)
    tokens = _tokenize_triple_pattern(where_body)
    if len(tokens) < 3:
        raise ValueError(f"Could not parse triple pattern from query body: {where_body}")

    s_tok, p_tok, o_tok = tokens[0], tokens[1], tokens[2]

    subject = None if s_tok.startswith("?") else s_tok
    predicate = None if p_tok.startswith("?") else p_tok

    object_value: Optional[str]
    object_type: Optional[str]
    if o_tok.startswith("?"):
        object_value, object_type = None, None
    elif o_tok.startswith('"') and o_tok.endswith('"'):
        object_value, object_type = o_tok[1:-1], "literal"
    else:
        object_value, object_type = o_tok, "uri"

    return QueryPattern(
        subject=subject,
        predicate=predicate,
        object_value=object_value,
        object_type=object_type,
    )


def fetch_candidates(vdb: VectorDataBase, collection_name: str, query: str, k: int, log: bool = False) -> List[CandidateTriple]:
    results = vdb.search(
        collection_name=collection_name,
        query_texts=query,
        limit=k,
        output_fields=["text"],
        log=log,
    )
    if not results:
        return []

    matches = results[0].get("matches", [])
    candidates: List[CandidateTriple] = []
    for match in matches:
        text = match.get("text")
        parsed = parse_nt_triple_line(text)
        if not parsed:
            continue
        object_type = "uri" if parsed.object_value.startswith("<") and parsed.object_value.endswith(">") else "literal"
        candidates.append(
            CandidateTriple(
                subject=parsed.subject,
                predicate=parsed.predicate,
                object_value=parsed.object_value,
                object_type=object_type,
                text=text,
            )
        )
    return candidates


def string_part_match(pattern: QueryPattern, candidate: CandidateTriple) -> bool:
    if pattern.subject is not None and candidate.subject != pattern.subject:
        return False
    if pattern.predicate is not None and candidate.predicate != pattern.predicate:
        return False
    if pattern.object_value is not None and candidate.object_value != pattern.object_value:
        return False
    if pattern.object_type is not None and candidate.object_type != pattern.object_type:
        return False
    return True


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Both vectors are normalized from _encode_text_batch(normalize=True).
    return float(np.dot(a, b))


def _build_part_embeddings(
    vdb: VectorDataBase,
    pattern: QueryPattern,
    candidates: Sequence[CandidateTriple],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    query_vectors: Dict[str, np.ndarray] = {}

    if pattern.subject is not None:
        query_vectors["subject"] = vdb._encode_text_batch([pattern.subject], normalize=True)[0]
    if pattern.predicate is not None:
        query_vectors["predicate"] = vdb._encode_text_batch([pattern.predicate], normalize=True)[0]
    if pattern.object_value is not None:
        object_text = (
            f"literal:{pattern.object_value}" if pattern.object_type == "literal" else pattern.object_value
        )
        query_vectors["object"] = vdb._encode_text_batch([object_text], normalize=True)[0]

    subject_texts = [c.subject for c in candidates]
    predicate_texts = [c.predicate for c in candidates]
    object_texts = [
        f"literal:{c.object_value}" if c.object_type == "literal" else c.object_value
        for c in candidates
    ]

    candidate_vectors = {
        "subject": vdb._encode_text_batch(subject_texts, normalize=True) if subject_texts else np.zeros((0, vdb._embedding_dim)),
        "predicate": vdb._encode_text_batch(predicate_texts, normalize=True) if predicate_texts else np.zeros((0, vdb._embedding_dim)),
        "object": vdb._encode_text_batch(object_texts, normalize=True) if object_texts else np.zeros((0, vdb._embedding_dim)),
    }
    return query_vectors, candidate_vectors


def embedding_part_match(
    pattern: QueryPattern,
    candidate: CandidateTriple,
    query_vectors: Dict[str, np.ndarray],
    candidate_vectors: Dict[str, np.ndarray],
    candidate_index: int,
    similarity_threshold: float,
) -> bool:
    if pattern.subject is not None:
        sim = _cosine_similarity(query_vectors["subject"], candidate_vectors["subject"][candidate_index])
        if sim < similarity_threshold:
            return False

    if pattern.predicate is not None:
        sim = _cosine_similarity(query_vectors["predicate"], candidate_vectors["predicate"][candidate_index])
        if sim < similarity_threshold:
            return False

    if pattern.object_value is not None:
        # Keep hard object-type guard to ensure literal-vs-uri consistency.
        if pattern.object_type is not None and candidate.object_type != pattern.object_type:
            return False
        sim = _cosine_similarity(query_vectors["object"], candidate_vectors["object"][candidate_index])
        if sim < similarity_threshold:
            return False

    return True


def _ci95(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1))
    return 1.96 * std / math.sqrt(len(values))


def main() -> int:
    args = parse_args()

    vdb = VectorDataBase(
        database_name=args.database_name,
        host=args.host,
        port=args.port,
        embedding_model=args.embedding_model,
        target_embedding_dim=args.target_embedding_dim,
    )
    vdb.connect()

    pattern = parse_query_pattern(args.query)
    k_resolver = CatalogKResolver(catalog_path=Path(args.catalog_path)) if args.k is None else None
    effective_k = args.k
    if effective_k is None and k_resolver is not None:
        effective_k = k_resolver.auto_k_for_pattern(
            subject=pattern.subject,
            predicate=pattern.predicate,
            object_value=pattern.object_value,
            object_type=pattern.object_type,
        )
    if effective_k is None:
        effective_k = 1000
    effective_k = milvus_safe_k(effective_k)

    candidates = fetch_candidates(vdb, args.collection, args.query, effective_k, log=args.log)

    if not candidates:
        raise RuntimeError("No candidates returned from vector search; benchmark cannot proceed.")

    query_vectors, candidate_vectors = _build_part_embeddings(vdb, pattern, candidates)

    # Warm-up (not recorded) to reduce one-time noise.
    _ = [c for c in candidates if string_part_match(pattern, c)]
    _ = [
        c
        for i, c in enumerate(candidates)
        if embedding_part_match(
            pattern,
            c,
            query_vectors,
            candidate_vectors,
            i,
            args.similarity_threshold,
        )
    ]

    run_rows: List[Dict[str, float]] = []
    for run_idx in range(1, args.runs + 1):
        t0 = perf_counter_ns()
        string_hits = [c for c in candidates if string_part_match(pattern, c)]
        t1 = perf_counter_ns()

        embedding_hits = [
            c
            for i, c in enumerate(candidates)
            if embedding_part_match(
                pattern,
                c,
                query_vectors,
                candidate_vectors,
                i,
                args.similarity_threshold,
            )
        ]
        t2 = perf_counter_ns()

        string_ms = (t1 - t0) / 1_000_000.0
        embedding_ms = (t2 - t1) / 1_000_000.0
        run_rows.append(
            {
                "run": run_idx,
                "string_ms": string_ms,
                "embedding_ms": embedding_ms,
                "string_hits": float(len(string_hits)),
                "embedding_hits": float(len(embedding_hits)),
            }
        )

        print(
            f"Run {run_idx:02d}/{args.runs} | "
            f"string={string_ms:.4f} ms ({len(string_hits)} hits) | "
            f"embedding={embedding_ms:.4f} ms ({len(embedding_hits)} hits)"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"string_vs_embedding_{ts}"

    json_path = out_dir / f"{base_name}.json"
    csv_path = out_dir / f"{base_name}.csv"

    string_values = [r["string_ms"] for r in run_rows]
    embedding_values = [r["embedding_ms"] for r in run_rows]

    payload = {
        "config": {
            "collection": args.collection,
            "query": args.query,
            "k": effective_k,
            "runs": args.runs,
            "similarity_threshold": args.similarity_threshold,
            "candidate_count": len(candidates),
        },
        "summary": {
            "string_mean_ms": float(np.mean(string_values)),
            "string_ci95_ms": _ci95(string_values),
            "embedding_mean_ms": float(np.mean(embedding_values)),
            "embedding_ci95_ms": _ci95(embedding_values),
        },
        "runs": run_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "string_ms", "embedding_ms", "string_hits", "embedding_hits"],
        )
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)

    print("\nBenchmark complete.")
    print(f"Candidates benchmarked: {len(candidates)}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
