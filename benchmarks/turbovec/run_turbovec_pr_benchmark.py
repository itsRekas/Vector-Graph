#!/usr/bin/env python3
"""TurboVec/TurboQuant precision-recall benchmark vs the Milvus HNSW baseline.

Replaces only the index/search stage: embeddings come from the same
VectorDataBase path as production, and scoring reuses the benchmarks/pr
pipeline (exact string post-filter + comunica-sparql-file ground truth) so the
numbers are directly comparable to the stored Milvus dim-8 result.

Fixed k=10 throughout (matches the stored Milvus run's per-query seed_k=10).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# Reuse the pr scoring pipeline. The pr folder is not a package, so add it to
# sys.path and import the module by name.
PR_DIR = Path(__file__).resolve().parents[1] / "pr"
if str(PR_DIR) not in sys.path:
    sys.path.insert(0, str(PR_DIR))

from run_vector_dim_accuracy_benchmark import (  # type: ignore  # noqa: E402
    QueryPattern,
    _matches_to_bindings,
    binding_to_id,
    jaccard,
    load_queries,
    parse_query_pattern,
    precision_recall,
    run_sparql_baseline,
)

from vector_endpoint.db.VectorDataBase import VectorDataBase  # noqa: E402
from vector_endpoint.load import iter_nt_lines  # noqa: E402

from turbovec import TurboQuantIndex  # noqa: E402


# Stored Milvus HNSW dim-8 baseline (k=10) for the comparison table.
MILVUS_REFERENCE = {
    "index": "milvus_hnsw",
    "dimension": 8,
    "k": 10,
    "avg_precision": 0.9473333333333334,
    "avg_recall": 0.934781984389565,
    "source": "../pr/results/vector_dim_pr_20260529T140118Z_summary.csv",
}


@dataclass(frozen=True)
class BitResult:
    bit_width: int
    dimension: int
    footprint_bytes: float
    avg_precision: float
    avg_recall: float
    mean_jaccard: float
    passes_threshold: bool
    bucket_metrics: Dict[str, Dict[str, float]]
    queries: List[dict]
    build_seconds: float
    search_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure TurboVec/TurboQuant precision/recall at dim=8 for one or "
            "more bit-widths, comparable to the Milvus HNSW dim-8 baseline."
        )
    )
    parser.add_argument("--input-file", required=True, help="Path to the .nt corpus file")
    parser.add_argument("--rdf-file", required=True, help="Path to .nt file for the SPARQL baseline")
    parser.add_argument("--queries-file", required=True, help="JSON query set (pr format)")
    parser.add_argument("--dimension", type=int, default=8, help="Per-component embedding dim")
    parser.add_argument("--bit-widths", default="2,4", help="Comma-separated TurboQuant bit-widths")
    parser.add_argument("--k", type=int, default=10, help="Fixed top-k for search and baseline LIMIT")
    parser.add_argument("--accuracy-threshold-pct", type=float, default=95.0, help="Pass threshold")
    parser.add_argument("--database-name", default="lubm_db", help="Database label")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--dim-adjustment", default="truncate", choices=["truncate"], help="Dim mode")
    parser.add_argument("--chunk-size", type=int, default=512, help="Embedding chunk size")
    parser.add_argument("--max-lines", type=int, default=None, help="Optional corpus cap for smoke runs")
    parser.add_argument("--out-dir", default="results", help="Output directory")
    parser.add_argument("--log", action="store_true", help="Verbose logs")
    return parser.parse_args()


def parse_bit_widths(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        bit = int(token)
        if bit not in (2, 4):
            raise ValueError(f"turbovec supports bit_width 2 or 4, got {bit}")
        values.append(bit)
    if not values:
        raise ValueError("No bit-widths provided")
    return values


def build_corpus(
    vdb: VectorDataBase,
    input_file: Path,
    chunk_size: int,
    max_lines: Optional[int],
    log: bool,
) -> Tuple[np.ndarray, List[str]]:
    """Embed every triple once; return (vectors [N, 3*dim] float32, texts)."""
    vectors_parts: List[np.ndarray] = []
    texts: List[str] = []
    chunk: List[str] = []
    total = 0

    def flush(chunk_lines: List[str]) -> None:
        nonlocal total
        if not chunk_lines:
            return
        records = [
            vdb._normalize_triple_record(vdb._parse_triple_line(line), line)
            for line in chunk_lines
        ]
        embeddings = vdb._embed_triple_batch(records, normalize=True)
        vectors_parts.append(np.asarray(embeddings, dtype=np.float32))
        texts.extend(record["text"] for record in records)
        total += len(chunk_lines)
        if log:
            print(f"  embedded {total} triples...", flush=True)

    for line in iter_nt_lines(input_file, max_lines=max_lines):
        chunk.append(line)
        if len(chunk) >= chunk_size:
            flush(chunk)
            chunk = []
    flush(chunk)

    if not vectors_parts:
        raise RuntimeError("No triples embedded; check the input file.")

    vectors = np.ascontiguousarray(np.vstack(vectors_parts), dtype=np.float32)
    return vectors, texts


def embed_queries(
    vdb: VectorDataBase,
    patterns: Sequence[QueryPattern],
) -> np.ndarray:
    """Embed each query pattern into the concatenated S|P|O vector space."""
    query_triples = [
        {
            "subject": p.subject,
            "predicate": p.predicate,
            "object": p.object_value,
            "object_type": p.object_type,
        }
        for p in patterns
    ]
    matrix = vdb._embed_triple_batch(query_triples, normalize=True)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def compute_ground_truth(
    queries: Sequence[Tuple[str, str, str]],
    rdf_file: str,
    k: int,
    log: bool,
) -> Dict[str, Set[str]]:
    """Run the SPARQL baseline once per query and cache the binding-id sets."""
    cache: Dict[str, Set[str]] = {}
    for idx, (qid, _bucket, query) in enumerate(queries, start=1):
        baseline_bindings = run_sparql_baseline(query, rdf_file, k)
        cache[qid] = {binding_to_id(b) for b in baseline_bindings}
        if log and (idx <= 5 or idx % 250 == 0):
            print(f"  ground truth {idx}/{len(queries)} ({qid}): |GT|={len(cache[qid])}", flush=True)
    return cache


def evaluate_bit_width(
    *,
    bit_width: int,
    dimension: int,
    vectors: np.ndarray,
    texts: Sequence[str],
    queries: Sequence[Tuple[str, str, str]],
    patterns: Sequence[QueryPattern],
    query_matrix: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    k: int,
    threshold_pct: float,
    log: bool,
) -> BitResult:
    dim_total = vectors.shape[1]

    t0 = perf_counter()
    index = TurboQuantIndex(dim=dim_total, bit_width=bit_width)
    index.add(vectors)
    build_seconds = perf_counter() - t0

    t1 = perf_counter()
    scores, ids = index.search(query_matrix, k=k)
    search_seconds = perf_counter() - t1
    ids = np.asarray(ids)
    scores = np.asarray(scores)

    precision_vals: List[float] = []
    recall_vals: List[float] = []
    jaccard_vals: List[float] = []
    bucket_precision: Dict[str, List[float]] = {}
    bucket_recall: Dict[str, List[float]] = {}
    query_rows: List[dict] = []

    n_vectors = len(texts)
    for q_idx, (qid, bucket, query) in enumerate(queries):
        pattern = patterns[q_idx]
        matches: List[dict] = []
        for rank in range(ids.shape[1]):
            row_id = int(ids[q_idx, rank])
            if row_id < 0 or row_id >= n_vectors:
                continue  # padding slot when fewer than k results
            matches.append(
                {
                    "id": row_id,
                    "distance": float(scores[q_idx, rank]),
                    "text": texts[row_id],
                }
            )

        bindings, _ids = _matches_to_bindings(matches, pattern)
        vector_ids = {binding_to_id(b) for b in bindings}
        baseline_ids = ground_truth.get(qid, set())

        tp = len(baseline_ids & vector_ids)
        fp = len(vector_ids - baseline_ids)
        fn = len(baseline_ids - vector_ids)
        precision, recall = precision_recall(tp, fp, fn)
        overlap_jaccard = jaccard(baseline_ids, vector_ids)
        exact_match = baseline_ids == vector_ids

        precision_vals.append(precision)
        recall_vals.append(recall)
        jaccard_vals.append(overlap_jaccard)
        bucket_precision.setdefault(bucket, []).append(precision)
        bucket_recall.setdefault(bucket, []).append(recall)

        query_rows.append(
            {
                "query_id": qid,
                "bucket": bucket,
                "query": query,
                "baseline_count": len(baseline_ids),
                "vector_count": len(vector_ids),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "jaccard": overlap_jaccard,
                "exact_match": exact_match,
                "seed_k": k,
            }
        )
        if log and (q_idx + 1 <= 10 or (q_idx + 1) % 250 == 0):
            print(
                f"  [bit={bit_width}] {qid} ({bucket}): |GT|={len(baseline_ids)} "
                f"|RET|={len(vector_ids)} TP={tp} FP={fp} FN={fn} "
                f"P={precision:.4f} R={recall:.4f}",
                flush=True,
            )

    avg_precision = float(np.mean(precision_vals)) if precision_vals else 0.0
    avg_recall = float(np.mean(recall_vals)) if recall_vals else 0.0
    mean_jaccard = float(np.mean(jaccard_vals)) if jaccard_vals else 0.0
    threshold = threshold_pct / 100.0
    passes = avg_precision >= threshold and avg_recall >= threshold

    bucket_metrics: Dict[str, Dict[str, float]] = {}
    for bucket_name in sorted(set(bucket_precision) | set(bucket_recall)):
        p_vals = bucket_precision.get(bucket_name, [])
        r_vals = bucket_recall.get(bucket_name, [])
        bucket_metrics[bucket_name] = {
            "count": float(len(p_vals)),
            "avg_precision": float(np.mean(p_vals)) if p_vals else 0.0,
            "avg_recall": float(np.mean(r_vals)) if r_vals else 0.0,
        }

    # TurboQuant footprint: bit-packed codes + per-vector norm and renorm scalar.
    footprint_bytes = dim_total * bit_width / 8.0 + 8.0

    return BitResult(
        bit_width=bit_width,
        dimension=dimension,
        footprint_bytes=footprint_bytes,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        mean_jaccard=mean_jaccard,
        passes_threshold=passes,
        bucket_metrics=bucket_metrics,
        queries=query_rows,
        build_seconds=build_seconds,
        search_seconds=search_seconds,
    )


def write_outputs(
    payload: dict,
    results: Sequence[BitResult],
    out_dir: Path,
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = payload["timestamp_utc"]
    base = out_dir / f"turbovec_pr_{ts}"
    json_path = base.with_suffix(".json")
    summary_csv = base.with_name(f"{base.name}_summary.csv")
    per_query_csv = base.with_name(f"{base.name}_per_query.csv")

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_fields = [
        "bit_width",
        "dimension",
        "footprint_bytes",
        "avg_precision",
        "avg_recall",
        "avg_precision_pct",
        "avg_recall_pct",
        "mean_jaccard",
        "queries_total",
        "passes_threshold",
        "sp*_avg_precision",
        "sp*_avg_recall",
        "*po_avg_precision",
        "*po_avg_recall",
        "s*o_avg_precision",
        "s*o_avg_recall",
        "build_seconds",
        "search_seconds",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for r in results:
            bm = r.bucket_metrics
            writer.writerow(
                {
                    "bit_width": r.bit_width,
                    "dimension": r.dimension,
                    "footprint_bytes": r.footprint_bytes,
                    "avg_precision": r.avg_precision,
                    "avg_recall": r.avg_recall,
                    "avg_precision_pct": r.avg_precision * 100.0,
                    "avg_recall_pct": r.avg_recall * 100.0,
                    "mean_jaccard": r.mean_jaccard,
                    "queries_total": len(r.queries),
                    "passes_threshold": r.passes_threshold,
                    "sp*_avg_precision": bm.get("sp*", {}).get("avg_precision"),
                    "sp*_avg_recall": bm.get("sp*", {}).get("avg_recall"),
                    "*po_avg_precision": bm.get("*po", {}).get("avg_precision"),
                    "*po_avg_recall": bm.get("*po", {}).get("avg_recall"),
                    "s*o_avg_precision": bm.get("s*o", {}).get("avg_precision"),
                    "s*o_avg_recall": bm.get("s*o", {}).get("avg_recall"),
                    "build_seconds": r.build_seconds,
                    "search_seconds": r.search_seconds,
                }
            )

    per_query_fields = [
        "bit_width",
        "dimension",
        "query_id",
        "bucket",
        "baseline_count",
        "vector_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "jaccard",
        "exact_match",
        "seed_k",
    ]
    with per_query_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_query_fields)
        writer.writeheader()
        for r in results:
            for q in r.queries:
                writer.writerow(
                    {
                        "bit_width": r.bit_width,
                        "dimension": r.dimension,
                        "query_id": q["query_id"],
                        "bucket": q["bucket"],
                        "baseline_count": q["baseline_count"],
                        "vector_count": q["vector_count"],
                        "tp": q["tp"],
                        "fp": q["fp"],
                        "fn": q["fn"],
                        "precision": q["precision"],
                        "recall": q["recall"],
                        "jaccard": q["jaccard"],
                        "exact_match": q["exact_match"],
                        "seed_k": q["seed_k"],
                    }
                )

    return json_path, summary_csv, per_query_csv


def main() -> int:
    args = parse_args()
    bit_widths = parse_bit_widths(args.bit_widths)
    out_dir = Path(args.out_dir)

    vdb = VectorDataBase(
        database_name=args.database_name,
        host="localhost",
        port=19530,
        embedding_model=args.embedding_model,
        target_embedding_dim=args.dimension,
        dim_adjustment=args.dim_adjustment,
    )

    print(f"Embedding corpus from {args.input_file} (dim={args.dimension} -> {3 * args.dimension}-d vectors)...")
    t_embed = perf_counter()
    vectors, texts = build_corpus(
        vdb, Path(args.input_file), args.chunk_size, args.max_lines, args.log
    )
    print(
        f"Embedded {len(texts)} triples into {vectors.shape} float32 vectors "
        f"in {perf_counter() - t_embed:.1f}s"
    )

    queries = load_queries(args.queries_file)
    patterns = [parse_query_pattern(query) for (_qid, _bucket, query) in queries]
    print(f"Loaded {len(queries)} queries.")

    query_matrix = embed_queries(vdb, patterns)

    print(f"Computing SPARQL ground truth once (k={args.k})...")
    t_gt = perf_counter()
    ground_truth = compute_ground_truth(queries, args.rdf_file, args.k, args.log)
    print(f"Ground truth computed in {perf_counter() - t_gt:.1f}s")

    results: List[BitResult] = []
    for bit_width in bit_widths:
        print(f"\n=== TurboVec bit_width={bit_width} (dim={3 * args.dimension}) ===")
        result = evaluate_bit_width(
            bit_width=bit_width,
            dimension=args.dimension,
            vectors=vectors,
            texts=texts,
            queries=queries,
            patterns=patterns,
            query_matrix=query_matrix,
            ground_truth=ground_truth,
            k=args.k,
            threshold_pct=args.accuracy_threshold_pct,
            log=args.log,
        )
        results.append(result)
        print(
            f"bit={bit_width}: avg_precision={result.avg_precision * 100:.2f}% "
            f"avg_recall={result.avg_recall * 100:.2f}% "
            f"footprint={result.footprint_bytes:.0f}B "
            f"(build={result.build_seconds:.2f}s search={result.search_seconds:.2f}s) "
            f"passes={result.passes_threshold}"
        )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "benchmark": "turbovec_pr",
        "dimension": args.dimension,
        "vector_dim": 3 * args.dimension,
        "k": args.k,
        "metric_primary": "precision_recall",
        "accuracy_threshold_pct": args.accuracy_threshold_pct,
        "bit_widths_tested": bit_widths,
        "milvus_reference": MILVUS_REFERENCE,
        "corpus": {
            "input_file": args.input_file,
            "num_vectors": len(texts),
        },
        # Keyed "per_dimension" so the pr plotter can also read this file; each
        # entry is one bit-width at the fixed dimension.
        "per_dimension": [
            {
                "dimension": r.dimension,
                "bit_width": r.bit_width,
                "footprint_bytes": r.footprint_bytes,
                "queries_total": len(r.queries),
                "avg_precision": r.avg_precision,
                "avg_recall": r.avg_recall,
                "avg_precision_pct": r.avg_precision * 100.0,
                "avg_recall_pct": r.avg_recall * 100.0,
                "mean_jaccard": r.mean_jaccard,
                "passes_threshold": r.passes_threshold,
                "bucket_metrics": r.bucket_metrics,
                "build_seconds": r.build_seconds,
                "search_seconds": r.search_seconds,
                "queries": r.queries,
            }
            for r in results
        ],
    }

    json_path, summary_csv, per_query_csv = write_outputs(payload, results, out_dir)

    print("\n=== Comparison (k={}) ===".format(args.k))
    print(
        f"  Milvus HNSW (stored): precision={MILVUS_REFERENCE['avg_precision'] * 100:.2f}% "
        f"recall={MILVUS_REFERENCE['avg_recall'] * 100:.2f}%"
    )
    for r in results:
        print(
            f"  TurboVec {r.bit_width}-bit:      precision={r.avg_precision * 100:.2f}% "
            f"recall={r.avg_recall * 100:.2f}%  (footprint {r.footprint_bytes:.0f}B/vec)"
        )
    print(f"\nJSON:        {json_path}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-query:   {per_query_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
