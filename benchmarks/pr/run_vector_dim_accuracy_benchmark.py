#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from time import perf_counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from vector_endpoint.auto_k import CatalogKResolver, milvus_safe_k, resolve_pagination_limit
from vector_endpoint.pattern_query import PatternQueryInput
from vector_endpoint.pagination_search import collect_pagination_pages
from vector_endpoint.catalog import parse_nt_triple_line
from vector_endpoint.db.VectorDataBase import VectorDataBase
from vector_endpoint.adaptive_exp import adaptive_batch_search, build_k_ladder

from grpc_pattern_client import (
    pattern_request_body_from_query,
    query_pattern_grpc,
    query_pattern_pagination_grpc,
)


DEFAULT_DIMS = "8"
DEFAULT_QUERIES = [
    (
        "Q1",
        "manual",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> }",
    ),
    (
        "Q2",
        "manual",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateStudent> }",
    ),
    (
        "Q3",
        "manual",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#Professor> }",
    ),
    (
        "Q4",
        "manual",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#Course> }",
    ),
    (
        "Q5",
        "manual",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> }",
    ),
    (
        "Q6",
        "manual",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#ResearchGroup> }",
    ),
]


@dataclass(frozen=True)
class QueryPattern:
    subject: Optional[str]
    predicate: Optional[str]
    object_value: Optional[str]
    object_type: Optional[str]
    select_vars: List[str]


@dataclass(frozen=True)
class CandidateTriple:
    subject: str
    predicate: str
    object_value: str
    object_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark vector dim accuracy versus SPARQL baseline using count + overlap. "
            "Can optionally run load phase before each tested dim."
        )
    )
    parser.add_argument("--collection", default="dim_benchmark", help="Benchmark collection (must be dim_benchmark)")
    parser.add_argument("--dimensions", default=DEFAULT_DIMS, help="Comma-separated dimension list")
    parser.add_argument("--accuracy-threshold-pct", type=float, default=95.0, help="Dimension pass threshold for avg precision and avg recall")
    parser.add_argument("--k", type=int, default=None, help="Top-k vector and baseline result limit (default: catalog auto-k)")
    parser.add_argument(
        "--catalog-path",
        default=str(Path(__file__).resolve().parents[2] / "catalog.pkl"),
        help="Catalog pickle path used for auto-k when --k is omitted.",
    )
    parser.add_argument(
        "--catalog-k-scale",
        type=float,
        default=1.2,
        help="seed_k = ceil(catalog_count * scale) per pattern (default 1.2).",
    )
    parser.add_argument(
        "--catalog-min-k",
        type=int,
        default=10,
        help="Minimum catalog seed k per pattern.",
    )
    parser.add_argument(
        "--grpc-endpoint",
        default=None,
        help="gRPC target for vector search (e.g. 127.0.0.1:50051 or grpc://127.0.0.1:50051).",
    )
    parser.add_argument("--database-name", default="lubm_db", help="Database label")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument(
        "--dim-adjustment",
        default="truncate",
        choices=["truncate"],
        help="How to adapt model output to target dim (currently only truncate).",
    )
    parser.add_argument(
        "--component-fusion",
        default="concat",
        choices=["concat", "hadamard"],
        help="Fuse S|P|O embeddings: concat (3d stored) or hadamard (d stored)",
    )
    parser.add_argument("--rdf-file", default="data/nts/RLUBM_cleaned.nt", help="Path to RDF NT file for baseline")
    parser.add_argument(
        "--queries-file",
        default=None,
        help="Optional JSON file with [{\"id\":\"Qx\",\"query\":\"SELECT ...\"}]",
    )
    parser.add_argument("--run-load-phase", action="store_true", help="Run load pipeline for each dim first")
    parser.add_argument("--load-input-file", default="data/nts/RLUBM_cleaned.nt", help="NT file passed to load phase")
    parser.add_argument("--load-chunk-size", type=int, default=100, help="Chunk size for load phase")
    parser.add_argument("--load-max-lines", type=int, default=None, help="Optional load max-lines for quick runs")
    parser.add_argument("--out-dir", default="results", help="Output directory")
    parser.add_argument("--log", action="store_true", help="Verbose logs")
    parser.add_argument(
        "--use-adaptive",
        action="store_true",
        help=(
            "Use adaptive_exp.adaptive_batch_search instead of a single fixed-k vdb.search. "
            "Seed k is the catalog auto-k (or --k as fallback); ladder is seed*multipliers."
        ),
    )
    parser.add_argument(
        "--adaptive-multipliers",
        default="1,10,100,1000",
        help="Comma-separated multipliers applied to seed_k for the ladder.",
    )
    parser.add_argument(
        "--adaptive-jaccard",
        type=float,
        default=0.99,
        help="Jaccard near-stability threshold for adaptive escalation (default 0.99).",
    )
    parser.add_argument(
        "--use-pagination",
        action="store_true",
        help=(
            "Use Milvus search_iterator pagination (client-driven pages via gRPC QueryPatternPage). "
            "Requires --k as page batch size; total cap defaults to 2 * catalog_k per query."
        ),
    )
    parser.add_argument(
        "--pagination-limit",
        type=int,
        default=None,
        help="Optional total Milvus hit cap for pagination (default 2 * catalog_k per query).",
    )
    parser.add_argument(
        "--latency-warmup-queries",
        type=int,
        default=1,
        help=(
            "Untimed vector fetches per dimension before benchmarking (default 1). "
            "Excludes gRPC/model cold-start from avg_vector_query_seconds without "
            "dropping any timed queries from P/R."
        ),
    )
    return parser.parse_args()


def require_safe_collection(collection: str) -> None:
    allowed = {"dim_benchmark", "version_5"}
    if collection not in allowed:
        allowed_msg = ", ".join(sorted(allowed))
        raise ValueError(f"Safety guard: benchmark only allows --collection in {{{allowed_msg}}}.")


def parse_dims(raw: str) -> List[int]:
    dims: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        dim = int(token)
        if dim <= 0:
            raise ValueError(f"Invalid dimension: {dim}")
        dims.append(dim)
    if not dims:
        raise ValueError("No dimensions provided.")
    return dims


def _extract_where_body(query: str) -> str:
    q_upper = query.upper()
    where_idx = q_upper.find("WHERE")
    if where_idx < 0:
        raise ValueError("Query must contain WHERE.")
    open_idx = query.find("{", where_idx)
    close_idx = query.rfind("}")
    if open_idx < 0 or close_idx <= open_idx:
        raise ValueError("Query WHERE clause must include {...}.")
    return query[open_idx + 1 : close_idx].strip()


def _tokenize_pattern(where_body: str) -> List[str]:
    import re

    return re.findall(r'(<[^>]+>|"[^"]*"|\?[A-Za-z_][A-Za-z0-9_]*)', where_body)


def _select_vars(query: str) -> List[str]:
    import re

    m = re.search(r"SELECT\s+(.*?)\s+WHERE", query, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    return [v.strip().lstrip("?") for v in re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", m.group(1))]


def parse_query_pattern(query: str) -> QueryPattern:
    body = _extract_where_body(query)
    toks = _tokenize_pattern(body)
    if len(toks) < 3:
        raise ValueError(f"Could not parse triple pattern: {body}")
    s_tok, p_tok, o_tok = toks[0], toks[1], toks[2]
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
        select_vars=_select_vars(query),
    )


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


def load_queries(queries_file: Optional[str]) -> List[Tuple[str, str, str]]:
    if not queries_file:
        return DEFAULT_QUERIES
    payload = json.loads(Path(queries_file).read_text(encoding="utf-8"))
    rows: List[Tuple[str, str, str]] = []
    items = payload.get("queries", payload) if isinstance(payload, dict) else payload
    for item in items:
        qid = item.get("id")
        query = item.get("query")
        bucket = item.get("bucket", "manual")
        if not qid or not query:
            continue
        rows.append((str(qid), str(bucket), str(query)))
    if not rows:
        raise ValueError("queries-file had no valid entries.")
    return rows


def add_limit_to_query(query: str, limit: int) -> str:
    if "LIMIT" in query.upper():
        return query
    query = query.rstrip()
    close_brace = query.rfind("}")
    if close_brace >= 0:
        return query[: close_brace + 1] + f" LIMIT {limit}"
    return query + f" LIMIT {limit}"


def _looks_like_uri(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _normalize_literal_value(value: str) -> str:
    raw = value.strip()
    match = re.match(r'^"(.*)"(?:\^\^.+|@[A-Za-z0-9-]+)?$', raw)
    if match:
        return match.group(1)
    return raw


def _normalize_binding_row(row: Dict) -> Dict[str, Dict[str, str]]:
    normalized: Dict[str, Dict[str, str]] = {}
    for key, raw in row.items():
        if isinstance(raw, dict) and "value" in raw:
            vtype = str(raw.get("type", "literal"))
            value = str(raw.get("value", ""))
            if vtype != "uri":
                value = _normalize_literal_value(value)
            normalized[key] = {
                "type": vtype,
                "value": value,
            }
            continue

        value = str(raw)
        vtype = "uri" if _looks_like_uri(value) else "literal"
        if vtype == "literal":
            value = _normalize_literal_value(value)
        normalized[key] = {
            "type": vtype,
            "value": value,
        }
    return normalized


def run_sparql_baseline(query: str, rdf_file: str, limit: int) -> List[Dict[str, Dict[str, str]]]:
    limited_query = add_limit_to_query(query, limit)
    cmd = ["comunica-sparql-file", rdf_file, limited_query]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "comunica-sparql-file failed")
    payload = json.loads(proc.stdout)
    if isinstance(payload, dict):
        bindings = payload.get("results", {}).get("bindings", [])
        return [_normalize_binding_row(row) for row in bindings if isinstance(row, dict)]
    if isinstance(payload, list):
        return [_normalize_binding_row(row) for row in payload if isinstance(row, dict)]
    return []


def binding_to_id(binding: Dict[str, Dict[str, str]]) -> str:
    parts = []
    for key in sorted(binding.keys()):
        value_obj = binding[key]
        if isinstance(value_obj, dict):
            value = str(value_obj.get("value", ""))
            vtype = str(value_obj.get("type", ""))
            if vtype != "uri":
                value = _normalize_literal_value(value)
        else:
            value = str(value_obj)
            vtype = "literal"
            value = _normalize_literal_value(value)
        parts.append(f"{key}={vtype}:{value}")
    return "|".join(parts)


def triple_to_binding(triple: CandidateTriple, select_vars: Sequence[str]) -> Dict[str, Dict[str, str]]:
    binding: Dict[str, Dict[str, str]] = {}
    for var in select_vars:
        v = var.lstrip("?")
        low = v.lower()
        if low in {"x", "s", "subject"}:
            binding[v] = {"type": "uri", "value": triple.subject.strip("<>")}
        elif low in {"p", "predicate"}:
            binding[v] = {"type": "uri", "value": triple.predicate.strip("<>")}
        elif low in {"o", "object"}:
            literal_value = _normalize_literal_value(triple.object_value)
            binding[v] = {
                "type": "literal" if triple.object_type == "literal" else "uri",
                "value": triple.object_value.strip("<>") if triple.object_type == "uri" else literal_value,
            }
    if not binding and select_vars:
        # Fallback: put the first variable on subject.
        first = select_vars[0].lstrip("?")
        binding[first] = {"type": "uri", "value": triple.subject.strip("<>")}
    return binding


def _hit_to_triple(hit: Dict) -> Optional[CandidateTriple]:
    text = hit.get("text")
    parsed = parse_nt_triple_line(text)
    if not parsed:
        return None
    obj_is_uri = parsed.object_value.startswith("<") and parsed.object_value.endswith(">")
    return CandidateTriple(
        subject=parsed.subject,
        predicate=parsed.predicate,
        object_value=parsed.object_value,
        object_type="uri" if obj_is_uri else "literal",
    )


def _matches_to_bindings(
    matches: List[Dict],
    pattern: QueryPattern,
    *,
    apply_pattern_filter: bool = True,
) -> Tuple[List[Dict[str, Dict[str, str]]], Set[int]]:
    """Map Milvus hits to bindings; optionally apply pattern post-filter."""
    bindings: List[Dict[str, Dict[str, str]]] = []
    ids: Set[int] = set()
    for hit in matches:
        triple = _hit_to_triple(hit)
        if triple is None:
            continue
        if apply_pattern_filter and not string_part_match(pattern, triple):
            continue
        mid = hit.get("id")
        if mid is not None:
            ids.add(mid)
        bindings.append(triple_to_binding(triple, pattern.select_vars))
    return bindings, ids


def raw_bindings_from_matches(
    matches: List[Dict],
    pattern: QueryPattern,
) -> Tuple[List[Dict[str, Dict[str, str]]], int, int]:
    """Parse all Milvus hits to bindings (no pattern filter).

    Returns (bindings, raw_hit_count, raw_parseable_count).
    """
    bindings, _ids = _matches_to_bindings(matches, pattern, apply_pattern_filter=False)
    parseable_count = 0
    for hit in matches:
        if _hit_to_triple(hit) is not None:
            parseable_count += 1
    return bindings, len(matches), parseable_count


def run_vector_bindings(
    *,
    vdb: VectorDataBase,
    collection: str,
    query: str,
    pattern: QueryPattern,
    k: int,
    log: bool,
) -> Tuple[List[Dict[str, Dict[str, str]]], List[Dict]]:
    results = vdb.search(
        collection_name=collection,
        query_texts=query,
        limit=k,
        output_fields=["text"],
        log=log,
    )
    if not results:
        return [], []
    matches = list(results[0].get("matches", []))
    bindings, _ids = _matches_to_bindings(matches, pattern)
    return bindings, matches


def run_vector_bindings_pagination_grpc(
    *,
    grpc_endpoint: str,
    query: str,
    page_k: int,
    pagination_limit: Optional[int],
) -> Tuple[List[Dict[str, Dict[str, str]]], List[Dict], Dict[str, int]]:
    body = pattern_request_body_from_query(
        query,
        k=page_k,
        pagination_limit=pagination_limit,
        use_pagination=True,
        include_raw_hits=True,
    )
    result = query_pattern_pagination_grpc(grpc_endpoint, body)
    telemetry = {
        "pages_fetched": result.pages_fetched,
        "milvus_hits_total": result.milvus_hits_total,
        "resolved_limit": result.resolved_limit,
        "catalog_k": result.catalog_k,
        "page_k": result.page_k or page_k,
    }
    return result.bindings, result.raw_matches, telemetry


def run_vector_bindings_pagination(
    *,
    vdb: VectorDataBase,
    collection: str,
    query: str,
    page_k: int,
    pagination_limit: Optional[int],
    k_resolver: Optional[CatalogKResolver],
) -> Tuple[List[Dict[str, Dict[str, str]]], List[Dict], Dict[str, int]]:
    body = pattern_request_body_from_query(
        query,
        k=page_k,
        pagination_limit=pagination_limit,
        use_pagination=True,
        include_raw_hits=True,
    )
    query_input = PatternQueryInput.from_json(body)
    bindings, last_page, raw_matches = collect_pagination_pages(
        query_input,
        collection_name=collection,
        database=vdb,
        resolver=k_resolver or CatalogKResolver(catalog_path=None),
    )
    pag = last_page.pagination
    telemetry = {
        "pages_fetched": pag.page_index,
        "milvus_hits_total": pag.milvus_hits_total,
        "resolved_limit": pag.limit,
        "catalog_k": pag.catalog_k,
        "page_k": pag.k,
    }
    return bindings, raw_matches, telemetry


def run_vector_fetch_only(
    *,
    vdb: Optional[VectorDataBase],
    collection: str,
    query: str,
    pattern: QueryPattern,
    use_grpc: bool,
    grpc_endpoint: str,
    use_pagination: bool,
    use_adaptive: bool,
    page_k: int,
    effective_k: int,
    seed_or_fixed_k: int,
    multipliers: Sequence[int],
    adaptive_jaccard: float,
    pagination_limit: Optional[int],
    k_resolver: Optional[CatalogKResolver],
    catalog_stability_floor: Optional[int],
    log: bool,
) -> None:
    """Run one vector search path without timing (latency warmup)."""
    if use_pagination:
        if use_grpc:
            run_vector_bindings_pagination_grpc(
                grpc_endpoint=grpc_endpoint,
                query=query,
                page_k=page_k,
                pagination_limit=pagination_limit,
            )
        else:
            run_vector_bindings_pagination(
                vdb=vdb,
                collection=collection,
                query=query,
                page_k=page_k,
                pagination_limit=pagination_limit,
                k_resolver=k_resolver,
            )
    elif use_adaptive:
        if use_grpc:
            run_vector_bindings_grpc(
                grpc_endpoint=grpc_endpoint,
                query=query,
                k=seed_or_fixed_k,
                use_adaptive=True,
                multipliers=multipliers,
                adaptive_jaccard=adaptive_jaccard,
            )
        else:
            run_vector_bindings_adaptive(
                vdb=vdb,
                collection=collection,
                query=query,
                pattern=pattern,
                seed_k=seed_or_fixed_k,
                multipliers=multipliers,
                jaccard_threshold=adaptive_jaccard,
                log=log,
                catalog_stability_floor=catalog_stability_floor,
            )
    elif use_grpc:
        run_vector_bindings_grpc(
            grpc_endpoint=grpc_endpoint,
            query=query,
            k=effective_k,
            use_adaptive=False,
            multipliers=multipliers,
            adaptive_jaccard=adaptive_jaccard,
        )
    else:
        run_vector_bindings(
            vdb=vdb,
            collection=collection,
            query=query,
            pattern=pattern,
            k=effective_k,
            log=log,
        )


def resolve_query_k(
    *,
    args: argparse.Namespace,
    pattern: QueryPattern,
    k_resolver: Optional[CatalogKResolver],
) -> Tuple[Optional[int], int, int, Optional[int]]:
    """Return (catalog query_k, seed_or_fixed_k, effective_k, catalog_stability_floor)."""
    query_k = args.k
    if (query_k is None or args.use_adaptive) and k_resolver is not None:
        catalog_k = k_resolver.auto_k_for_pattern(
            subject=pattern.subject,
            predicate=pattern.predicate,
            object_value=pattern.object_value,
            object_type=pattern.object_type,
        )
        if catalog_k is not None:
            query_k = catalog_k
    seed_or_fixed_k = query_k if query_k is not None else (args.k if args.k is not None else 5000)
    effective_k = milvus_safe_k(seed_or_fixed_k)
    catalog_stability_floor: Optional[int] = None
    if k_resolver is not None and k_resolver.available:
        catalog_stability_floor = k_resolver.catalog_match_count(
            subject=pattern.subject,
            predicate=pattern.predicate,
            object_value=pattern.object_value,
            object_type=pattern.object_type,
        )
    return query_k, seed_or_fixed_k, effective_k, catalog_stability_floor


def run_vector_bindings_grpc(
    *,
    grpc_endpoint: str,
    query: str,
    k: Optional[int],
    use_adaptive: bool,
    multipliers: Sequence[int],
    adaptive_jaccard: float,
) -> Tuple[List[Dict[str, Dict[str, str]]], List[Dict]]:
    body = pattern_request_body_from_query(
        query,
        k=None if use_adaptive else k,
        adaptive_multipliers=list(multipliers) if use_adaptive else None,
        adaptive_jaccard=adaptive_jaccard if use_adaptive else None,
        include_raw_hits=True,
    )
    result = query_pattern_grpc(grpc_endpoint, body)
    return result.bindings, result.raw_matches


def run_vector_bindings_adaptive(
    *,
    vdb: VectorDataBase,
    collection: str,
    query: str,
    pattern: QueryPattern,
    seed_k: int,
    multipliers: Sequence[int],
    jaccard_threshold: float,
    log: bool,
    catalog_stability_floor: Optional[int] = None,
) -> Tuple[List[Dict[str, Dict[str, str]]], int, List[Dict]]:
    """Run adaptive escalation for a single query.

    Returns (bindings, rounds_used, last_round_matches). `rounds_used` is the
    index of the final round that ran (0-based), useful for telemetry.

    catalog_stability_floor: when set, stability-based early stop is allowed
        only once the post-filtered hit count reaches this catalog lower bound.
    """
    rounds_seen = {"n": 0}
    last_matches: List[Dict] = []

    def _filter(matches: List[Dict], query_idx: int) -> Tuple[List[Dict], Set[int]]:
        rounds_seen["n"] += 1
        nonlocal last_matches
        last_matches = list(matches)
        return _matches_to_bindings(matches, pattern)

    final_rows = adaptive_batch_search(
        vdb=vdb,
        collection_name=collection,
        search_queries=[query],
        seed_ks=[seed_k],
        filter_fn=_filter,
        multipliers=tuple(multipliers),
        jaccard_threshold=jaccard_threshold,
        log=log,
        stability_count_floors=[catalog_stability_floor],
    )
    bindings = final_rows[0] if final_rows else []
    return bindings, rounds_seen["n"], last_matches


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    # Empty-ground-truth + empty-retrieval counts as perfect agreement.
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def compute_pr_metrics(
    baseline_bindings: List[Dict[str, Dict[str, str]]],
    candidate_bindings: List[Dict[str, Dict[str, str]]],
) -> Dict[str, float | int | bool]:
    baseline_ids = {binding_to_id(b) for b in baseline_bindings}
    candidate_ids = {binding_to_id(b) for b in candidate_bindings}
    tp = len(baseline_ids & candidate_ids)
    fp = len(candidate_ids - baseline_ids)
    fn = len(baseline_ids - candidate_ids)
    precision, recall = precision_recall(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "jaccard": jaccard(baseline_ids, candidate_ids),
        "exact_match": baseline_ids == candidate_ids,
        "baseline_count": len(baseline_ids),
        "candidate_count": len(candidate_ids),
    }


def maybe_run_load_phase(args: argparse.Namespace, dim: int) -> None:
    script = Path(__file__).resolve().parent / "run_dim_load_pipeline.py"
    cmd: List[str] = [
        sys.executable,
        str(script),
        "--input-file",
        args.load_input_file,
        "--collection",
        args.collection,
        "--dimensions",
        str(dim),
        "--database-name",
        args.database_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--embedding-model",
        args.embedding_model,
        "--dim-adjustment",
        args.dim_adjustment,
        "--component-fusion",
        args.component_fusion,
        "--chunk-size",
        str(args.load_chunk_size),
        "--out-dir",
        str(Path(args.out_dir) / "load_phase"),
    ]
    if args.load_max_lines is not None:
        cmd.extend(["--max-lines", str(args.load_max_lines)])
    if args.log:
        cmd.append("--log")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Load phase failed for dim={dim}")


def write_outputs(payload: Dict, out_dir: Path) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = payload["timestamp_utc"]
    base = out_dir / f"vector_dim_pr_{ts}"
    json_path = base.with_suffix(".json")
    summary_csv_path = base.with_name(f"{base.name}_summary.csv")
    per_query_csv_path = base.with_name(f"{base.name}_per_query.csv")

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dimension",
                "avg_precision",
                "avg_recall",
                "avg_precision_pct",
                "avg_recall_pct",
                "mean_jaccard",
                "avg_raw_precision",
                "avg_raw_recall",
                "mean_raw_jaccard",
                "queries_total",
                "passes_threshold",
                "sp*_avg_precision",
                "sp*_avg_recall",
                "sp*_avg_raw_precision",
                "sp*_avg_raw_recall",
                "*po_avg_precision",
                "*po_avg_recall",
                "*po_avg_raw_precision",
                "*po_avg_raw_recall",
                "s*o_avg_precision",
                "s*o_avg_recall",
                "s*o_avg_raw_precision",
                "s*o_avg_raw_recall",
                "avg_vector_query_seconds",
                "total_vector_query_seconds",
            ],
        )
        writer.writeheader()
        for row in payload["per_dimension"]:
            bucket_metrics = row.get("bucket_metrics", {})
            writer.writerow(
                {
                    "dimension": row["dimension"],
                    "avg_precision": row["avg_precision"],
                    "avg_recall": row["avg_recall"],
                    "avg_precision_pct": row["avg_precision_pct"],
                    "avg_recall_pct": row["avg_recall_pct"],
                    "mean_jaccard": row["mean_jaccard"],
                    "avg_raw_precision": row.get("avg_raw_precision"),
                    "avg_raw_recall": row.get("avg_raw_recall"),
                    "mean_raw_jaccard": row.get("mean_raw_jaccard"),
                    "queries_total": row["queries_total"],
                    "passes_threshold": row["passes_threshold"],
                    "avg_vector_query_seconds": row.get("avg_vector_query_seconds"),
                    "total_vector_query_seconds": row.get("total_vector_query_seconds"),
                    "sp*_avg_precision": bucket_metrics.get("sp*", {}).get("avg_precision"),
                    "sp*_avg_recall": bucket_metrics.get("sp*", {}).get("avg_recall"),
                    "sp*_avg_raw_precision": bucket_metrics.get("sp*", {}).get("avg_raw_precision"),
                    "sp*_avg_raw_recall": bucket_metrics.get("sp*", {}).get("avg_raw_recall"),
                    "*po_avg_precision": bucket_metrics.get("*po", {}).get("avg_precision"),
                    "*po_avg_recall": bucket_metrics.get("*po", {}).get("avg_recall"),
                    "*po_avg_raw_precision": bucket_metrics.get("*po", {}).get("avg_raw_precision"),
                    "*po_avg_raw_recall": bucket_metrics.get("*po", {}).get("avg_raw_recall"),
                    "s*o_avg_precision": bucket_metrics.get("s*o", {}).get("avg_precision"),
                    "s*o_avg_recall": bucket_metrics.get("s*o", {}).get("avg_recall"),
                    "s*o_avg_raw_precision": bucket_metrics.get("s*o", {}).get("avg_raw_precision"),
                    "s*o_avg_raw_recall": bucket_metrics.get("s*o", {}).get("avg_raw_recall"),
                }
            )

    with per_query_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
                "raw_tp",
                "raw_fp",
                "raw_fn",
                "raw_precision",
                "raw_recall",
                "raw_jaccard",
                "raw_hit_count",
                "raw_parseable_count",
                "raw_binding_count",
                "seed_k",
                "adaptive_rounds",
                "catalog_stability_floor",
                "vector_query_seconds",
                "pagination_page_k",
                "pagination_limit",
                "pagination_catalog_k",
                "pages_fetched",
                "milvus_hits_total",
            ],
        )
        writer.writeheader()
        for dim_row in payload["per_dimension"]:
            dim = dim_row["dimension"]
            for q in dim_row.get("queries", []):
                writer.writerow(
                    {
                        "dimension": dim,
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
                        "raw_tp": q.get("raw_tp"),
                        "raw_fp": q.get("raw_fp"),
                        "raw_fn": q.get("raw_fn"),
                        "raw_precision": q.get("raw_precision"),
                        "raw_recall": q.get("raw_recall"),
                        "raw_jaccard": q.get("raw_jaccard"),
                        "raw_hit_count": q.get("raw_hit_count"),
                        "raw_parseable_count": q.get("raw_parseable_count"),
                        "raw_binding_count": q.get("raw_binding_count"),
                        "seed_k": q.get("seed_k"),
                        "adaptive_rounds": q.get("adaptive_rounds"),
                        "catalog_stability_floor": q.get("catalog_stability_floor"),
                        "vector_query_seconds": q.get("vector_query_seconds"),
                        "pagination_page_k": q.get("pagination_page_k"),
                        "pagination_limit": q.get("pagination_limit"),
                        "pagination_catalog_k": q.get("pagination_catalog_k"),
                        "pages_fetched": q.get("pages_fetched"),
                        "milvus_hits_total": q.get("milvus_hits_total"),
                    }
                )
    return json_path, summary_csv_path, per_query_csv_path


def _parse_multipliers(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0:
            raise ValueError(f"Invalid multiplier {v}")
        out.append(v)
    if not out:
        raise ValueError("--adaptive-multipliers must contain at least one value")
    return out


def main() -> int:
    args = parse_args()
    if args.use_pagination and args.use_adaptive:
        raise ValueError("Cannot combine --use-pagination and --use-adaptive")
    if args.use_pagination and args.k is None:
        raise ValueError("--use-pagination requires --k (page batch size)")
    if args.latency_warmup_queries < 0:
        raise ValueError("--latency-warmup-queries must be >= 0")
    require_safe_collection(args.collection)
    dims = parse_dims(args.dimensions)
    queries = load_queries(args.queries_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_dimension = []
    use_catalog_k = (
        args.k is None or args.use_adaptive or args.use_pagination or args.grpc_endpoint is not None
    )
    k_resolver = (
        CatalogKResolver(
            catalog_path=Path(args.catalog_path),
            scale=args.catalog_k_scale,
            min_k=args.catalog_min_k,
        )
        if use_catalog_k
        else None
    )
    multipliers = _parse_multipliers(args.adaptive_multipliers) if args.use_adaptive else []
    use_grpc = args.grpc_endpoint is not None
    if use_grpc:
        print(f"gRPC vector path: {args.grpc_endpoint}")
    if args.use_adaptive:
        print(
            f"Adaptive escalation enabled: multipliers={multipliers} "
            f"jaccard>={args.adaptive_jaccard}"
        )
    if args.use_pagination:
        print(
            f"Pagination enabled: page_k={args.k} "
            f"limit={args.pagination_limit or '2*catalog_k'}"
        )
    if k_resolver is not None and k_resolver.available:
        print(
            f"Catalog k: scale={args.catalog_k_scale} min_k={args.catalog_min_k} "
            f"path={args.catalog_path}"
        )

    for dim in dims:
        print(f"\n=== Dimension {dim} ===")
        if args.run_load_phase:
            print("Running load phase...")
            maybe_run_load_phase(args, dim)

        vdb: Optional[VectorDataBase] = None
        if not use_grpc:
            vdb = VectorDataBase(
                database_name=args.database_name,
                host=args.host,
                port=args.port,
                embedding_model=args.embedding_model,
                target_embedding_dim=dim,
                dim_adjustment=args.dim_adjustment,
                component_fusion=args.component_fusion,
            )
            vdb.connect()

        query_rows = []
        precision_vals: List[float] = []
        recall_vals: List[float] = []
        jaccard_vals = []
        raw_precision_vals: List[float] = []
        raw_recall_vals: List[float] = []
        raw_jaccard_vals: List[float] = []
        vector_query_seconds_vals: List[float] = []
        bucket_precision: Dict[str, List[float]] = {}
        bucket_recall: Dict[str, List[float]] = {}
        bucket_raw_precision: Dict[str, List[float]] = {}
        bucket_raw_recall: Dict[str, List[float]] = {}

        warmup_count = max(0, args.latency_warmup_queries)
        if warmup_count > 0:
            warmup_count = min(warmup_count, len(queries))
            print(
                f"Latency warmup: {warmup_count} untimed fetch(es) "
                "(cold-start excluded from avg_vector_query_seconds)"
            )
            page_k = milvus_safe_k(int(args.k)) if args.use_pagination else 0
            for _query_id, _bucket, warmup_query in queries[:warmup_count]:
                warmup_pattern = parse_query_pattern(warmup_query)
                _query_k, seed_or_fixed_k, effective_k, catalog_stability_floor = resolve_query_k(
                    args=args,
                    pattern=warmup_pattern,
                    k_resolver=k_resolver,
                )
                run_vector_fetch_only(
                    vdb=vdb,
                    collection=args.collection,
                    query=warmup_query,
                    pattern=warmup_pattern,
                    use_grpc=use_grpc,
                    grpc_endpoint=args.grpc_endpoint or "",
                    use_pagination=args.use_pagination,
                    use_adaptive=args.use_adaptive,
                    page_k=page_k,
                    effective_k=effective_k,
                    seed_or_fixed_k=seed_or_fixed_k,
                    multipliers=multipliers,
                    adaptive_jaccard=args.adaptive_jaccard,
                    pagination_limit=args.pagination_limit,
                    k_resolver=k_resolver,
                    catalog_stability_floor=catalog_stability_floor,
                    log=args.log,
                )

        for query_idx, (query_id, bucket, query) in enumerate(queries, start=1):
            pattern = parse_query_pattern(query)
            query_k, seed_or_fixed_k, effective_k, catalog_stability_floor = resolve_query_k(
                args=args,
                pattern=pattern,
                k_resolver=k_resolver,
            )

            adaptive_rounds = None
            raw_matches: List[Dict] = []
            pagination_telemetry: Optional[Dict] = None
            if args.use_pagination:
                page_k = milvus_safe_k(int(args.k))
                catalog_k_val = query_k
                resolved_limit = resolve_pagination_limit(
                    page_k,
                    catalog_k=catalog_k_val,
                    explicit_limit=args.pagination_limit,
                )
                baseline_bindings = run_sparql_baseline(query, args.rdf_file, resolved_limit)
                vector_query_start = perf_counter()
                if use_grpc:
                    vector_bindings, raw_matches, pagination_telemetry = (
                        run_vector_bindings_pagination_grpc(
                            grpc_endpoint=args.grpc_endpoint,
                            query=query,
                            page_k=page_k,
                            pagination_limit=args.pagination_limit,
                        )
                    )
                else:
                    vector_bindings, raw_matches, pagination_telemetry = (
                        run_vector_bindings_pagination(
                            vdb=vdb,
                            collection=args.collection,
                            query=query,
                            page_k=page_k,
                            pagination_limit=args.pagination_limit,
                            k_resolver=k_resolver,
                        )
                    )
                vector_query_seconds = perf_counter() - vector_query_start
            elif args.use_adaptive:
                ladder = build_k_ladder(seed_or_fixed_k, multipliers=tuple(multipliers))
                baseline_limit = ladder[-1] if ladder else effective_k
                baseline_bindings = run_sparql_baseline(query, args.rdf_file, baseline_limit)
                vector_query_start = perf_counter()
                if use_grpc:
                    vector_bindings, raw_matches = run_vector_bindings_grpc(
                        grpc_endpoint=args.grpc_endpoint,
                        query=query,
                        k=seed_or_fixed_k,
                        use_adaptive=True,
                        multipliers=multipliers,
                        adaptive_jaccard=args.adaptive_jaccard,
                    )
                    adaptive_rounds = len(multipliers)
                else:
                    vector_bindings, adaptive_rounds, raw_matches = run_vector_bindings_adaptive(
                        vdb=vdb,
                        collection=args.collection,
                        query=query,
                        pattern=pattern,
                        seed_k=seed_or_fixed_k,
                        multipliers=multipliers,
                        jaccard_threshold=args.adaptive_jaccard,
                        log=args.log,
                        catalog_stability_floor=catalog_stability_floor,
                    )
                vector_query_seconds = perf_counter() - vector_query_start
            else:
                baseline_bindings = run_sparql_baseline(query, args.rdf_file, effective_k)
                vector_query_start = perf_counter()
                if use_grpc:
                    vector_bindings, raw_matches = run_vector_bindings_grpc(
                        grpc_endpoint=args.grpc_endpoint,
                        query=query,
                        k=effective_k,
                        use_adaptive=False,
                        multipliers=multipliers,
                        adaptive_jaccard=args.adaptive_jaccard,
                    )
                else:
                    vector_bindings, raw_matches = run_vector_bindings(
                        vdb=vdb,
                        collection=args.collection,
                        query=query,
                        pattern=pattern,
                        k=effective_k,
                        log=args.log,
                    )
                vector_query_seconds = perf_counter() - vector_query_start

            post_metrics = compute_pr_metrics(baseline_bindings, vector_bindings)
            raw_binding_list, raw_hit_count, raw_parseable_count = raw_bindings_from_matches(
                raw_matches, pattern
            )
            raw_metrics = compute_pr_metrics(baseline_bindings, raw_binding_list)
            raw_binding_count = raw_metrics["candidate_count"]

            precision_vals.append(float(post_metrics["precision"]))
            recall_vals.append(float(post_metrics["recall"]))
            jaccard_vals.append(float(post_metrics["jaccard"]))
            raw_precision_vals.append(float(raw_metrics["precision"]))
            raw_recall_vals.append(float(raw_metrics["recall"]))
            raw_jaccard_vals.append(float(raw_metrics["jaccard"]))
            vector_query_seconds_vals.append(vector_query_seconds)
            bucket_precision.setdefault(bucket, []).append(float(post_metrics["precision"]))
            bucket_recall.setdefault(bucket, []).append(float(post_metrics["recall"]))
            bucket_raw_precision.setdefault(bucket, []).append(float(raw_metrics["precision"]))
            bucket_raw_recall.setdefault(bucket, []).append(float(raw_metrics["recall"]))

            row = {
                "query_id": query_id,
                "bucket": bucket,
                "query": query,
                "baseline_count": post_metrics["baseline_count"],
                "vector_count": post_metrics["candidate_count"],
                "tp": post_metrics["tp"],
                "fp": post_metrics["fp"],
                "fn": post_metrics["fn"],
                "precision": post_metrics["precision"],
                "recall": post_metrics["recall"],
                "jaccard": post_metrics["jaccard"],
                "exact_match": post_metrics["exact_match"],
                "raw_tp": raw_metrics["tp"],
                "raw_fp": raw_metrics["fp"],
                "raw_fn": raw_metrics["fn"],
                "raw_precision": raw_metrics["precision"],
                "raw_recall": raw_metrics["recall"],
                "raw_jaccard": raw_metrics["jaccard"],
                "raw_hit_count": raw_hit_count,
                "raw_parseable_count": raw_parseable_count,
                "raw_binding_count": raw_binding_count,
                "seed_k": seed_or_fixed_k,
                "adaptive_rounds": adaptive_rounds,
                "catalog_stability_floor": catalog_stability_floor,
                "vector_query_seconds": round(vector_query_seconds, 4),
                "pagination_page_k": (
                    pagination_telemetry.get("page_k") if pagination_telemetry else None
                ),
                "pagination_limit": (
                    pagination_telemetry.get("resolved_limit") if pagination_telemetry else None
                ),
                "pagination_catalog_k": (
                    pagination_telemetry.get("catalog_k") if pagination_telemetry else None
                ),
                "pages_fetched": (
                    pagination_telemetry.get("pages_fetched") if pagination_telemetry else None
                ),
                "milvus_hits_total": (
                    pagination_telemetry.get("milvus_hits_total") if pagination_telemetry else None
                ),
            }
            query_rows.append(row)
            if args.log or query_idx <= 10 or query_idx % 250 == 0:
                rounds_str = f" rounds={adaptive_rounds}" if adaptive_rounds is not None else ""
                print(
                    f"{query_id} ({bucket}): seed_k={seed_or_fixed_k}{rounds_str} "
                    f"|GT|={post_metrics['baseline_count']} |RET|={post_metrics['candidate_count']} "
                    f"TP={post_metrics['tp']} FP={post_metrics['fp']} FN={post_metrics['fn']} "
                    f"P={post_metrics['precision']:.4f} R={post_metrics['recall']:.4f} "
                    f"raw_P={raw_metrics['precision']:.4f} raw_R={raw_metrics['recall']:.4f} "
                    f"vector={vector_query_seconds * 1000:.1f}ms"
                )

        total_queries = len(queries)
        avg_precision = sum(precision_vals) / len(precision_vals) if precision_vals else 0.0
        avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0
        avg_precision_pct = avg_precision * 100.0
        avg_recall_pct = avg_recall * 100.0
        mean_jaccard = sum(jaccard_vals) / len(jaccard_vals) if jaccard_vals else 0.0
        avg_raw_precision = (
            sum(raw_precision_vals) / len(raw_precision_vals) if raw_precision_vals else 0.0
        )
        avg_raw_recall = sum(raw_recall_vals) / len(raw_recall_vals) if raw_recall_vals else 0.0
        mean_raw_jaccard = (
            sum(raw_jaccard_vals) / len(raw_jaccard_vals) if raw_jaccard_vals else 0.0
        )
        total_vector_query_seconds = sum(vector_query_seconds_vals)
        avg_vector_query_seconds = (
            total_vector_query_seconds / len(vector_query_seconds_vals)
            if vector_query_seconds_vals
            else 0.0
        )
        threshold = args.accuracy_threshold_pct / 100.0
        passes_threshold = avg_precision >= threshold and avg_recall >= threshold

        bucket_metrics: Dict[str, Dict[str, float]] = {}
        all_bucket_names = sorted(
            set(bucket_precision.keys())
            | set(bucket_recall.keys())
            | set(bucket_raw_precision.keys())
            | set(bucket_raw_recall.keys())
        )
        for bucket_name in all_bucket_names:
            p_vals = bucket_precision.get(bucket_name, [])
            r_vals = bucket_recall.get(bucket_name, [])
            raw_p_vals = bucket_raw_precision.get(bucket_name, [])
            raw_r_vals = bucket_raw_recall.get(bucket_name, [])
            bucket_metrics[bucket_name] = {
                "count": float(len(p_vals)),
                "avg_precision": (sum(p_vals) / len(p_vals)) if p_vals else 0.0,
                "avg_recall": (sum(r_vals) / len(r_vals)) if r_vals else 0.0,
                "avg_raw_precision": (sum(raw_p_vals) / len(raw_p_vals)) if raw_p_vals else 0.0,
                "avg_raw_recall": (sum(raw_r_vals) / len(raw_r_vals)) if raw_r_vals else 0.0,
            }

        dim_payload = {
            "dimension": dim,
            "queries_total": total_queries,
            "latency_warmup_queries": warmup_count,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_precision_pct": avg_precision_pct,
            "avg_recall_pct": avg_recall_pct,
            "mean_jaccard": mean_jaccard,
            "avg_raw_precision": avg_raw_precision,
            "avg_raw_recall": avg_raw_recall,
            "mean_raw_jaccard": mean_raw_jaccard,
            "passes_threshold": passes_threshold,
            "avg_vector_query_seconds": round(avg_vector_query_seconds, 4),
            "total_vector_query_seconds": round(total_vector_query_seconds, 2),
            "bucket_metrics": bucket_metrics,
            "queries": query_rows,
        }
        per_dimension.append(dim_payload)
        print(
            f"dim={dim} avg_precision={avg_precision_pct:.2f}% avg_recall={avg_recall_pct:.2f}% "
            f"avg_raw_precision={avg_raw_precision * 100:.2f}% "
            f"avg_raw_recall={avg_raw_recall * 100:.2f}% "
            f"avg_vector_query={avg_vector_query_seconds * 1000:.1f}ms "
            f"(threshold={args.accuracy_threshold_pct:.2f}% each)"
        )

    passing_dims = sorted([d["dimension"] for d in per_dimension if d["passes_threshold"]])
    lowest_passing_dimension = passing_dims[0] if passing_dims else None

    k_mode = "fixed"
    if args.use_pagination:
        k_mode = "pagination"
    elif args.use_adaptive and args.k is None:
        k_mode = "catalog_adaptive_x1" if multipliers == [1] else "catalog_adaptive"
    elif args.k is None:
        k_mode = "catalog_auto"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "collection": args.collection,
        "dimensions_tested": dims,
        "k": args.k,
        "k_mode": k_mode,
        "catalog_k_scale": args.catalog_k_scale,
        "catalog_min_k": args.catalog_min_k,
        "component_fusion": args.component_fusion,
        "grpc_endpoint": args.grpc_endpoint,
        "metric_primary": "precision_recall",
        "metric_secondary": "raw_retrieval_precision_recall",
        "accuracy_threshold_pct": args.accuracy_threshold_pct,
        "run_load_phase": args.run_load_phase,
        "use_adaptive": args.use_adaptive,
        "use_pagination": args.use_pagination,
        "pagination_limit": args.pagination_limit,
        "adaptive_multipliers": multipliers if args.use_adaptive else None,
        "adaptive_jaccard": args.adaptive_jaccard if args.use_adaptive else None,
        "latency_warmup_queries": args.latency_warmup_queries,
        "per_dimension": per_dimension,
        "lowest_passing_dimension": lowest_passing_dimension,
    }

    json_path, csv_summary_path, csv_per_query_path = write_outputs(payload, out_dir)
    print(f"\nBenchmark JSON: {json_path}")
    print(f"Summary CSV:    {csv_summary_path}")
    print(f"Per-query CSV:  {csv_per_query_path}")
    print(f"Lowest passing dim: {lowest_passing_dimension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
