#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Allow importing src modules while running from this folder.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_k import CatalogKResolver, milvus_safe_k
from catalog import parse_nt_triple_line
from vector_endpoint.db.VectorDataBase import VectorDataBase
from adaptive_exp import adaptive_batch_search, build_k_ladder


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
        default=str(Path(__file__).resolve().parents[1] / "catalog.pkl"),
        help="Catalog pickle path used for auto-k when --k is omitted.",
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


def _matches_to_bindings(
    matches: List[Dict],
    pattern: QueryPattern,
) -> Tuple[List[Dict[str, Dict[str, str]]], Set[int]]:
    """Apply pattern post-filter and return (bindings, set of milvus match ids)."""
    bindings: List[Dict[str, Dict[str, str]]] = []
    ids: Set[int] = set()
    for hit in matches:
        text = hit.get("text")
        parsed = parse_nt_triple_line(text)
        if not parsed:
            continue
        obj_is_uri = parsed.object_value.startswith("<") and parsed.object_value.endswith(">")
        triple = CandidateTriple(
            subject=parsed.subject,
            predicate=parsed.predicate,
            object_value=parsed.object_value,
            object_type="uri" if obj_is_uri else "literal",
        )
        if not string_part_match(pattern, triple):
            continue
        mid = hit.get("id")
        if mid is not None:
            ids.add(mid)
        bindings.append(triple_to_binding(triple, pattern.select_vars))
    return bindings, ids


def run_vector_bindings(
    *,
    vdb: VectorDataBase,
    collection: str,
    query: str,
    pattern: QueryPattern,
    k: int,
    log: bool,
) -> List[Dict[str, Dict[str, str]]]:
    results = vdb.search(
        collection_name=collection,
        query_texts=query,
        limit=k,
        output_fields=["text"],
        log=log,
    )
    if not results:
        return []
    bindings, _ids = _matches_to_bindings(results[0].get("matches", []), pattern)
    return bindings


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
) -> Tuple[List[Dict[str, Dict[str, str]]], int]:
    """Run adaptive escalation for a single query.

    Returns (bindings, rounds_used). `rounds_used` is the index of the final
    round that ran (0-based), useful for telemetry.

    catalog_stability_floor: when set, stability-based early stop is allowed
        only once the post-filtered hit count reaches this catalog lower bound.
    """
    rounds_seen = {"n": 0}

    def _filter(matches: List[Dict], query_idx: int) -> Tuple[List[Dict], Set[int]]:
        rounds_seen["n"] += 1
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
    return bindings, rounds_seen["n"]


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
                "queries_total",
                "passes_threshold",
                "sp*_avg_precision",
                "sp*_avg_recall",
                "*po_avg_precision",
                "*po_avg_recall",
                "s*o_avg_precision",
                "s*o_avg_recall",
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
                    "queries_total": row["queries_total"],
                    "passes_threshold": row["passes_threshold"],
                    "sp*_avg_precision": bucket_metrics.get("sp*", {}).get("avg_precision"),
                    "sp*_avg_recall": bucket_metrics.get("sp*", {}).get("avg_recall"),
                    "*po_avg_precision": bucket_metrics.get("*po", {}).get("avg_precision"),
                    "*po_avg_recall": bucket_metrics.get("*po", {}).get("avg_recall"),
                    "s*o_avg_precision": bucket_metrics.get("s*o", {}).get("avg_precision"),
                    "s*o_avg_recall": bucket_metrics.get("s*o", {}).get("avg_recall"),
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
                "seed_k",
                "adaptive_rounds",
                "catalog_stability_floor",
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
                        "seed_k": q.get("seed_k"),
                        "adaptive_rounds": q.get("adaptive_rounds"),
                        "catalog_stability_floor": q.get("catalog_stability_floor"),
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
    require_safe_collection(args.collection)
    dims = parse_dims(args.dimensions)
    queries = load_queries(args.queries_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_dimension = []
    k_resolver = CatalogKResolver(catalog_path=Path(args.catalog_path)) if (args.k is None or args.use_adaptive) else None
    multipliers = _parse_multipliers(args.adaptive_multipliers) if args.use_adaptive else []
    if args.use_adaptive:
        max_multiplier = max(multipliers)
        print(
            f"Adaptive escalation enabled: multipliers={multipliers} "
            f"jaccard>={args.adaptive_jaccard}"
        )

    for dim in dims:
        print(f"\n=== Dimension {dim} ===")
        if args.run_load_phase:
            print("Running load phase...")
            maybe_run_load_phase(args, dim)

        vdb = VectorDataBase(
            database_name=args.database_name,
            host=args.host,
            port=args.port,
            embedding_model=args.embedding_model,
            target_embedding_dim=dim,
            dim_adjustment=args.dim_adjustment,
        )
        vdb.connect()

        query_rows = []
        precision_vals: List[float] = []
        recall_vals: List[float] = []
        jaccard_vals = []
        bucket_precision: Dict[str, List[float]] = {}
        bucket_recall: Dict[str, List[float]] = {}

        for query_idx, (query_id, bucket, query) in enumerate(queries, start=1):
            pattern = parse_query_pattern(query)
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

            adaptive_rounds = None
            catalog_stability_floor: Optional[int] = None
            if args.use_adaptive:
                if k_resolver is not None and k_resolver.available:
                    catalog_stability_floor = k_resolver.catalog_match_count(
                        subject=pattern.subject,
                        predicate=pattern.predicate,
                        object_value=pattern.object_value,
                        object_type=pattern.object_type,
                    )
                ladder = build_k_ladder(seed_or_fixed_k, multipliers=tuple(multipliers))
                baseline_limit = ladder[-1] if ladder else effective_k
                baseline_bindings = run_sparql_baseline(query, args.rdf_file, baseline_limit)
                vector_bindings, adaptive_rounds = run_vector_bindings_adaptive(
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
            else:
                baseline_bindings = run_sparql_baseline(query, args.rdf_file, effective_k)
                vector_bindings = run_vector_bindings(
                    vdb=vdb,
                    collection=args.collection,
                    query=query,
                    pattern=pattern,
                    k=effective_k,
                    log=args.log,
                )

            baseline_ids = {binding_to_id(b) for b in baseline_bindings}
            vector_ids = {binding_to_id(b) for b in vector_bindings}

            baseline_count = len(baseline_ids)
            vector_count = len(vector_ids)
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

            row = {
                "query_id": query_id,
                "bucket": bucket,
                "query": query,
                "baseline_count": baseline_count,
                "vector_count": vector_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "jaccard": overlap_jaccard,
                "exact_match": exact_match,
                "seed_k": seed_or_fixed_k,
                "adaptive_rounds": adaptive_rounds,
                "catalog_stability_floor": catalog_stability_floor,
            }
            query_rows.append(row)
            if args.log or query_idx <= 10 or query_idx % 250 == 0:
                rounds_str = f" rounds={adaptive_rounds}" if adaptive_rounds is not None else ""
                print(
                    f"{query_id} ({bucket}): seed_k={seed_or_fixed_k}{rounds_str} "
                    f"|GT|={baseline_count} |RET|={vector_count} "
                    f"TP={tp} FP={fp} FN={fn} P={precision:.4f} R={recall:.4f}"
                )

        total_queries = len(queries)
        avg_precision = sum(precision_vals) / len(precision_vals) if precision_vals else 0.0
        avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0
        avg_precision_pct = avg_precision * 100.0
        avg_recall_pct = avg_recall * 100.0
        mean_jaccard = sum(jaccard_vals) / len(jaccard_vals) if jaccard_vals else 0.0
        threshold = args.accuracy_threshold_pct / 100.0
        passes_threshold = avg_precision >= threshold and avg_recall >= threshold

        bucket_metrics: Dict[str, Dict[str, float]] = {}
        for bucket_name in sorted(set(bucket_precision.keys()) | set(bucket_recall.keys())):
            p_vals = bucket_precision.get(bucket_name, [])
            r_vals = bucket_recall.get(bucket_name, [])
            bucket_metrics[bucket_name] = {
                "count": float(len(p_vals)),
                "avg_precision": (sum(p_vals) / len(p_vals)) if p_vals else 0.0,
                "avg_recall": (sum(r_vals) / len(r_vals)) if r_vals else 0.0,
            }

        dim_payload = {
            "dimension": dim,
            "queries_total": total_queries,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_precision_pct": avg_precision_pct,
            "avg_recall_pct": avg_recall_pct,
            "mean_jaccard": mean_jaccard,
            "passes_threshold": passes_threshold,
            "bucket_metrics": bucket_metrics,
            "queries": query_rows,
        }
        per_dimension.append(dim_payload)
        print(
            f"dim={dim} avg_precision={avg_precision_pct:.2f}% avg_recall={avg_recall_pct:.2f}% "
            f"(threshold={args.accuracy_threshold_pct:.2f}% each)"
        )

    passing_dims = sorted([d["dimension"] for d in per_dimension if d["passes_threshold"]])
    lowest_passing_dimension = passing_dims[0] if passing_dims else None

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "collection": args.collection,
        "dimensions_tested": dims,
        "k": args.k,
        "metric_primary": "precision_recall",
        "accuracy_threshold_pct": args.accuracy_threshold_pct,
        "run_load_phase": args.run_load_phase,
        "use_adaptive": args.use_adaptive,
        "adaptive_multipliers": multipliers if args.use_adaptive else None,
        "adaptive_jaccard": args.adaptive_jaccard if args.use_adaptive else None,
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
