#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


DEFAULT_DIMS = "8"
DEFAULT_COUNT_OVERRIDES = {"Q2": 4320, "Q4": 1890}
DEFAULT_QUERIES = [
    (
        "Q1",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> }",
    ),
    (
        "Q2",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateStudent> }",
    ),
    (
        "Q3",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#Professor> }",
    ),
    (
        "Q4",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#Course> }",
    ),
    (
        "Q5",
        "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> }",
    ),
    (
        "Q6",
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
    parser.add_argument("--accuracy-threshold-pct", type=float, default=95.0, help="Dimension pass threshold")
    parser.add_argument("--overlap-threshold", type=float, default=0.95, help="Per-query Jaccard overlap threshold")
    parser.add_argument(
        "--count-overrides-json",
        default=json.dumps(DEFAULT_COUNT_OVERRIDES),
        help=(
            "JSON map of query_id -> expected baseline count used for count_match. "
            "Default applies your rounded counts for Q2/Q4."
        ),
    )
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


def parse_count_overrides(raw: str) -> Dict[str, int]:
    if raw is None or raw.strip() == "":
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("count-overrides-json must be a JSON object.")
    out: Dict[str, int] = {}
    for key, value in data.items():
        count = int(value)
        if count < 0:
            raise ValueError(f"Invalid expected count for {key}: {count}")
        out[str(key)] = count
    return out


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


def load_queries(queries_file: Optional[str]) -> List[Tuple[str, str]]:
    if not queries_file:
        return DEFAULT_QUERIES
    payload = json.loads(Path(queries_file).read_text(encoding="utf-8"))
    rows: List[Tuple[str, str]] = []
    for item in payload:
        qid = item.get("id")
        query = item.get("query")
        if not qid or not query:
            continue
        rows.append((str(qid), str(query)))
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


def _normalize_binding_row(row: Dict) -> Dict[str, Dict[str, str]]:
    normalized: Dict[str, Dict[str, str]] = {}
    for key, raw in row.items():
        if isinstance(raw, dict) and "value" in raw:
            normalized[key] = {
                "type": str(raw.get("type", "literal")),
                "value": str(raw.get("value", "")),
            }
            continue

        value = str(raw)
        normalized[key] = {
            "type": "uri" if _looks_like_uri(value) else "literal",
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
        else:
            value = str(value_obj)
            vtype = "literal"
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
            binding[v] = {
                "type": "literal" if triple.object_type == "literal" else "uri",
                "value": triple.object_value.strip("<>") if triple.object_type == "uri" else triple.object_value,
            }
    if not binding and select_vars:
        # Fallback: put the first variable on subject.
        first = select_vars[0].lstrip("?")
        binding[first] = {"type": "uri", "value": triple.subject.strip("<>")}
    return binding


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

    bindings: List[Dict[str, Dict[str, str]]] = []
    for hit in results[0].get("matches", []):
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
        bindings.append(triple_to_binding(triple, pattern.select_vars))
    return bindings


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


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


def write_outputs(payload: Dict, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = payload["timestamp_utc"]
    base = out_dir / f"vector_dim_accuracy_{ts}"
    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dimension",
                "overall_accuracy_pct",
                "count_match_rate_pct",
                "mean_jaccard",
                "queries_passed",
                "queries_total",
                "passes_threshold",
            ],
        )
        writer.writeheader()
        for row in payload["per_dimension"]:
            writer.writerow(
                {
                    "dimension": row["dimension"],
                    "overall_accuracy_pct": row["overall_accuracy_pct"],
                    "count_match_rate_pct": row["count_match_rate_pct"],
                    "mean_jaccard": row["mean_jaccard"],
                    "queries_passed": row["queries_passed"],
                    "queries_total": row["queries_total"],
                    "passes_threshold": row["passes_threshold"],
                }
            )
    return json_path, csv_path


def main() -> int:
    args = parse_args()
    require_safe_collection(args.collection)
    dims = parse_dims(args.dimensions)
    count_overrides = parse_count_overrides(args.count_overrides_json)
    queries = load_queries(args.queries_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_dimension = []
    k_resolver = CatalogKResolver(catalog_path=Path(args.catalog_path)) if args.k is None else None

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
        query_passes = 0
        count_match_passes = 0
        jaccard_vals = []

        for query_id, query in queries:
            pattern = parse_query_pattern(query)
            query_k = args.k
            if query_k is None and k_resolver is not None:
                query_k = k_resolver.auto_k_for_pattern(
                    subject=pattern.subject,
                    predicate=pattern.predicate,
                    object_value=pattern.object_value,
                    object_type=pattern.object_type,
                )
            effective_k = query_k if query_k is not None else 1000
            effective_k = milvus_safe_k(effective_k)

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

            baseline_count = int(count_overrides.get(query_id, len(baseline_ids)))
            vector_count = len(vector_ids)
            count_match = baseline_count == vector_count
            overlap_jaccard = jaccard(baseline_ids, vector_ids)
            overlap_pass = overlap_jaccard >= args.overlap_threshold
            query_pass = bool(count_match and overlap_pass)
            query_accuracy_pct = 100.0 if query_pass else 0.0

            if count_match:
                count_match_passes += 1
            if query_pass:
                query_passes += 1
            jaccard_vals.append(overlap_jaccard)

            row = {
                "query_id": query_id,
                "query": query,
                "baseline_count": baseline_count,
                "vector_count": vector_count,
                "count_match": count_match,
                "overlap_jaccard": overlap_jaccard,
                "overlap_pass": overlap_pass,
                "query_accuracy_pct": query_accuracy_pct,
                "count_override_used": query_id in count_overrides,
            }
            query_rows.append(row)
            print(
                f"{query_id}: baseline={row['baseline_count']} vector={row['vector_count']} "
                f"count_match={count_match} jaccard={overlap_jaccard:.4f} pass={query_pass}"
            )

        total_queries = len(queries)
        overall_accuracy_pct = (query_passes / total_queries) * 100.0 if total_queries else 0.0
        count_match_rate_pct = (count_match_passes / total_queries) * 100.0 if total_queries else 0.0
        mean_jaccard = sum(jaccard_vals) / len(jaccard_vals) if jaccard_vals else 0.0
        passes_threshold = overall_accuracy_pct >= args.accuracy_threshold_pct

        dim_payload = {
            "dimension": dim,
            "queries_passed": query_passes,
            "queries_total": total_queries,
            "overall_accuracy_pct": overall_accuracy_pct,
            "count_match_rate_pct": count_match_rate_pct,
            "mean_jaccard": mean_jaccard,
            "passes_threshold": passes_threshold,
            "queries": query_rows,
        }
        per_dimension.append(dim_payload)
        print(
            f"dim={dim} overall_accuracy={overall_accuracy_pct:.2f}% "
            f"(threshold={args.accuracy_threshold_pct:.2f}%)"
        )

    passing_dims = sorted([d["dimension"] for d in per_dimension if d["passes_threshold"]])
    lowest_passing_dimension = passing_dims[0] if passing_dims else None

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "collection": args.collection,
        "dimensions_tested": dims,
        "k": args.k,
        "overlap_threshold": args.overlap_threshold,
        "accuracy_threshold_pct": args.accuracy_threshold_pct,
        "run_load_phase": args.run_load_phase,
        "per_dimension": per_dimension,
        "lowest_passing_dimension": lowest_passing_dimension,
    }

    json_path, csv_path = write_outputs(payload, out_dir)
    print(f"\nBenchmark JSON: {json_path}")
    print(f"Benchmark CSV:  {csv_path}")
    print(f"Lowest passing dim: {lowest_passing_dimension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
