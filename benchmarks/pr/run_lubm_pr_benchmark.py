#!/usr/bin/env python3
"""Precision/recall benchmark for official LUBM Q1–Q14 join queries.

Ground truth: comunica-sparql-file over RLUBM_cleaned.nt
Vector path:   comunica-vector against the running vector-endpoint (full join engine)

Unlike run_vector_dim_accuracy_benchmark.py (single-pattern Milvus search + post-filter),
this script evaluates end-to-end query results from comunica-vector.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

PR_DIR = Path(__file__).resolve().parent
if str(PR_DIR) not in sys.path:
    sys.path.insert(0, str(PR_DIR))

from run_vector_dim_accuracy_benchmark import (  # noqa: E402
    add_limit_to_query,
    binding_to_id,
    jaccard,
    precision_recall,
    _normalize_binding_row,
    _parse_multipliers,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VE_SRC = REPO_ROOT / "vector-endpoint" / "src"
if str(VE_SRC) not in sys.path:
    sys.path.insert(0, str(VE_SRC))

from vector_endpoint.adaptive_exp.stability import build_k_ladder  # noqa: E402
from vector_endpoint.auto_k import CatalogKResolver  # noqa: E402


@dataclass(frozen=True)
class QueryPattern:
    subject: Optional[str]
    predicate: Optional[str]
    object_value: Optional[str]
    object_type: Optional[str]


@dataclass(frozen=True)
class PatternCatalogPlan:
    pattern_index: int
    pattern: QueryPattern
    catalog_count: Optional[int]
    seed_k: int
    ladder: Tuple[int, ...]
    ladder_max: int


@dataclass(frozen=True)
class LubmQuery:
    query_id: str
    query_num: int
    query_type: str
    expected_count: Optional[int]
    query: str


def default_comunica_vector_cmd() -> List[str]:
    import shutil

    if shutil.which("comunica-vector"):
        return ["comunica-vector"]
    repo_root = Path(__file__).resolve().parents[3]
    vector_js = repo_root / "comunica" / "engines" / "query-sparql" / "bin" / "vector.js"
    if vector_js.is_file():
        return ["node", str(vector_js)]
    return ["comunica-vector"]


def default_comunica_sparql_file_cmd() -> List[str]:
    import shutil

    if shutil.which("comunica-sparql-file"):
        return ["comunica-sparql-file"]
    repo_root = Path(__file__).resolve().parents[3]
    query_js = repo_root / "comunica" / "engines" / "query-sparql-file" / "bin" / "query.js"
    if query_js.is_file():
        return ["node", str(query_js)]
    return ["comunica-sparql-file"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_queries = repo_root / "RLUBM" / "Query_Types.txt"
    default_rdf = Path(__file__).resolve().parents[2] / "data" / "nts" / "RLUBM_cleaned.nt"

    parser = argparse.ArgumentParser(
        description=(
            "Measure precision/recall for LUBM Q1–Q14: comunica-vector vs "
            "comunica-sparql-file ground truth."
        )
    )
    parser.add_argument(
        "--queries-file",
        default=str(default_queries),
        help="Query_Types.txt path or JSON [{id, query, ...}]",
    )
    parser.add_argument("--rdf-file", default=str(default_rdf), help="RLUBM NT file for baseline")
    parser.add_argument(
        "--vector-endpoint",
        default="http://localhost:2222/vector",
        help="comunica-vector source URL",
    )
    parser.add_argument(
        "--comunica-vector-cmd",
        default=None,
        help="comunica-vector command prefix (default: auto-detect PATH or local bin/vector.js)",
    )
    parser.add_argument(
        "--comunica-sparql-file-cmd",
        default=None,
        help="comunica-sparql-file command prefix (default: auto-detect PATH or local bin/query.js)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Fixed top-k / LIMIT for every query. Ignored for vector path when "
            "--use-adaptive is set (still used as adaptive seed if --use-adaptive)."
        ),
    )
    default_catalog = Path(__file__).resolve().parents[2] / "catalog.pkl"
    parser.add_argument(
        "--catalog-path",
        default=str(default_catalog),
        help="catalog.pkl for per-BGP k (same as vector-endpoint)",
    )
    parser.add_argument(
        "--catalog-k-scale",
        type=float,
        default=1.2,
        help="seed_k = ceil(catalog_count * scale) per BGP (matches endpoint default)",
    )
    parser.add_argument(
        "--catalog-min-k",
        type=int,
        default=10,
        help="Minimum catalog seed k per BGP",
    )
    parser.add_argument(
        "--use-adaptive",
        action="store_true",
        help=(
            "Omit -k from comunica-vector; endpoint uses catalog seed k + "
            "adaptive_batch_search per BGP (multipliers/jaccard forwarded). "
            "Baseline LIMIT uses max ladder top over all BGPs in the query."
        ),
    )
    parser.add_argument(
        "--adaptive-multipliers",
        default="1,2,4,8",
        help="Ladder multipliers for baseline LIMIT and vector adaptive search.",
    )
    parser.add_argument(
        "--adaptive-jaccard",
        type=float,
        default=0.99,
        help="Endpoint adaptive stability threshold (informational in output).",
    )
    parser.add_argument(
        "--k-margin",
        type=float,
        default=1.25,
        help="When --k is omitted, k = max(min-k, ceil(expected_count * margin))",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=50,
        help="Minimum k when using auto-k",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=200000,
        help="Cap on auto-k (queries needing more are clamped with a warning)",
    )
    parser.add_argument(
        "--max-expected",
        type=int,
        default=None,
        help="Skip queries whose expected result count exceeds this (from Query_Types.txt)",
    )
    parser.add_argument(
        "--query-nums",
        default=None,
        help="Comma-separated query numbers to run, e.g. 1,2,3 (default: 1–14)",
    )
    parser.add_argument(
        "--accuracy-threshold-pct",
        type=float,
        default=95.0,
        help="Pass threshold for avg precision and avg recall",
    )
    parser.add_argument("--timeout", type=int, default=1200, help="Per-query subprocess timeout (seconds)")
    parser.add_argument("--out-dir", default="results", help="Output directory")
    parser.add_argument(
        "--write-queries-json",
        default=None,
        help="Optional path to write all parsed Q1–Q14 queries as JSON (pr format)",
    )
    parser.add_argument(
        "--export-queries-only",
        action="store_true",
        help="Only write --write-queries-json and exit (ignores --query-nums filter)",
    )
    parser.add_argument("--log", action="store_true", help="Verbose logs")
    return parser.parse_args()


def _parse_query_nums(raw: Optional[str]) -> Optional[set[int]]:
    if not raw:
        return None
    nums: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if token:
            nums.add(int(token))
    return nums


def _normalize_query_text(query: str) -> str:
    return " ".join(query.split())


def parse_query_types_txt(path: Path, *, max_num: int = 14) -> List[LubmQuery]:
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?=Query\s+\d+)", content, flags=re.IGNORECASE)
    queries: List[LubmQuery] = []

    for block in blocks:
        header = re.match(r"Query\s+(\d+)\s*:?", block.strip(), flags=re.IGNORECASE)
        if not header:
            continue
        num = int(header.group(1))
        if num > max_num:
            continue

        expected: Optional[int] = None
        m_exp = re.search(r"Results\s*:\s*([\d,]+)\s*tuples?", block, flags=re.IGNORECASE)
        if m_exp:
            expected = int(m_exp.group(1).replace(",", ""))

        query_type = "unknown"
        m_type = re.search(r"Query Type:\s*(.+)", block, flags=re.IGNORECASE)
        if m_type:
            query_type = m_type.group(1).strip()

        m_sel = re.search(r"(SELECT.*?})", block, flags=re.DOTALL | re.IGNORECASE)
        if not m_sel:
            continue
        query = _normalize_query_text(m_sel.group(1))
        queries.append(
            LubmQuery(
                query_id=f"LUBM_Q{num}",
                query_num=num,
                query_type=query_type,
                expected_count=expected,
                query=query,
            )
        )

    queries.sort(key=lambda q: q.query_num)
    return queries


def load_queries_from_json(path: Path) -> List[LubmQuery]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("queries", payload) if isinstance(payload, dict) else payload
    queries: List[LubmQuery] = []
    for item in items:
        qid = str(item.get("id", ""))
        query = item.get("query")
        if not qid or not query:
            continue
        num_match = re.search(r"(\d+)", qid)
        num = int(num_match.group(1)) if num_match else len(queries) + 1
        queries.append(
            LubmQuery(
                query_id=qid,
                query_num=num,
                query_type=str(item.get("bucket", item.get("query_type", "lubm"))),
                expected_count=item.get("expected_count"),
                query=_normalize_query_text(str(query)),
            )
        )
    return queries


def load_lubm_queries(path: Path) -> List[LubmQuery]:
    if path.suffix.lower() == ".json":
        return load_queries_from_json(path)
    return parse_query_types_txt(path)


def queries_to_json_payload(queries: Sequence[LubmQuery]) -> dict:
    return {
        "queries": [
            {
                "id": q.query_id,
                "bucket": q.query_type,
                "expected_count": q.expected_count,
                "query": q.query,
            }
            for q in queries
        ]
    }


def auto_k_for_query(q: LubmQuery, *, margin: float, min_k: int, max_k: int) -> int:
    if q.expected_count is None or q.expected_count <= 0:
        return min_k
    import math

    k = max(min_k, int(math.ceil(q.expected_count * margin)))
    return min(k, max_k)


def _extract_where_body(query: str) -> str:
    where_idx = query.upper().find("WHERE")
    if where_idx < 0:
        raise ValueError("Query must contain WHERE.")
    open_idx = query.find("{", where_idx)
    close_idx = query.rfind("}")
    if open_idx < 0 or close_idx <= open_idx:
        raise ValueError("Query WHERE clause must include {...}.")
    return query[open_idx + 1 : close_idx].strip()


def _tokenize_pattern(where_body: str) -> List[str]:
    return re.findall(r'(<[^>]+>|"[^"]*"|\?[A-Za-z_][A-Za-z0-9_]*)', where_body)


def _pattern_from_clause(clause: str) -> QueryPattern:
    toks = _tokenize_pattern(clause.strip())
    if len(toks) < 3:
        raise ValueError(f"Could not parse triple pattern: {clause}")
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
    )


def parse_query_patterns(query: str) -> List[QueryPattern]:
    """Extract all BGP triple patterns from a SPARQL WHERE block."""
    body = _extract_where_body(query)
    # Split on ". " before the next triple (variable), not on dots inside IRIs.
    clauses = [c.strip() for c in re.split(r"\.\s+(?=\?)", body.strip().rstrip(".")) if c.strip()]
    if not clauses:
        raise ValueError("No triple patterns found in WHERE clause")
    return [_pattern_from_clause(clause) for clause in clauses]


def pattern_catalog_plans(
    query: str,
    resolver: CatalogKResolver,
    *,
    multipliers: Sequence[int],
    max_k: int,
    explicit_seed_k: Optional[int] = None,
) -> List[PatternCatalogPlan]:
    """Per-BGP catalog counts and k ladders (same logic as vector-endpoint)."""
    plans: List[PatternCatalogPlan] = []
    for idx, pattern in enumerate(parse_query_patterns(query)):
        count = None
        if resolver.available:
            count = resolver.catalog_match_count(
                subject=pattern.subject,
                predicate=pattern.predicate,
                object_value=pattern.object_value,
                object_type=pattern.object_type,
            )
        if explicit_seed_k is not None:
            seed_k = explicit_seed_k
        elif count is not None:
            seed_k = resolver.auto_k_for_pattern(
                subject=pattern.subject,
                predicate=pattern.predicate,
                object_value=pattern.object_value,
                object_type=pattern.object_type,
            ) or resolver.min_k
        else:
            seed_k = resolver.min_k

        ladder = tuple(build_k_ladder(seed_k, multipliers=tuple(multipliers)))
        ladder_max = min(ladder[-1] if ladder else seed_k, max_k)
        plans.append(
            PatternCatalogPlan(
                pattern_index=idx,
                pattern=pattern,
                catalog_count=count,
                seed_k=seed_k,
                ladder=ladder,
                ladder_max=ladder_max,
            )
        )
    return plans


def plans_to_json(plans: Sequence[PatternCatalogPlan]) -> List[dict]:
    rows: List[dict] = []
    for plan in plans:
        p = plan.pattern
        rows.append(
            {
                "pattern_index": plan.pattern_index,
                "subject": p.subject,
                "predicate": p.predicate,
                "object": p.object_value,
                "object_type": p.object_type,
                "catalog_count": plan.catalog_count,
                "seed_k": plan.seed_k,
                "ladder": list(plan.ladder),
                "ladder_max": plan.ladder_max,
            }
        )
    return rows


def resolve_k_plan(
    q: LubmQuery,
    args: argparse.Namespace,
    multipliers: Sequence[int],
    resolver: CatalogKResolver,
) -> Tuple[Optional[int], int, Optional[int], List[PatternCatalogPlan]]:
    """Return (vector_k, baseline_limit, summary_seed_k, per_pattern_plans).

    vector_k is None when --use-adaptive (comunica-vector omits -k).
    baseline_limit is the max ladder top across BGPs so GT is not capped below
    what any single vector BGP step can reach at catalog scale.
    """
    plans = pattern_catalog_plans(
        q.query,
        resolver,
        multipliers=multipliers,
        max_k=args.max_k,
        explicit_seed_k=args.k,
    )
    if not plans:
        fallback = args.k if args.k is not None else args.min_k
        return (
            (None if args.use_adaptive else fallback),
            fallback,
            fallback,
            [],
        )

    ladder_tops = [p.ladder_max for p in plans]
    baseline_limit = max(ladder_tops)
    if q.expected_count is not None and q.expected_count > 0:
        import math

        gt_floor = max(args.min_k, int(math.ceil(q.expected_count * args.k_margin)))
        baseline_limit = max(baseline_limit, min(gt_floor, args.max_k))

    summary_seed_k = max(p.seed_k for p in plans)

    if args.use_adaptive:
        return None, baseline_limit, summary_seed_k, plans

    if args.k is not None:
        k_used = args.k
    else:
        k_used = max(summary_seed_k, args.min_k)
        k_used = min(k_used, args.max_k)
    return k_used, max(baseline_limit, k_used), summary_seed_k, plans


def parse_comunica_bindings(payload: object) -> List[Dict[str, Dict[str, str]]]:
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], dict):
            bindings = payload["results"].get("bindings", [])
            if isinstance(bindings, list):
                return [_normalize_binding_row(row) for row in bindings if isinstance(row, dict)]
        if "rows" in payload and isinstance(payload["rows"], list):
            return [_normalize_binding_row(row) for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [_normalize_binding_row(row) for row in payload if isinstance(row, dict)]
    return []


def run_comunica_vector(
    query: str,
    *,
    endpoint: str,
    k: Optional[int],
    timeout: int,
    cmd_prefix: Sequence[str],
    adaptive_multipliers: Optional[Sequence[int]] = None,
    adaptive_jaccard: Optional[float] = None,
) -> Tuple[List[Dict[str, Dict[str, str]]], float]:
    cmd = [*cmd_prefix, endpoint]
    if k is not None:
        limited_query = add_limit_to_query(query, k)
        cmd.extend(["-k", str(k), "-q", limited_query])
    else:
        if adaptive_multipliers:
            mult_str = ",".join(str(m) for m in adaptive_multipliers)
            cmd.extend(["--adaptive-multipliers", mult_str])
        if adaptive_jaccard is not None:
            cmd.extend(["--adaptive-jaccard", str(adaptive_jaccard)])
        cmd.extend(["-q", query])
    start = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "comunica-vector failed")
    payload = json.loads(proc.stdout)
    return parse_comunica_bindings(payload), elapsed


def run_sparql_baseline_timed(
    query: str,
    rdf_file: str,
    limit: int,
    timeout: int,
    cmd_prefix: Sequence[str],
) -> Tuple[List[Dict[str, Dict[str, str]]], float]:
    limited_query = add_limit_to_query(query, limit)
    cmd = [*cmd_prefix, rdf_file, limited_query]
    start = perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "comunica-sparql-file failed")
    payload = json.loads(proc.stdout)
    return parse_comunica_bindings(payload), elapsed


def write_outputs(payload: dict, out_dir: Path) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = payload["timestamp_utc"]
    base = out_dir / f"lubm_pr_{ts}"
    json_path = base.with_suffix(".json")
    summary_csv = base.with_name(f"{base.name}_summary.csv")
    per_query_csv = base.with_name(f"{base.name}_per_query.csv")

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload["summary"]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "queries_total",
                "queries_run",
                "queries_skipped",
                "queries_failed",
                "avg_precision",
                "avg_recall",
                "avg_precision_pct",
                "avg_recall_pct",
                "mean_jaccard",
                "passes_threshold",
            ],
        )
        writer.writeheader()
        writer.writerow(summary)

    with per_query_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query_num",
                "query_type",
                "expected_count",
                "use_adaptive",
                "seed_k",
                "baseline_limit",
                "k_used",
                "baseline_count",
                "vector_count",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "jaccard",
                "exact_match",
                "baseline_seconds",
                "vector_seconds",
                "error",
            ],
        )
        writer.writeheader()
        for row in payload["queries"]:
            csv_row = {k: row.get(k) for k in writer.fieldnames}
            writer.writerow(csv_row)

    return json_path, summary_csv, per_query_csv


def main() -> int:
    args = parse_args()
    queries_path = Path(args.queries_file)
    rdf_path = Path(args.rdf_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vector_cmd = (
        args.comunica_vector_cmd.split()
        if args.comunica_vector_cmd
        else default_comunica_vector_cmd()
    )
    sparql_file_cmd = (
        args.comunica_sparql_file_cmd.split()
        if args.comunica_sparql_file_cmd
        else default_comunica_sparql_file_cmd()
    )

    if not queries_path.is_file():
        print(f"Error: queries file not found: {queries_path}", file=sys.stderr)
        return 1
    if not rdf_path.is_file():
        print(f"Error: RDF file not found: {rdf_path}", file=sys.stderr)
        return 1

    all_queries = load_lubm_queries(queries_path)

    if args.write_queries_json:
        out_json = Path(args.write_queries_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(queries_to_json_payload(all_queries), indent=2),
            encoding="utf-8",
        )
        print(f"Wrote queries JSON: {out_json} ({len(all_queries)} queries)")
        if args.export_queries_only:
            return 0

    allowed_nums = _parse_query_nums(args.query_nums)
    if allowed_nums is not None:
        all_queries = [q for q in all_queries if q.query_num in allowed_nums]

    if not all_queries:
        print("Error: no queries selected.", file=sys.stderr)
        return 1

    query_rows: List[dict] = []
    skipped: List[str] = []
    precision_vals: List[float] = []
    recall_vals: List[float] = []
    jaccard_vals: List[float] = []
    failed = 0

    multipliers = _parse_multipliers(args.adaptive_multipliers)

    catalog_path = Path(args.catalog_path)
    resolver = CatalogKResolver(
        catalog_path=catalog_path if catalog_path.is_file() else None,
        scale=args.catalog_k_scale,
        min_k=args.catalog_min_k,
    )
    if resolver.available:
        print(f"Catalog k planning: {catalog_path} (scale={args.catalog_k_scale})")
    else:
        print(
            f"WARN: catalog unavailable ({resolver.error or catalog_path}); "
            f"falling back to min_k={args.catalog_min_k} per pattern"
        )

    print(f"Loaded {len(all_queries)} LUBM queries from {queries_path}")
    print(f"RDF baseline: {rdf_path}")
    print(f"Vector endpoint: {args.vector_endpoint}")
    print(f"comunica-vector: {' '.join(vector_cmd)}")
    print(f"comunica-sparql-file: {' '.join(sparql_file_cmd)}")
    if args.use_adaptive:
        print(
            f"Adaptive k: per-BGP catalog seed + multipliers {multipliers} "
            f"(jaccard>={args.adaptive_jaccard}); baseline LIMIT = max BGP ladder top"
        )
    elif args.k is None:
        print(
            f"Fixed k from catalog: max per-BGP seed (scale={args.catalog_k_scale}), "
            f"max_k={args.max_k}"
        )
    else:
        print(f"Fixed k={args.k} for all queries")

    for q in all_queries:
        if args.max_expected is not None and q.expected_count is not None:
            if q.expected_count > args.max_expected:
                skipped.append(q.query_id)
                if args.log:
                    print(f"SKIP {q.query_id}: expected {q.expected_count} > max {args.max_expected}")
                continue

        vector_k, baseline_limit, seed_k, pattern_plans = resolve_k_plan(
            q, args, multipliers, resolver
        )

        if (
            q.expected_count is not None
            and q.expected_count > 0
            and baseline_limit < q.expected_count
        ):
            print(
                f"WARN {q.query_id}: baseline_limit={baseline_limit} < expected "
                f"{q.expected_count} (raise --max-k)"
            )

        if args.log and pattern_plans:
            for plan in pattern_plans:
                print(
                    f"  BGP{plan.pattern_index}: catalog_count={plan.catalog_count} "
                    f"seed_k={plan.seed_k} ladder_max={plan.ladder_max}"
                )

        row: dict = {
            "query_id": q.query_id,
            "query_num": q.query_num,
            "query_type": q.query_type,
            "expected_count": q.expected_count,
            "use_adaptive": args.use_adaptive,
            "seed_k": seed_k,
            "baseline_limit": baseline_limit,
            "k_used": vector_k if vector_k is not None else "adaptive",
            "pattern_catalog_plans": plans_to_json(pattern_plans),
            "query": q.query,
            "baseline_count": None,
            "vector_count": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "jaccard": None,
            "exact_match": None,
            "baseline_seconds": None,
            "vector_seconds": None,
            "error": None,
        }

        try:
            baseline_bindings, baseline_secs = run_sparql_baseline_timed(
                q.query, str(rdf_path), baseline_limit, args.timeout, sparql_file_cmd
            )
            vector_bindings, vector_secs = run_comunica_vector(
                q.query,
                endpoint=args.vector_endpoint,
                k=vector_k,
                timeout=args.timeout,
                cmd_prefix=vector_cmd,
                adaptive_multipliers=multipliers if args.use_adaptive else None,
                adaptive_jaccard=args.adaptive_jaccard if args.use_adaptive else None,
            )

            baseline_ids = {binding_to_id(b) for b in baseline_bindings}
            vector_ids = {binding_to_id(b) for b in vector_bindings}

            tp = len(baseline_ids & vector_ids)
            fp = len(vector_ids - baseline_ids)
            fn = len(baseline_ids - vector_ids)
            precision, recall = precision_recall(tp, fp, fn)
            overlap_jaccard = jaccard(baseline_ids, vector_ids)

            row.update(
                {
                    "baseline_count": len(baseline_ids),
                    "vector_count": len(vector_ids),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "jaccard": overlap_jaccard,
                    "exact_match": baseline_ids == vector_ids,
                    "baseline_seconds": round(baseline_secs, 4),
                    "vector_seconds": round(vector_secs, 4),
                }
            )
            precision_vals.append(precision)
            recall_vals.append(recall)
            jaccard_vals.append(overlap_jaccard)

            k_label = "adaptive" if vector_k is None else str(vector_k)
            print(
                f"{q.query_id} ({q.query_type}) "
                f"vector_k={k_label} baseline_limit={baseline_limit} seed_k={seed_k} "
                f"|GT|={len(baseline_ids)} |RET|={len(vector_ids)} "
                f"TP={tp} FP={fp} FN={fn} P={precision:.4f} R={recall:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            row["error"] = str(exc)
            print(f"FAIL {q.query_id}: {exc}", file=sys.stderr)

        query_rows.append(row)

    run_count = len(precision_vals)
    avg_precision = sum(precision_vals) / run_count if run_count else 0.0
    avg_recall = sum(recall_vals) / run_count if run_count else 0.0
    mean_jaccard = sum(jaccard_vals) / run_count if run_count else 0.0
    threshold = args.accuracy_threshold_pct / 100.0
    passes = avg_precision >= threshold and avg_recall >= threshold

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "metric_primary": "precision_recall",
        "queries_file": str(queries_path),
        "rdf_file": str(rdf_path),
        "vector_endpoint": args.vector_endpoint,
        "catalog_path": str(catalog_path),
        "catalog_k_scale": args.catalog_k_scale,
        "catalog_min_k": args.catalog_min_k,
        "use_adaptive": args.use_adaptive,
        "adaptive_multipliers": list(multipliers) if args.use_adaptive else None,
        "adaptive_jaccard": args.adaptive_jaccard if args.use_adaptive else None,
        "k": args.k,
        "k_margin": args.k_margin,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "accuracy_threshold_pct": args.accuracy_threshold_pct,
        "summary": {
            "queries_total": len(all_queries),
            "queries_run": run_count,
            "queries_skipped": len(skipped),
            "queries_failed": failed,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_precision_pct": avg_precision * 100.0,
            "avg_recall_pct": avg_recall * 100.0,
            "mean_jaccard": mean_jaccard,
            "passes_threshold": passes,
        },
        "skipped_query_ids": skipped,
        "queries": query_rows,
    }

    json_path, summary_csv, per_query_csv = write_outputs(payload, out_dir)
    print(f"\nBenchmark JSON: {json_path}")
    print(f"Summary CSV:    {summary_csv}")
    print(f"Per-query CSV:  {per_query_csv}")
    print(
        f"Avg precision={avg_precision * 100:.2f}% avg_recall={avg_recall * 100:.2f}% "
        f"(threshold={args.accuracy_threshold_pct:.2f}% each, pass={passes})"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
