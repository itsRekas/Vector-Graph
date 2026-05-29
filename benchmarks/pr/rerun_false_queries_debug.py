#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from vector_endpoint.auto_k import CatalogKResolver, milvus_safe_k
from vector_endpoint.catalog import parse_nt_triple_line
from vector_endpoint.db.VectorDataBase import VectorDataBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun false queries from per_query CSV and export full raw baseline/vector "
            "comparison JSON without reloading data."
        )
    )
    parser.add_argument("--collection", default="dim_benchmark", help="Existing loaded collection")
    parser.add_argument("--dimension", type=int, default=128, help="Dimension label to filter in per_query CSV")
    parser.add_argument("--k", type=int, default=None, help="Top-k / LIMIT used for rerun (default: catalog auto-k)")
    parser.add_argument(
        "--catalog-path",
        default=str(Path(__file__).resolve().parents[2] / "catalog.pkl"),
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
        help="How model output is adjusted to target dim",
    )
    parser.add_argument("--rdf-file", default="../../data/nts/RLUBM_cleaned.nt", help="RDF file for baseline")
    parser.add_argument(
        "--per-query-csv",
        default=None,
        help="Specific per_query CSV path (default: latest in results/)",
    )
    parser.add_argument(
        "--queries-file",
        default="results/random_queries_3000.json",
        help="Query catalog JSON with query_id -> query text",
    )
    parser.add_argument("--limit-queries", type=int, default=None, help="Optional limit of false queries to rerun")
    parser.add_argument("--out-dir", default="results", help="Output directory")
    parser.add_argument("--log", action="store_true", help="Verbose logs")
    return parser.parse_args()


def find_latest_per_query_csv(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("vector_dim_pr_*_per_query.csv"))
    if not candidates:
        raise FileNotFoundError(f"No per_query CSV files found in {results_dir}")
    return candidates[-1]


def load_false_query_ids(per_query_csv: Path, dimension: int) -> List[str]:
    out: List[str] = []
    with per_query_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                dim_val = int(row.get("dimension", ""))
            except ValueError:
                continue
            if dim_val != dimension:
                continue
            if str(row.get("exact_match", "")).lower() == "false":
                qid = row.get("query_id")
                if qid:
                    out.append(qid)
    if not out:
        raise RuntimeError(f"No false queries found for dimension={dimension} in {per_query_csv}")
    return out


def load_query_map(queries_file: Path) -> Dict[str, Dict[str, str]]:
    payload = json.loads(queries_file.read_text(encoding="utf-8"))
    items = payload.get("queries", payload) if isinstance(payload, dict) else payload
    out: Dict[str, Dict[str, str]] = {}
    for item in items:
        qid = item.get("id")
        query = item.get("query")
        if not qid or not query:
            continue
        out[str(qid)] = {"query": str(query), "bucket": str(item.get("bucket", "unknown"))}
    return out


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
            normalized[key] = {"type": vtype, "value": value}
            continue
        value = str(raw)
        vtype = "uri" if _looks_like_uri(value) else "literal"
        if vtype == "literal":
            value = _normalize_literal_value(value)
        normalized[key] = {"type": vtype, "value": value}
    return normalized


def run_sparql_baseline_raw(query: str, rdf_file: str, k: int) -> Tuple[List[Dict], List[Dict[str, Dict[str, str]]]]:
    limited_query = add_limit_to_query(query, k)
    cmd = ["comunica-sparql-file", rdf_file, limited_query]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "comunica-sparql-file failed")
    payload = json.loads(proc.stdout)
    if isinstance(payload, dict):
        raw_rows = payload.get("results", {}).get("bindings", [])
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        raw_rows = []
    normalized = [_normalize_binding_row(row) for row in raw_rows if isinstance(row, dict)]
    return raw_rows, normalized


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


def parse_query_pattern(query: str) -> Dict[str, object]:
    body = _extract_where_body(query)
    toks = _tokenize_pattern(body)
    if len(toks) < 3:
        raise ValueError(f"Could not parse triple pattern: {body}")
    s_tok, p_tok, o_tok = toks[0], toks[1], toks[2]
    subject = None if s_tok.startswith("?") else s_tok
    predicate = None if p_tok.startswith("?") else p_tok
    if o_tok.startswith("?"):
        object_value, object_type = None, None
    elif o_tok.startswith('"') and o_tok.endswith('"'):
        object_value, object_type = o_tok[1:-1], "literal"
    else:
        object_value, object_type = o_tok, "uri"
    return {
        "subject": subject,
        "predicate": predicate,
        "object_value": object_value,
        "object_type": object_type,
        "select_vars": _select_vars(query),
    }


def string_part_match(pattern: Dict[str, object], subject: str, predicate: str, object_value: str, object_type: str) -> bool:
    if pattern["subject"] is not None and subject != pattern["subject"]:
        return False
    if pattern["predicate"] is not None and predicate != pattern["predicate"]:
        return False
    if pattern["object_value"] is not None and object_value != pattern["object_value"]:
        return False
    if pattern["object_type"] is not None and object_type != pattern["object_type"]:
        return False
    return True


def triple_to_binding(subject: str, predicate: str, object_value: str, object_type: str, select_vars: Sequence[str]) -> Dict[str, Dict[str, str]]:
    binding: Dict[str, Dict[str, str]] = {}
    for var in select_vars:
        v = var.lstrip("?")
        low = v.lower()
        if low in {"x", "s", "subject"}:
            binding[v] = {"type": "uri", "value": subject.strip("<>")}
        elif low in {"p", "predicate"}:
            binding[v] = {"type": "uri", "value": predicate.strip("<>")}
        elif low in {"o", "object"}:
            literal_value = _normalize_literal_value(object_value)
            binding[v] = {
                "type": "literal" if object_type == "literal" else "uri",
                "value": object_value.strip("<>") if object_type == "uri" else literal_value,
            }
    if not binding and select_vars:
        first = select_vars[0].lstrip("?")
        binding[first] = {"type": "uri", "value": subject.strip("<>")}
    return binding


def run_vector_bindings_raw(
    vdb: VectorDataBase,
    collection: str,
    query: str,
    pattern: Dict[str, object],
    k: int,
    log: bool,
) -> Tuple[int, List[Dict], List[Dict[str, Dict[str, str]]]]:
    results = vdb.search(
        collection_name=collection,
        query_texts=query,
        limit=k,
        output_fields=["text"],
        log=log,
    )
    if not results:
        return 0, [], []
    raw_matches = list(results[0].get("matches", []))
    raw_post_filtered_matches: List[Dict] = []
    bindings: List[Dict[str, Dict[str, str]]] = []
    for hit in raw_matches:
        text = hit.get("text")
        parsed = parse_nt_triple_line(text)
        if not parsed:
            continue
        object_value = parsed.object_value
        object_type = "uri" if object_value.startswith("<") and object_value.endswith(">") else "literal"
        if not string_part_match(pattern, parsed.subject, parsed.predicate, object_value, object_type):
            continue
        raw_post_filtered_matches.append(hit)
        bindings.append(
            triple_to_binding(
                parsed.subject,
                parsed.predicate,
                object_value,
                object_type,
                pattern["select_vars"],  # type: ignore[arg-type]
            )
        )
    return len(raw_matches), raw_post_filtered_matches, bindings


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


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def false_reason(baseline_count: int, vector_count: int, tp: int) -> str:
    if baseline_count == vector_count and tp < baseline_count:
        return "count_equal_but_identity_mismatch"
    if vector_count < baseline_count:
        return "missing_ground_truth_results"
    if vector_count > baseline_count:
        return "extra_vector_results"
    return "mixed_mismatch"


def main() -> int:
    args = parse_args()
    if args.collection != "dim_benchmark":
        raise ValueError("Safety guard: this script only allows --collection dim_benchmark.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query_csv = Path(args.per_query_csv) if args.per_query_csv else find_latest_per_query_csv(out_dir)
    query_map = load_query_map(Path(args.queries_file))
    false_ids = load_false_query_ids(per_query_csv, args.dimension)
    if args.limit_queries is not None:
        false_ids = false_ids[: args.limit_queries]

    vdb = VectorDataBase(
        database_name=args.database_name,
        host=args.host,
        port=args.port,
        embedding_model=args.embedding_model,
        target_embedding_dim=args.dimension,
        dim_adjustment=args.dim_adjustment,
    )
    vdb.connect()
    if args.collection not in getattr(vdb, "_collections", set()):
        raise RuntimeError(
            f"Collection '{args.collection}' is not available after connect. "
            "Ensure Milvus is running and the target collection is loaded."
        )

    rows: List[Dict] = []
    k_resolver = CatalogKResolver(catalog_path=Path(args.catalog_path)) if args.k is None else None
    for idx, qid in enumerate(false_ids, start=1):
        if qid not in query_map:
            continue
        query_text = query_map[qid]["query"]
        bucket = query_map[qid]["bucket"]

        pattern = parse_query_pattern(query_text)
        query_k = args.k
        if query_k is None and k_resolver is not None:
            query_k = k_resolver.auto_k_for_pattern(
                subject=pattern.get("subject"),  # type: ignore[arg-type]
                predicate=pattern.get("predicate"),  # type: ignore[arg-type]
                object_value=pattern.get("object_value"),  # type: ignore[arg-type]
                object_type=pattern.get("object_type"),  # type: ignore[arg-type]
            )
        effective_k = query_k if query_k is not None else 5000
        effective_k = milvus_safe_k(effective_k)

        baseline_raw, baseline_bindings = run_sparql_baseline_raw(query_text, args.rdf_file, effective_k)
        vector_candidates_count, vector_post_filtered_raw, vector_bindings = run_vector_bindings_raw(
            vdb=vdb,
            collection=args.collection,
            query=query_text,
            pattern=pattern,
            k=effective_k,
            log=args.log,
        )

        baseline_ids = [binding_to_id(b) for b in baseline_bindings]
        vector_ids = [binding_to_id(b) for b in vector_bindings]
        baseline_set = set(baseline_ids)
        vector_set = set(vector_ids)
        tp = len(baseline_set & vector_set)
        fp = len(vector_set - baseline_set)
        fn = len(baseline_set - vector_set)
        precision, recall = precision_recall(tp, fp, fn)
        jac = jaccard(baseline_set, vector_set)

        rows.append(
            {
                "query_id": qid,
                "bucket": bucket,
                "dimension": args.dimension,
                "k": effective_k,
                "query": query_text,
                "baseline_count": len(baseline_set),
                "vector_count": len(vector_set),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "jaccard": jac,
                "false_reason": false_reason(len(baseline_set), len(vector_set), tp),
                "baseline_raw_bindings": baseline_raw,
                "vector_candidates_count": vector_candidates_count,
                "vector_post_filtered_matches": vector_post_filtered_raw,
                "baseline_normalized_bindings": baseline_bindings,
                "vector_normalized_bindings": vector_bindings,
                "baseline_ids": sorted(baseline_set),
                "vector_ids": sorted(vector_set),
                "intersection_ids": sorted(baseline_set & vector_set),
                "fp_ids": sorted(vector_set - baseline_set),
                "fn_ids": sorted(baseline_set - vector_set),
            }
        )

        if args.log or idx <= 10 or idx % 100 == 0:
            print(
                f"{idx:04d}/{len(false_ids)} {qid}: "
                f"|GT|={len(baseline_set)} |RET|={len(vector_set)} TP={tp} FP={fp} FN={fn}"
            )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"false_query_debug_{ts}.json"
    payload = {
        "generated_at_utc": ts,
        "source_per_query_csv": str(per_query_csv),
        "source_queries_file": str(Path(args.queries_file)),
        "collection": args.collection,
        "dimension": args.dimension,
        "k": args.k if args.k is not None else "auto",
        "false_query_count": len(rows),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote debug JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
