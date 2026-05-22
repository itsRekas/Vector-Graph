#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow importing src modules while running from this folder.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from auto_k import CatalogKResolver, milvus_safe_k


@dataclass(frozen=True)
class QueryPattern:
    subject: Optional[str]
    predicate: Optional[str]
    object_value: Optional[str]
    object_type: Optional[str]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_queries = Path(__file__).resolve().parent / "results" / "random_queries_3000.json"
    parser = argparse.ArgumentParser(
        description=(
            "Run catalog-only baseline counts for query set and output CSV with "
            "query_id,k,baseline_count."
        )
    )
    parser.add_argument(
        "--queries-file",
        default=str(default_queries),
        help="JSON file with query entries (default: pr_benchmark/results/random_queries_3000.json)",
    )
    parser.add_argument(
        "--rdf-file",
        default=str(root_dir / "data" / "nts" / "RLUBM_cleaned.nt"),
        help="RDF NT file path for Comunica baseline.",
    )
    parser.add_argument(
        "--catalog-path",
        default=str(root_dir / "catalog.pkl"),
        help="Catalog pickle path used to compute auto-k.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(Path(__file__).resolve().parent / "results" / "catalog_count_3000.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_queries(queries_file: str) -> List[Tuple[str, str]]:
    payload = json.loads(Path(queries_file).read_text(encoding="utf-8"))
    rows: List[Tuple[str, str]] = []
    items = payload.get("queries", payload) if isinstance(payload, dict) else payload
    for item in items:
        qid = item.get("id")
        query = item.get("query")
        if not qid or not query:
            continue
        rows.append((str(qid), str(query)))
    if not rows:
        raise ValueError("queries-file had no valid entries.")
    return rows


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
    )


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


def main() -> int:
    args = parse_args()
    queries = load_queries(args.queries_file)
    resolver = CatalogKResolver(catalog_path=Path(args.catalog_path))
    if not resolver.available:
        raise RuntimeError(f"Could not load catalog from {args.catalog_path}: {resolver.error or 'unknown error'}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "k", "baseline_count"])
        writer.writeheader()

        total = len(queries)
        for idx, (query_id, query) in enumerate(queries, start=1):
            pattern = parse_query_pattern(query)
            query_k = resolver.auto_k_for_pattern(
                subject=pattern.subject,
                predicate=pattern.predicate,
                object_value=pattern.object_value,
                object_type=pattern.object_type,
            )
            effective_k = milvus_safe_k(query_k if query_k is not None else 5000)

            baseline_bindings = run_sparql_baseline(query, args.rdf_file, effective_k)
            baseline_count = len({binding_to_id(b) for b in baseline_bindings})
            writer.writerow(
                {
                    "query_id": query_id,
                    "k": effective_k,
                    "baseline_count": baseline_count,
                }
            )

            if idx <= 10 or idx % 250 == 0:
                print(f"{idx}/{total} {query_id}: k={effective_k} baseline_count={baseline_count}")

    print(f"Wrote {len(queries)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
