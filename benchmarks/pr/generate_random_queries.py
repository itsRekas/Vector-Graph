#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from vector_endpoint.catalog import parse_nt_triple_line


@dataclass(frozen=True)
class Triple:
    s: str
    p: str
    o: str
    o_type: str  # "uri" or "literal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random SPARQL single-pattern queries in three buckets: "
            "sp*, *po, s*o."
        )
    )
    parser.add_argument(
        "--input-file",
        default="data/nts/RLUBM_cleaned.nt",
        help="Path to NT file used for sampling",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sp-count", type=int, default=1000, help="Number of sp* queries")
    parser.add_argument("--po-count", type=int, default=1000, help="Number of *po queries")
    parser.add_argument("--so-count", type=int, default=1000, help="Number of s*o queries")
    parser.add_argument(
        "--out",
        default="results/random_queries_3000.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def load_triples(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parsed = parse_nt_triple_line(line)
            if not parsed:
                continue
            o_val = parsed.object_value
            o_type = "uri" if o_val.startswith("<") and o_val.endswith(">") else "literal"
            triples.append(Triple(s=parsed.subject, p=parsed.predicate, o=o_val, o_type=o_type))
    if not triples:
        raise RuntimeError(f"No parseable triples found in: {path}")
    return triples


def _format_object(value: str, o_type: str) -> str:
    if o_type == "uri":
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _build_sp_query(s: str, p: str) -> str:
    return f"SELECT ?O WHERE {{ {s} {p} ?O }}"


def _build_po_query(p: str, o: str, o_type: str) -> str:
    return f"SELECT ?S WHERE {{ ?S {p} {_format_object(o, o_type)} }}"


def _build_so_query(s: str, o: str, o_type: str) -> str:
    return f"SELECT ?P WHERE {{ {s} ?P {_format_object(o, o_type)} }}"


def sample_unique_keys(
    keys: Sequence[Tuple[str, ...]],
    count: int,
    rng: random.Random,
    label: str,
) -> List[Tuple[str, ...]]:
    if len(keys) < count:
        raise ValueError(f"Not enough unique {label} keys: requested={count}, available={len(keys)}")
    idx = list(range(len(keys)))
    rng.shuffle(idx)
    return [keys[i] for i in idx[:count]]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file)
    triples = load_triples(input_path)
    rng = random.Random(args.seed)

    # Build unique key pools.
    sp_pool: Dict[Tuple[str, str], int] = defaultdict(int)
    po_pool: Dict[Tuple[str, str, str], int] = defaultdict(int)
    so_pool: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for t in triples:
        sp_pool[(t.s, t.p)] += 1
        po_pool[(t.p, t.o, t.o_type)] += 1
        so_pool[(t.s, t.o, t.o_type)] += 1

    sp_keys = sample_unique_keys(list(sp_pool.keys()), args.sp_count, rng, "sp*")
    po_keys = sample_unique_keys(list(po_pool.keys()), args.po_count, rng, "*po")
    so_keys = sample_unique_keys(list(so_pool.keys()), args.so_count, rng, "s*o")

    rows: List[Dict[str, str]] = []
    for i, (s, p) in enumerate(sp_keys, start=1):
        rows.append(
            {
                "id": f"SP{i:04d}",
                "bucket": "sp*",
                "query": _build_sp_query(s, p),
            }
        )
    for i, (p, o, o_type) in enumerate(po_keys, start=1):
        rows.append(
            {
                "id": f"PO{i:04d}",
                "bucket": "*po",
                "query": _build_po_query(p, o, o_type),
            }
        )
    for i, (s, o, o_type) in enumerate(so_keys, start=1):
        rows.append(
            {
                "id": f"SO{i:04d}",
                "bucket": "s*o",
                "query": _build_so_query(s, o, o_type),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "input_file": str(input_path),
        "counts": {"sp*": args.sp_count, "*po": args.po_count, "s*o": args.so_count},
        "queries": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote query set: {out_path}")
    print(f"Total queries: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
