#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import string
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from catalog import Catalog, parse_nt_triple_line


Triple = Tuple[str, str, str]


def rand_token(prefix: str, n: int = 12) -> str:
    return f"<{prefix}:{''.join(random.choices(string.ascii_lowercase + string.digits, k=n))}>"


def gen_high_cardinality(n: int) -> Iterable[Triple]:
    for _ in range(n):
        s = rand_token("s")
        p = rand_token("p", 8)
        o = rand_token("o") if random.random() < 0.6 else "".join(random.choices(string.ascii_letters, k=10))
        yield (s, p, o)


def gen_low_cardinality_skewed(n: int) -> Iterable[Triple]:
    subjects = [f"<s:{i}>" for i in range(100)]
    predicates = [f"<p:{i}>" for i in range(12)]
    objects = [f"<o:{i}>" for i in range(500)]

    for _ in range(n):
        s = subjects[0] if random.random() < 0.35 else random.choice(subjects)
        p = predicates[0] if random.random() < 0.45 else random.choice(predicates)
        o = objects[0] if random.random() < 0.25 else random.choice(objects)
        yield (s, p, o)


def gen_mixed_realistic(n: int) -> Iterable[Triple]:
    predicates = [
        "<http://example.org/type>",
        "<http://example.org/name>",
        "<http://example.org/age>",
        "<http://example.org/category>",
        "<http://example.org/brand>",
        "<http://example.org/price>",
    ]
    categories = ["Laptop", "Phone", "Tablet", "Monitor", "Accessory"]
    brands = ["A", "B", "C", "D", "E", "F"]

    for i in range(n):
        entity = f"<http://example.org/item/{i // 6}>"
        p = predicates[i % len(predicates)]
        if p.endswith("/type>"):
            o = "<http://example.org/Product>"
        elif p.endswith("/name>"):
            o = f"Item-{i // 6}"
        elif p.endswith("/age>"):
            o = str((i // 6) % 100)
        elif p.endswith("/category>"):
            o = random.choice(categories)
        elif p.endswith("/brand>"):
            o = random.choice(brands)
        else:
            o = str(round(random.uniform(9.99, 2999.99), 2))
        yield (entity, p, o)


CASE_GENERATORS: Dict[str, Callable[[int], Iterable[Triple]]] = {
    "high": gen_high_cardinality,
    "skewed": gen_low_cardinality_skewed,
    "mixed": gen_mixed_realistic,
}


def run_case(case_name: str, triples: List[Triple], lookup_queries: int, track_spo: bool) -> Dict[str, object]:
    catalog = Catalog(track_spo=track_spo)
    n = len(triples)

    t0 = time.perf_counter()
    added = catalog.add_batch(triples)
    t1 = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        pkl = Path(td) / f"{case_name}.pkl"
        t2 = time.perf_counter()
        catalog.save_pickle(pkl)
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        loaded = Catalog.load_pickle(pkl)
        t5 = time.perf_counter()

    keys_s = list(catalog.s_counts.keys())
    keys_p = list(catalog.p_counts.keys())
    keys_o = list(catalog.o_counts.keys())
    keys_sp = list(catalog.sp_counts.keys())
    keys_so = list(catalog.so_counts.keys())
    keys_po = list(catalog.po_counts.keys())

    checksum = 0
    t6 = time.perf_counter()
    for i in range(lookup_queries):
        mode = i % 6
        if mode == 0 and keys_s:
            checksum += catalog.count(subject=keys_s[i % len(keys_s)])
        elif mode == 1 and keys_p:
            checksum += catalog.count(predicate=keys_p[i % len(keys_p)])
        elif mode == 2 and keys_o:
            checksum += catalog.count(object_value=keys_o[i % len(keys_o)])
        elif mode == 3 and keys_sp:
            s, p = keys_sp[i % len(keys_sp)]
            checksum += catalog.count(subject=s, predicate=p)
        elif mode == 4 and keys_so:
            s, o = keys_so[i % len(keys_so)]
            checksum += catalog.count(subject=s, object_value=o)
        elif mode == 5 and keys_po:
            p, o = keys_po[i % len(keys_po)]
            checksum += catalog.count(predicate=p, object_value=o)
    t7 = time.perf_counter()

    build_s = t1 - t0
    save_s = t3 - t2
    load_s = t5 - t4
    lookup_s = t7 - t6
    result = {
        "case": case_name,
        "n": n,
        "track_spo": track_spo,
        "build_time_s": build_s,
        "add_rate_triples_per_s": (added / build_s) if build_s > 0 else 0.0,
        "pickle_save_s": save_s,
        "pickle_load_s": load_s,
        "lookup_queries": lookup_queries,
        "lookup_qps": (lookup_queries / lookup_s) if lookup_s > 0 else 0.0,
        "checksum": checksum,
        "summary": loaded.summary(),
    }
    return result


def load_nt_triples(path: Path, max_lines: int | None = None) -> List[Triple]:
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            triple = parse_nt_triple_line(line)
            if triple:
                triples.append((triple.subject, triple.predicate, triple.object_value))
                if max_lines is not None and len(triples) >= max_lines:
                    break
    return triples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark catalog build, pickle roundtrip, and count lookup throughput.")
    parser.add_argument("--n", type=int, default=200_000, help="Number of triples per case")
    parser.add_argument("--lookups", type=int, default=200_000, help="Number of count lookups per case")
    parser.add_argument("--cases", default="high,skewed,mixed", help="Comma-separated cases: high,skewed,mixed")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--track-spo", action="store_true", help="Also track full (s,p,o) keys")
    parser.add_argument("--input-nt", default=None, help="Optional path to a real .nt file to benchmark")
    parser.add_argument("--nt-max-lines", type=int, default=None, help="Optional cap on parsed triples from --input-nt")
    parser.add_argument("--output-json", default=None, help="Optional path to save JSON results")
    parser.add_argument("--output-csv", default=None, help="Optional path to save CSV results")
    return parser.parse_args()


def print_result(result: Dict[str, object]) -> None:
    print(f"\nCASE={result['case']} N={result['n']} track_spo={result['track_spo']}")
    print(
        f"build_time_s={result['build_time_s']:.6f} "
        f"add_rate_triples_per_s={result['add_rate_triples_per_s']:,.0f}"
    )
    print(
        f"pickle_save_s={result['pickle_save_s']:.6f} "
        f"pickle_load_s={result['pickle_load_s']:.6f}"
    )
    print(f"lookup_qps={result['lookup_qps']:,.0f} checksum={result['checksum']}")
    print(f"summary={result['summary']}")


def write_json(path: str, results: List[Dict[str, object]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


def write_csv(path: str, results: List[Dict[str, object]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        row = dict(r)
        summary = row.pop("summary", {})
        for k, v in summary.items():
            row[f"summary_{k}"] = v
        rows.append(row)

    if not rows:
        return
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    selected = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in selected if c not in CASE_GENERATORS]
    if unknown:
        raise ValueError(f"Unknown case(s): {unknown}. Allowed: {sorted(CASE_GENERATORS)}")

    print("Running catalog benchmarks...")
    print(f"cases={selected} n={args.n} lookups={args.lookups} seed={args.seed} track_spo={args.track_spo}")
    if args.input_nt:
        print(f"input_nt={args.input_nt} nt_max_lines={args.nt_max_lines}")

    results: List[Dict[str, object]] = []
    for case in selected:
        gen_fn = CASE_GENERATORS[case]
        triples = list(gen_fn(args.n))
        result = run_case(case, triples, args.lookups, args.track_spo)
        results.append(result)
        print_result(result)

    if args.input_nt:
        nt_path = Path(args.input_nt)
        if not nt_path.exists():
            raise FileNotFoundError(f"--input-nt file not found: {nt_path}")
        nt_triples = load_nt_triples(nt_path, args.nt_max_lines)
        nt_case_name = f"nt:{nt_path.name}"
        nt_result = run_case(nt_case_name, nt_triples, args.lookups, args.track_spo)
        results.append(nt_result)
        print_result(nt_result)

    if args.output_json:
        write_json(args.output_json, results)
        print(f"\nSaved JSON results to {args.output_json}")

    if args.output_csv:
        write_csv(args.output_csv, results)
        print(f"Saved CSV results to {args.output_csv}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

