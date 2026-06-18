#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from vector_endpoint.catalog import parse_nt_triple_line


@dataclass(frozen=True)
class Triple:
    s: str
    p: str
    o: str
    o_type: str  # "uri" or "literal"


Key = Tuple[str, ...]
SampledKey = Tuple[Key, int, Optional[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random SPARQL single-pattern queries in three buckets: "
            "sp*, *po, s*o. Supports legacy stratified or dim-sweep weighted sampling."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("legacy-stratified", "dim-sweep"),
        default="dim-sweep",
        help="Query sampling profile (default: dim-sweep).",
    )
    parser.add_argument(
        "--input-file",
        default="../../data/nts/RLUBM_cleaned.nt",
        help="Path to NT file used for sampling",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sp-count", type=int, default=None, help="Number of sp* queries")
    parser.add_argument("--po-count", type=int, default=None, help="Number of *po queries")
    parser.add_argument("--so-count", type=int, default=None, help="Number of s*o queries")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (profile-specific default if omitted)",
    )
    parser.add_argument(
        "--copy-to",
        default=None,
        help="Optional second output path (e.g. results/ copy of sweep file).",
    )
    parser.add_argument(
        "--stratify-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="legacy-stratified only: sample evenly across result-count strata.",
    )
    return parser.parse_args()


def _apply_profile_defaults(args: argparse.Namespace) -> None:
    if args.profile == "dim-sweep":
        if args.sp_count is None:
            args.sp_count = 250
        if args.po_count is None:
            args.po_count = 500
        if args.so_count is None:
            args.so_count = 50
        if args.out is None:
            args.out = "results/PR_dynamic_sweep/random_queries_dim_sweep.json"
        if args.copy_to is None:
            args.copy_to = "results/random_queries_dim_sweep.json"
    else:
        if args.sp_count is None:
            args.sp_count = 1000
        if args.po_count is None:
            args.po_count = 1000
        if args.so_count is None:
            args.so_count = 1000
        if args.out is None:
            args.out = "results/random_queries_3000_stratified.json"


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


def _bin_exact(count: int) -> str:
    return str(count)


def _bin_po(count: int) -> str:
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    if count <= 25:
        return "11-25"
    if count <= 100:
        return "26-100"
    if count <= 500:
        return "101-500"
    if count <= 2000:
        return "501-2000"
    return "2001+"


def _bin_so(count: int) -> str:
    return "2" if count >= 2 else "1"


BIN_ORDERS: Dict[str, List[str]] = {
    "sp*": ["1", "2", "3", "4", "5", "6", "7"],
    "*po": ["1", "2", "3-5", "6-10", "11-25", "26-100", "101-500", "501-2000", "2001+"],
    "s*o": ["1", "2"],
}

BIN_FN: Dict[str, Callable[[int], str]] = {
    "sp*": _bin_exact,
    "*po": _bin_po,
    "s*o": _bin_so,
}


def sample_unique_keys(
    keys: Sequence[Key],
    count: int,
    rng: random.Random,
    label: str,
) -> List[Key]:
    if len(keys) < count:
        raise ValueError(f"Not enough unique {label} keys: requested={count}, available={len(keys)}")
    idx = list(range(len(keys)))
    rng.shuffle(idx)
    return [keys[i] for i in idx[:count]]


def _sample_without_replacement(
    candidates: List[Tuple[Key, int]],
    count: int,
    rng: random.Random,
    *,
    exclude: set[Key],
) -> List[Tuple[Key, int]]:
    pool = [(k, c) for k, c in candidates if k not in exclude]
    if len(pool) < count:
        raise ValueError(f"Not enough candidates: need {count}, have {len(pool)}")
    rng.shuffle(pool)
    return pool[:count]


def stratified_sample_keys(
    pool: Dict[Key, int],
    count: int,
    rng: random.Random,
    bucket: str,
) -> List[SampledKey]:
    """Sample keys with roughly uniform coverage across result-count bins."""
    bin_fn = BIN_FN[bucket]
    bin_order = BIN_ORDERS[bucket]

    by_bin: Dict[str, List[Tuple[Key, int]]] = defaultdict(list)
    for key, match_count in pool.items():
        by_bin[bin_fn(match_count)].append((key, match_count))

    active_bins = [b for b in bin_order if by_bin.get(b)]
    if not active_bins:
        raise ValueError(f"No keys available for bucket {bucket}")

    per_bin = count // len(active_bins)
    remainder = count % len(active_bins)

    selected: List[SampledKey] = []
    selected_keys: set[Key] = set()

    for i, bin_label in enumerate(active_bins):
        target = per_bin + (1 if i < remainder else 0)
        candidates = by_bin[bin_label][:]
        rng.shuffle(candidates)
        taken = 0
        for key, match_count in candidates:
            if taken >= target:
                break
            if key in selected_keys:
                continue
            selected.append((key, match_count, None))
            selected_keys.add(key)
            taken += 1

    if len(selected) < count:
        remaining = [(k, c) for k, c in pool.items() if k not in selected_keys]
        rng.shuffle(remaining)
        for key, match_count in remaining:
            if len(selected) >= count:
                break
            selected.append((key, match_count, None))
            selected_keys.add(key)

    if len(selected) < count:
        raise ValueError(
            f"Not enough {bucket} keys after stratified sampling: "
            f"requested={count}, selected={len(selected)}"
        )

    rng.shuffle(selected)
    return selected[:count]


def sample_keys(
    pool: Dict[Key, int],
    count: int,
    rng: random.Random,
    bucket: str,
    *,
    stratify: bool,
) -> List[SampledKey]:
    if stratify:
        return stratified_sample_keys(pool, count, rng, bucket)
    keys = sample_unique_keys(list(pool.keys()), count, rng, bucket)
    return [(k, pool[k], None) for k in keys]


def sample_po_dim_sweep(
    pool: Dict[Key, int],
    count: int,
    rng: random.Random,
) -> List[SampledKey]:
    """All count>=100 keys, then even split of remainder across 10-24 and 25-99."""
    large = sorted(((k, c) for k, c in pool.items() if c >= 100), key=lambda x: -x[1])
    if count < len(large):
        raise ValueError(
            f"*po count={count} is smaller than large pool size={len(large)} (count>=100)"
        )

    selected: List[SampledKey] = [(k, c, "large") for k, c in large]
    selected_keys = {k for k, _, _ in selected}

    remainder = count - len(selected)
    n_low = remainder // 2
    n_high = remainder - n_low

    low_candidates = [(k, c) for k, c in pool.items() if 10 <= c <= 24]
    high_candidates = [(k, c) for k, c in pool.items() if 25 <= c <= 99]

    for band, n_need, candidates in (
        ("mid_10_24", n_low, low_candidates),
        ("mid_25_99", n_high, high_candidates),
    ):
        picked = _sample_without_replacement(candidates, n_need, rng, exclude=selected_keys)
        for key, match_count in picked:
            selected.append((key, match_count, band))
            selected_keys.add(key)

    rng.shuffle(selected)
    return selected


def sample_sp_dim_sweep(
    pool: Dict[Key, int],
    count: int,
    rng: random.Random,
) -> List[SampledKey]:
    """Counts 2-7 only, evenly across exact counts with short-bin redistribution."""
    exact_bins = [2, 3, 4, 5, 6, 7]
    by_bin: Dict[int, List[Tuple[Key, int]]] = {
        b: [(k, c) for k, c in pool.items() if c == b] for b in exact_bins
    }

    per_bin = count // len(exact_bins)
    remainder = count % len(exact_bins)
    targets = [per_bin + (1 if i < remainder else 0) for i in range(len(exact_bins))]

    selected: List[SampledKey] = []
    selected_keys: set[Key] = set()
    deficits = 0

    for b, target in zip(exact_bins, targets):
        candidates = by_bin[b][:]
        rng.shuffle(candidates)
        take = min(target, len(candidates))
        deficits += target - take
        for key, match_count in candidates[:take]:
            selected.append((key, match_count, f"count_{b}"))
            selected_keys.add(key)

    if deficits > 0:
        overflow: List[Tuple[Key, int]] = []
        for b in exact_bins:
            for key, match_count in by_bin[b]:
                if key not in selected_keys:
                    overflow.append((key, match_count))
        rng.shuffle(overflow)
        for key, match_count in overflow:
            if deficits <= 0:
                break
            selected.append((key, match_count, f"count_{match_count}"))
            selected_keys.add(key)
            deficits -= 1

    if len(selected) < count:
        raise ValueError(
            f"Not enough sp* keys with count 2-7: requested={count}, selected={len(selected)}"
        )

    rng.shuffle(selected)
    return selected[:count]


def sample_so_dim_sweep(
    pool: Dict[Key, int],
    count: int,
    rng: random.Random,
) -> List[SampledKey]:
    """Prefer all count=2 keys, fill remainder from count=1."""
    count2 = [(k, c) for k, c in pool.items() if c == 2]
    count1 = [(k, c) for k, c in pool.items() if c == 1]

    selected: List[SampledKey] = [(k, c, "count_2") for k, c in count2]
    selected_keys = {k for k, _, _ in selected}

    if len(selected) > count:
        rng.shuffle(selected)
        selected = selected[:count]
        selected_keys = {k for k, _, _ in selected}

    need = count - len(selected)
    if need > 0:
        picked = _sample_without_replacement(count1, need, rng, exclude=selected_keys)
        for key, match_count in picked:
            selected.append((key, match_count, "count_1"))

    rng.shuffle(selected)
    return selected


def _distribution_summary(rows: List[Dict], bucket: str) -> Dict[str, int]:
    bin_fn = BIN_FN[bucket]
    summary: Dict[str, int] = defaultdict(int)
    for row in rows:
        summary[bin_fn(int(row["expected_count"]))] += 1
    return dict(summary)


def _po_band_summary(rows: List[Dict]) -> Dict[str, int]:
    summary: Dict[str, int] = defaultdict(int)
    for row in rows:
        band = row.get("sampling_band")
        if band:
            summary[band] += 1
    return dict(summary)


def _rows_from_samples(
    sp_sampled: List[SampledKey],
    po_sampled: List[SampledKey],
    so_sampled: List[SampledKey],
) -> List[Dict]:
    rows: List[Dict] = []
    for i, item in enumerate(sp_sampled, start=1):
        key, expected, band = item
        s, p = key
        row = {
            "id": f"SP{i:04d}",
            "bucket": "sp*",
            "expected_count": expected,
            "query": _build_sp_query(s, p),
        }
        if band:
            row["sampling_band"] = band
        rows.append(row)

    for i, item in enumerate(po_sampled, start=1):
        key, expected, band = item
        p, o, o_type = key
        row = {
            "id": f"PO{i:04d}",
            "bucket": "*po",
            "expected_count": expected,
            "query": _build_po_query(p, o, o_type),
        }
        if band:
            row["sampling_band"] = band
        rows.append(row)

    for i, item in enumerate(so_sampled, start=1):
        key, expected, band = item
        s, o, o_type = key
        row = {
            "id": f"SO{i:04d}",
            "bucket": "s*o",
            "expected_count": expected,
            "query": _build_so_query(s, o, o_type),
        }
        if band:
            row["sampling_band"] = band
        rows.append(row)

    return rows


def main() -> int:
    args = parse_args()
    _apply_profile_defaults(args)
    input_path = Path(args.input_file)
    triples = load_triples(input_path)
    rng = random.Random(args.seed)

    sp_pool: Dict[Key, int] = defaultdict(int)
    po_pool: Dict[Key, int] = defaultdict(int)
    so_pool: Dict[Key, int] = defaultdict(int)

    for t in triples:
        sp_pool[(t.s, t.p)] += 1
        po_pool[(t.p, t.o, t.o_type)] += 1
        so_pool[(t.s, t.o, t.o_type)] += 1

    if args.profile == "dim-sweep":
        sp_sampled = sample_sp_dim_sweep(sp_pool, args.sp_count, rng)
        po_sampled = sample_po_dim_sweep(po_pool, args.po_count, rng)
        so_sampled = sample_so_dim_sweep(so_pool, args.so_count, rng)
    else:
        sp_sampled = sample_keys(sp_pool, args.sp_count, rng, "sp*", stratify=args.stratify_counts)
        po_sampled = sample_keys(po_pool, args.po_count, rng, "*po", stratify=args.stratify_counts)
        so_sampled = sample_keys(so_pool, args.so_count, rng, "s*o", stratify=args.stratify_counts)

    rows = _rows_from_samples(sp_sampled, po_sampled, so_sampled)

    po_rows = [r for r in rows if r["bucket"] == "*po"]
    payload: Dict = {
        "profile": args.profile,
        "seed": args.seed,
        "input_file": str(input_path),
        "counts": {"sp*": args.sp_count, "*po": args.po_count, "s*o": args.so_count},
        "distribution": {
            "sp*": _distribution_summary([r for r in rows if r["bucket"] == "sp*"], "sp*"),
            "*po": _distribution_summary(po_rows, "*po"),
            "s*o": _distribution_summary([r for r in rows if r["bucket"] == "s*o"], "s*o"),
        },
        "queries": rows,
    }

    if args.profile == "dim-sweep":
        payload["sampling"] = {
            "*po": _po_band_summary(po_rows),
            "sp*": {"min_count": 2, "max_count": 7},
            "s*o": {"prefer_count_2": True},
        }
    else:
        payload["stratify_counts"] = args.stratify_counts

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.copy_to:
        copy_path = Path(args.copy_to)
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, copy_path)

    print(f"Wrote query set: {out_path}")
    if args.copy_to:
        print(f"Copied to: {args.copy_to}")
    print(f"Profile: {args.profile}")
    print(f"Total queries: {len(rows)}")
    for bucket in ("sp*", "*po", "s*o"):
        print(f"  {bucket} distribution: {payload['distribution'][bucket]}")
    if args.profile == "dim-sweep":
        print(f"  *po sampling bands: {payload['sampling']['*po']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
