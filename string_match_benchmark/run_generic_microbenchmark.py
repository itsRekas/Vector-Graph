#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import string
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Dict, List, Sequence, Tuple

import numpy as np


PART_TO_INDEX = {"s": 0, "p": 1, "o": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone micro-benchmark: exact pattern-style string match vs "
            "embedding self-match over synthetic data."
        )
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=1_000_000,
        help="Number of synthetic records to generate.",
    )
    parser.add_argument(
        "--constrained-parts",
        default="o",
        choices=["s", "p", "o", "sp", "so", "po", "spo"],
        help="Which triple parts are checked (pattern-style constants).",
    )
    parser.add_argument(
        "--mismatch-ratio",
        type=float,
        default=0.2,
        help="Fraction of rows forced to mismatch on one constrained part.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension for synthetic embedding vectors.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.999,
        help="Cosine threshold used by embedding match checks.",
    )
    parser.add_argument("--runs", type=int, default=10, help="Measured runs.")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Warmup runs per matcher (not recorded).",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=8,
        help="Minimum generated string length.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=64,
        help="Maximum generated string length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic workload generation.",
    )
    parser.add_argument(
        "--alphabet",
        default=string.ascii_letters + string.digits,
        help="Characters used for synthetic strings.",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory for JSON/CSV outputs.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log progress every N rows during generation/embedding (0 disables).",
    )
    return parser.parse_args()


def _ci95(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1))
    return 1.96 * std / math.sqrt(len(values))


def _safe_p(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def _random_text(rng: random.Random, min_len: int, max_len: int, alphabet: str) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(length))


def _different_text(
    rng: random.Random,
    original: str,
    min_len: int,
    max_len: int,
    alphabet: str,
) -> str:
    candidate = original
    while candidate == original:
        candidate = _random_text(rng, min_len=min_len, max_len=max_len, alphabet=alphabet)
    return candidate


def _embed_text_hash(text: str, dim: int) -> np.ndarray:
    # Deterministic hash-based embedding keeps this benchmark standalone.
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def _embed_texts(
    texts: np.ndarray,
    dim: int,
    log_every: int = 0,
    label: str = "",
) -> np.ndarray:
    out = np.empty((texts.shape[0], dim), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i] = _embed_text_hash(str(text), dim=dim)
        if log_every > 0 and (i + 1) % log_every == 0:
            prefix = f"{label}: " if label else ""
            print(f"{prefix}embedded {i + 1:,}/{texts.shape[0]:,}")
    return out


def _string_match_count(
    left_parts: Tuple[np.ndarray, np.ndarray, np.ndarray],
    right_parts: Tuple[np.ndarray, np.ndarray, np.ndarray],
    constrained_indices: Tuple[int, ...],
) -> int:
    count = 0
    n = left_parts[0].shape[0]
    for i in range(n):
        ok = True
        for idx in constrained_indices:
            if left_parts[idx][i] != right_parts[idx][i]:
                ok = False
                break
        if ok:
            count += 1
    return count


def _embedding_match_count(
    left_embs: Dict[int, np.ndarray],
    right_embs: Dict[int, np.ndarray],
    constrained_indices: Tuple[int, ...],
    similarity_threshold: float,
) -> int:
    n = next(iter(left_embs.values())).shape[0]
    mask = np.ones(n, dtype=bool)
    for idx in constrained_indices:
        scores = np.sum(left_embs[idx] * right_embs[idx], axis=1)
        mask &= scores >= similarity_threshold
    return int(np.count_nonzero(mask))


def _build_synthetic_pairs(
    num_records: int,
    constrained_indices: Tuple[int, ...],
    mismatch_ratio: float,
    rng: random.Random,
    min_len: int,
    max_len: int,
    alphabet: str,
    log_every: int = 0,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], int]:
    left_subject = np.empty(num_records, dtype=object)
    left_predicate = np.empty(num_records, dtype=object)
    left_object = np.empty(num_records, dtype=object)
    right_subject = np.empty(num_records, dtype=object)
    right_predicate = np.empty(num_records, dtype=object)
    right_object = np.empty(num_records, dtype=object)

    forced_mismatches = int(round(num_records * mismatch_ratio))
    mismatch_indices = set(rng.sample(range(num_records), forced_mismatches)) if forced_mismatches > 0 else set()

    for i in range(num_records):
        ls = _random_text(rng, min_len=min_len, max_len=max_len, alphabet=alphabet)
        lp = _random_text(rng, min_len=min_len, max_len=max_len, alphabet=alphabet)
        lo = _random_text(rng, min_len=min_len, max_len=max_len, alphabet=alphabet)
        left_subject[i], left_predicate[i], left_object[i] = ls, lp, lo

        rs, rp, ro = ls, lp, lo
        if i in mismatch_indices and constrained_indices:
            target_idx = rng.choice(constrained_indices)
            if target_idx == 0:
                rs = _different_text(rng, ls, min_len=min_len, max_len=max_len, alphabet=alphabet)
            elif target_idx == 1:
                rp = _different_text(rng, lp, min_len=min_len, max_len=max_len, alphabet=alphabet)
            else:
                ro = _different_text(rng, lo, min_len=min_len, max_len=max_len, alphabet=alphabet)

        right_subject[i], right_predicate[i], right_object[i] = rs, rp, ro

        if log_every > 0 and (i + 1) % log_every == 0:
            print(f"Generated {i + 1:,}/{num_records:,} records")

    left_parts = (left_subject, left_predicate, left_object)
    right_parts = (right_subject, right_predicate, right_object)
    expected_hits = num_records - forced_mismatches
    return left_parts, right_parts, expected_hits


def _summarize_runs(ms_values: List[float], throughput_values: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": float(np.mean(ms_values)) if ms_values else 0.0,
        "p50_ms": _safe_p(ms_values, 50),
        "p95_ms": _safe_p(ms_values, 95),
        "min_ms": float(min(ms_values)) if ms_values else 0.0,
        "max_ms": float(max(ms_values)) if ms_values else 0.0,
        "ci95_ms": _ci95(ms_values),
        "mean_cmp_per_sec": float(np.mean(throughput_values)) if throughput_values else 0.0,
        "p50_cmp_per_sec": _safe_p(throughput_values, 50),
        "p95_cmp_per_sec": _safe_p(throughput_values, 95),
    }


def main() -> int:
    args = parse_args()
    if args.num_records <= 0:
        raise ValueError("--num-records must be positive")
    if args.min_len <= 0:
        raise ValueError("--min-len must be positive")
    if args.max_len < args.min_len:
        raise ValueError("--max-len must be >= --min-len")
    if not (0.0 <= args.mismatch_ratio <= 1.0):
        raise ValueError("--mismatch-ratio must be between 0 and 1")
    if args.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be positive")

    constrained_indices = tuple(PART_TO_INDEX[p] for p in args.constrained_parts)
    rng = random.Random(args.seed)

    print("Generating synthetic paired records...")
    left_parts, right_parts, expected_hits = _build_synthetic_pairs(
        num_records=args.num_records,
        constrained_indices=constrained_indices,
        mismatch_ratio=args.mismatch_ratio,
        rng=rng,
        min_len=args.min_len,
        max_len=args.max_len,
        alphabet=args.alphabet,
        log_every=args.log_every,
    )
    print(f"Generated {args.num_records:,} records. Expected true matches: {expected_hits:,}")

    left_embs: Dict[int, np.ndarray] = {}
    right_embs: Dict[int, np.ndarray] = {}
    print("Precomputing embeddings (outside timed kernel)...")
    for idx in constrained_indices:
        part_label = ("subject", "predicate", "object")[idx]
        left_embs[idx] = _embed_texts(
            left_parts[idx],
            dim=args.embedding_dim,
            log_every=args.log_every,
            label=f"{part_label}/left",
        )
        right_embs[idx] = _embed_texts(
            right_parts[idx],
            dim=args.embedding_dim,
            log_every=args.log_every,
            label=f"{part_label}/right",
        )

    print("Running warmups...")
    for _ in range(args.warmup_runs):
        _ = _string_match_count(left_parts, right_parts, constrained_indices)
        _ = _embedding_match_count(left_embs, right_embs, constrained_indices, args.similarity_threshold)

    run_rows: List[Dict[str, float]] = []
    print("Running measured iterations...")
    for run_idx in range(1, args.runs + 1):
        t0 = perf_counter_ns()
        string_hits = _string_match_count(left_parts, right_parts, constrained_indices)
        t1 = perf_counter_ns()
        embedding_hits = _embedding_match_count(
            left_embs,
            right_embs,
            constrained_indices,
            args.similarity_threshold,
        )
        t2 = perf_counter_ns()

        string_ms = (t1 - t0) / 1_000_000.0
        embedding_ms = (t2 - t1) / 1_000_000.0
        string_cmp_per_sec = args.num_records / ((t1 - t0) / 1_000_000_000.0)
        embedding_cmp_per_sec = args.num_records / ((t2 - t1) / 1_000_000_000.0)
        run_rows.append(
            {
                "run": float(run_idx),
                "string_ms": string_ms,
                "embedding_ms": embedding_ms,
                "string_hits": float(string_hits),
                "embedding_hits": float(embedding_hits),
                "string_cmp_per_sec": string_cmp_per_sec,
                "embedding_cmp_per_sec": embedding_cmp_per_sec,
            }
        )
        print(
            f"Run {run_idx:02d}/{args.runs} | "
            f"string={string_ms:.3f} ms ({string_hits} hits) | "
            f"embedding={embedding_ms:.3f} ms ({embedding_hits} hits)"
        )

    string_ms_values = [r["string_ms"] for r in run_rows]
    embedding_ms_values = [r["embedding_ms"] for r in run_rows]
    string_tput_values = [r["string_cmp_per_sec"] for r in run_rows]
    embedding_tput_values = [r["embedding_cmp_per_sec"] for r in run_rows]

    summary_string = _summarize_runs(string_ms_values, string_tput_values)
    summary_embedding = _summarize_runs(embedding_ms_values, embedding_tput_values)
    mean_ratio = (
        summary_embedding["mean_ms"] / summary_string["mean_ms"]
        if summary_string["mean_ms"] > 0
        else float("inf")
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"generic_string_vs_embedding_{ts}"
    json_path = out_dir / f"{base_name}.json"
    csv_path = out_dir / f"{base_name}.csv"

    payload = {
        "config": {
            "num_records": args.num_records,
            "constrained_parts": args.constrained_parts,
            "mismatch_ratio": args.mismatch_ratio,
            "embedding_dim": args.embedding_dim,
            "similarity_threshold": args.similarity_threshold,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "seed": args.seed,
        },
        "dataset": {
            "expected_true_matches": expected_hits,
            "expected_mismatches": args.num_records - expected_hits,
        },
        "summary": {
            "string": summary_string,
            "embedding": summary_embedding,
            "embedding_over_string_mean_time_ratio": mean_ratio,
        },
        "runs": run_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "string_ms",
                "embedding_ms",
                "string_hits",
                "embedding_hits",
                "string_cmp_per_sec",
                "embedding_cmp_per_sec",
            ],
        )
        writer.writeheader()
        for row in run_rows:
            writer.writerow(row)

    print("\nBenchmark complete.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(
        "Mean latency ratio (embedding/string): "
        f"{payload['summary']['embedding_over_string_mean_time_ratio']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
