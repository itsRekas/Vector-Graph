#!/usr/bin/env python3
"""Plot F1 vs latency for PR dynamic sweep (latest summary CSV per dimension).

Collects the newest vector_dim_pr_*_summary.csv under results/PR_dynamic_sweep/dim*/
and writes a new PNG without modifying benchmark data or older plots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from plot_vector_dim_accuracy import (
    plot_f1_vs_latency,
    plot_raw_f1_vs_latency,
    plot_raw_hit_f1_vs_latency,
)


DEFAULT_DIMS = (8, 16, 32, 64, 128, 256, 384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "PR_dynamic_sweep",
        help="PR_dynamic_sweep results root",
    )
    parser.add_argument(
        "--dims",
        default=",".join(str(d) for d in DEFAULT_DIMS),
        help="Comma-separated embedding dimensions to include",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix (default: sweep-dir/pr_dynamic_sweep_f1_vs_latency_<utc>)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional chart title override (post-filter F1 chart only)",
    )
    parser.add_argument(
        "--raw-title",
        default=None,
        help="Optional title override for raw F1 chart",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Only write post-filter F1 vs latency chart",
    )
    parser.add_argument(
        "--skip-raw-hit",
        action="store_true",
        help="Skip raw hit-count F1 vs latency chart",
    )
    parser.add_argument(
        "--raw-hit-title",
        default=None,
        help="Optional title override for raw hit-count F1 chart",
    )
    return parser.parse_args()


def latest_summary_for_dim(sweep_dir: Path, dim: int) -> Path:
    dim_dir = sweep_dir / f"dim{dim}"
    if not dim_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {dim_dir}")

    candidates = sorted(dim_dir.glob("vector_dim_pr_*_summary.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No summary CSV found under {dim_dir}")

    return candidates[-1]


def main() -> int:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    dims = [int(part.strip()) for part in args.dims.split(",") if part.strip()]

    summary_paths = []
    for dim in dims:
        path = latest_summary_for_dim(sweep_dir, dim)
        summary_paths.append(path)
        print(f"dim{dim}: {path.name}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.output_prefix or (sweep_dir / f"pr_dynamic_sweep_rerun_{ts}")
    prefix = prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    title = args.title or "F1 vs average latency"

    out_path = plot_f1_vs_latency(summary_paths, prefix, title=title)
    print(f"Saved: {out_path}")

    if not args.skip_raw:
        raw_title = args.raw_title or "raw F1 vs average latency"
        raw_path = plot_raw_f1_vs_latency(summary_paths, prefix, title=raw_title)
        print(f"Saved: {raw_path}")

    if not args.skip_raw_hit:
        raw_hit_title = args.raw_hit_title or "raw hit F1 vs average latency"
        raw_hit_path = plot_raw_hit_f1_vs_latency(
            summary_paths, prefix, title=raw_hit_title
        )
        print(f"Saved: {raw_hit_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
