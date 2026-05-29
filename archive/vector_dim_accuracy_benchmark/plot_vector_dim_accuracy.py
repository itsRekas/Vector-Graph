#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot vector-dimension accuracy benchmark outputs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Benchmark JSON from run_vector_dim_accuracy_benchmark.py",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output filename prefix (default: input stem in same folder).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    per_dim = payload.get("per_dimension", [])
    if not per_dim:
        raise RuntimeError("No per-dimension data found in input JSON.")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required for plotting.") from exc

    dims = [int(row["dimension"]) for row in per_dim]
    acc = [float(row["overall_accuracy_pct"]) for row in per_dim]
    threshold = float(payload.get("accuracy_threshold_pct", 95.0))

    prefix = Path(args.output_prefix) if args.output_prefix else input_path.with_name(input_path.stem)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Plot 1: overall accuracy by dimension.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dims, acc, marker="o", linewidth=2.0, label="Overall accuracy %")
    ax.axhline(threshold, linestyle="--", linewidth=1.2, label=f"Threshold {threshold:.1f}%")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Vector Dimension vs Accuracy")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    acc_path = prefix.with_name(f"{prefix.name}_accuracy_vs_dim.png")
    fig.savefig(acc_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: V4-style count comparison for each query (grouped bars by dimension + baseline).
    query_ids = [row["query_id"] for row in per_dim[0].get("queries", [])]
    if query_ids:
        fig2, ax2 = plt.subplots(figsize=(max(12, len(query_ids) * 1.8), 6))
        x = np.arange(len(query_ids))
        n_dims = len(dims)
        total_bars = n_dims + 1  # baseline + all dims
        width = 0.8 / max(total_bars, 1)

        baseline_counts = [per_dim[0]["queries"][i]["baseline_count"] for i in range(len(query_ids))]
        baseline_pos = x - (total_bars / 2.0 - 0.5) * width
        ax2.bar(baseline_pos, baseline_counts, width, label="SPARQL baseline", alpha=0.9)

        for idx, row in enumerate(per_dim):
            vector_counts = [q["vector_count"] for q in row.get("queries", [])]
            bar_pos = baseline_pos + (idx + 1) * width
            ax2.bar(bar_pos, vector_counts, width, label=f"dim={row['dimension']}", alpha=0.75)

        ax2.set_xlabel("Queries")
        ax2.set_ylabel("Result count")
        ax2.set_title("Sample Query Counts: SPARQL Baseline vs Vector Dims")
        ax2.set_xticks(x)
        ax2.set_xticklabels(query_ids)
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend(ncol=2)
        plt.tight_layout()
        counts_path = prefix.with_name(f"{prefix.name}_query_counts.png")
        fig2.savefig(counts_path, dpi=220, bbox_inches="tight")
        plt.close(fig2)
    else:
        counts_path = None

    print(f"Saved: {acc_path}")
    if counts_path is not None:
        print(f"Saved: {counts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
