#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot vector-dimension accuracy benchmark outputs."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Benchmark JSON from run_vector_dim_accuracy_benchmark.py",
    )
    parser.add_argument(
        "--summary-inputs",
        nargs="+",
        default=None,
        help="One or more *_summary.csv files for grouped bar plots.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional output filename prefix (default: input stem in same folder).",
    )
    args = parser.parse_args()
    if not args.input and not args.summary_inputs:
        parser.error("Provide --input (JSON) or --summary-inputs (CSV files).")
    return args


def resolve_prefix(
    output_prefix: str | None, input_path: Path | None, summary_inputs: Iterable[Path] | None
) -> Path:
    if output_prefix:
        return Path(output_prefix)
    if input_path is not None:
        return input_path.with_name(input_path.stem)
    first_summary = next(iter(summary_inputs or []), None)
    if first_summary is None:
        raise RuntimeError("Could not determine output prefix.")
    return first_summary.with_name(first_summary.stem.replace("_summary", ""))


def plot_from_json(input_path: Path, prefix: Path) -> list[Path]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    per_dim = payload.get("per_dimension", [])
    if not per_dim:
        raise RuntimeError("No per-dimension data found in input JSON.")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    dims = [int(row["dimension"]) for row in per_dim]
    avg_precision = [float(row["avg_precision"]) for row in per_dim]
    avg_recall = [float(row["avg_recall"]) for row in per_dim]
    threshold_pct = float(payload.get("accuracy_threshold_pct", 95.0))
    threshold = threshold_pct / 100.0

    saved_paths: list[Path] = []

    # Plot 1: Average precision/recall by dimension (primary figure).
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dims, avg_precision, marker="o", linewidth=2.0, label="Average precision")
    ax.plot(dims, avg_recall, marker="o", linewidth=2.0, label="Average recall")
    ax.axhline(threshold, linestyle="--", linewidth=1.2, label=f"Threshold {threshold_pct:.1f}%")
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Average Precision/Recall vs Embedding Dimension")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    pr_path = prefix.with_name(f"{prefix.name}_precision_recall_vs_dim.png")
    fig.savefig(pr_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(pr_path)

    # Plot 2: Bucket-level averages (sp*, *po, s*o).
    bucket_names = sorted(
        {
            bucket_name
            for row in per_dim
            for bucket_name in row.get("bucket_metrics", {}).keys()
        }
    )
    if bucket_names:
        fig2, axes = plt.subplots(1, len(bucket_names), figsize=(6 * len(bucket_names), 5), squeeze=False)
        for i, bucket_name in enumerate(bucket_names):
            ax_b = axes[0][i]
            p_vals = [
                float(row.get("bucket_metrics", {}).get(bucket_name, {}).get("avg_precision", 0.0))
                for row in per_dim
            ]
            r_vals = [
                float(row.get("bucket_metrics", {}).get(bucket_name, {}).get("avg_recall", 0.0))
                for row in per_dim
            ]
            ax_b.plot(dims, p_vals, marker="o", linewidth=1.8, label="Avg precision")
            ax_b.plot(dims, r_vals, marker="o", linewidth=1.8, label="Avg recall")
            ax_b.axhline(threshold, linestyle="--", linewidth=1.0)
            ax_b.set_title(f"Bucket {bucket_name}")
            ax_b.set_xlabel("Embedding dimension")
            ax_b.set_ylabel("Score (0-1)")
            ax_b.set_ylim(0, 1.05)
            ax_b.grid(alpha=0.3)
            ax_b.legend()
        plt.tight_layout()
        bucket_path = prefix.with_name(f"{prefix.name}_bucket_precision_recall.png")
        fig2.savefig(bucket_path, dpi=220, bbox_inches="tight")
        plt.close(fig2)
        saved_paths.append(bucket_path)

    return saved_paths


def _read_summary_row(summary_path: Path) -> dict[str, str]:
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly 1 data row in summary: {summary_path}")
    return rows[0]


def plot_grouped_bars_from_summaries(summary_paths: list[Path], prefix: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required for plotting.") from exc

    rows = []
    for summary_path in summary_paths:
        row = _read_summary_row(summary_path)
        row["_source_file"] = str(summary_path)
        rows.append(row)

    rows.sort(key=lambda r: int(r["dimension"]))
    dims = [int(r["dimension"]) for r in rows]
    query_totals = {int(r["queries_total"]) for r in rows}
    query_total_label = str(next(iter(query_totals))) if len(query_totals) == 1 else "mixed"

    categories = ["sp*", "s*o", "*po", "total"]
    precision_cols = [
        "sp*_avg_precision",
        "s*o_avg_precision",
        "*po_avg_precision",
        "avg_precision",
    ]
    recall_cols = [
        "sp*_avg_recall",
        "s*o_avg_recall",
        "*po_avg_recall",
        "avg_recall",
    ]

    x = np.arange(len(dims))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    saved_paths: list[Path] = []

    fig_p, ax_p = plt.subplots(figsize=(10, 5))
    for idx, (label, col_name) in enumerate(zip(categories, precision_cols)):
        values = [float(r[col_name]) for r in rows]
        ax_p.bar(x + offsets[idx], values, width=width, label=label, color=colors[idx])
    ax_p.set_xticks(x)
    ax_p.set_xticklabels([str(d) for d in dims])
    ax_p.set_xlabel("Embedding dimension")
    ax_p.set_ylabel("Precision")
    ax_p.set_ylim(0, 1.05)
    ax_p.set_title(f"Precision by Dimension ({query_total_label} queries)")
    ax_p.grid(axis="y", alpha=0.3)
    ax_p.legend()
    plt.tight_layout()
    precision_path = prefix.with_name(f"{prefix.name}_precision_by_dim_grouped_bars.png")
    fig_p.savefig(precision_path, dpi=220, bbox_inches="tight")
    plt.close(fig_p)
    saved_paths.append(precision_path)

    fig_r, ax_r = plt.subplots(figsize=(10, 5))
    for idx, (label, col_name) in enumerate(zip(categories, recall_cols)):
        values = [float(r[col_name]) for r in rows]
        ax_r.bar(x + offsets[idx], values, width=width, label=label, color=colors[idx])
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([str(d) for d in dims])
    ax_r.set_xlabel("Embedding dimension")
    ax_r.set_ylabel("Recall")
    ax_r.set_ylim(0, 1.05)
    ax_r.set_title(f"Recall by Dimension ({query_total_label} queries)")
    ax_r.grid(axis="y", alpha=0.3)
    ax_r.legend()
    plt.tight_layout()
    recall_path = prefix.with_name(f"{prefix.name}_recall_by_dim_grouped_bars.png")
    fig_r.savefig(recall_path, dpi=220, bbox_inches="tight")
    plt.close(fig_r)
    saved_paths.append(recall_path)

    return saved_paths


def main() -> int:
    args = parse_args()
    input_path = Path(args.input) if args.input else None
    summary_paths = [Path(p) for p in (args.summary_inputs or [])]
    prefix = resolve_prefix(args.output_prefix, input_path, summary_paths)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path]
    if summary_paths:
        saved_paths = plot_grouped_bars_from_summaries(summary_paths, prefix)
    else:
        if input_path is None:
            raise RuntimeError("Provide --input for JSON plotting mode.")
        saved_paths = plot_from_json(input_path, prefix)

    for path in saved_paths:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
