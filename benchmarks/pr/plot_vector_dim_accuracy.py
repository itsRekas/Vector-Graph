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
    parser.add_argument(
        "--f1-latency",
        action="store_true",
        help="With --summary-inputs, also write F1 vs latency line chart.",
    )
    parser.add_argument(
        "--f1-latency-only",
        action="store_true",
        help="With --summary-inputs, only write F1 vs latency (skip grouped bars).",
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
    has_raw = all("avg_raw_precision" in row and "avg_raw_recall" in row for row in per_dim)
    avg_raw_precision = [float(row["avg_raw_precision"]) for row in per_dim] if has_raw else []
    avg_raw_recall = [float(row["avg_raw_recall"]) for row in per_dim] if has_raw else []
    threshold_pct = float(payload.get("accuracy_threshold_pct", 95.0))
    threshold = threshold_pct / 100.0

    saved_paths: list[Path] = []

    # Plot 1: Average precision/recall by dimension (primary figure).
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dims, avg_precision, marker="o", linewidth=2.0, label="Average precision")
    ax.plot(dims, avg_recall, marker="o", linewidth=2.0, label="Average recall")
    if has_raw:
        ax.plot(
            dims,
            avg_raw_precision,
            marker="s",
            linewidth=1.8,
            linestyle="--",
            label="Raw avg precision",
        )
        ax.plot(
            dims,
            avg_raw_recall,
            marker="s",
            linewidth=1.8,
            linestyle="--",
            label="Raw avg recall",
        )
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


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _per_query_path_from_summary(summary_path: Path) -> Path:
    return summary_path.with_name(summary_path.name.replace("_summary.csv", "_per_query.csv"))


def _avg_raw_hit_precision(per_query_path: Path) -> float:
    """Raw precision using raw_hit_count (not deduplicated binding count)."""
    precisions: list[float] = []
    with per_query_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_hit_count = int(row["raw_hit_count"])
            raw_tp = int(row["raw_tp"])
            precisions.append(raw_tp / raw_hit_count if raw_hit_count else 0.0)
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def _plot_f1_vs_latency_rows(
    rows: list[dict[str, str]],
    prefix: Path,
    *,
    title: str | None = None,
    precision_col: str = "avg_precision",
    recall_col: str = "avg_recall",
    value_col: str | None = None,
    output_suffix: str = "f1_vs_latency",
    ylabel: str = "F1@K",
) -> Path:
    """Metric vs average vector query latency (reference-style line chart)."""
    try:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc

    for row in rows:
        if value_col is not None:
            if value_col not in row:
                source = row.get("_source_file", "<unknown>")
                raise RuntimeError(f"Row missing {value_col!r}: {source}")
        elif precision_col not in row or recall_col not in row:
            source = row.get("_source_file", "<unknown>")
            raise RuntimeError(
                f"Row missing {precision_col!r} or {recall_col!r}: {source}"
            )

    rows = sorted(rows, key=lambda r: int(r["dimension"]))
    dims = [int(r["dimension"]) for r in rows]
    if value_col is not None:
        y_vals = [float(r[value_col]) for r in rows]
    else:
        y_vals = [
            f1_score(float(r[precision_col]), float(r[recall_col])) for r in rows
        ]
    latency_ms = [float(r["avg_vector_query_seconds"]) * 1000.0 for r in rows]

    if title is None:
        if value_col is not None:
            title = "Recall vs average latency"
        else:
            title = (
                "raw F1 vs average latency"
                if precision_col.startswith("avg_raw_")
                else "F1 vs average latency"
            )

    palette = cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(
        latency_ms,
        y_vals,
        color="#94a3b8",
        linewidth=1.6,
        zorder=1,
    )
    for idx, (dim, x_ms, y_val) in enumerate(zip(dims, latency_ms, y_vals)):
        color = palette[idx % len(palette)]
        ax.plot(
            x_ms,
            y_val,
            marker="o",
            markersize=9,
            linestyle="",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"({dim}x3)d",
            zorder=2,
        )

    ax.set_xlabel("Average latency for one query, ms")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        title="Dimension",
        loc="lower right",
        ncol=2,
        fontsize=9,
        title_fontsize=9,
        framealpha=0.95,
    )

    y_min = min(y_vals)
    y_max = max(y_vals)
    y_pad = max(0.005, (y_max - y_min) * 0.25)
    ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))

    x_min, x_max = min(latency_ms), max(latency_ms)
    x_pad = max(1.0, (x_max - x_min) * 0.15)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    plt.tight_layout()
    out_path = prefix.with_name(f"{prefix.name}_{output_suffix}.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_f1_vs_latency(
    summary_paths: list[Path],
    prefix: Path,
    *,
    title: str | None = None,
    precision_col: str = "avg_precision",
    recall_col: str = "avg_recall",
    output_suffix: str = "f1_vs_latency",
    ylabel: str = "F1@K",
) -> Path:
    rows = []
    for summary_path in summary_paths:
        row = _read_summary_row(summary_path)
        row["_source_file"] = str(summary_path)
        rows.append(row)
    return _plot_f1_vs_latency_rows(
        rows,
        prefix,
        title=title,
        precision_col=precision_col,
        recall_col=recall_col,
        output_suffix=output_suffix,
        ylabel=ylabel,
    )


def plot_raw_f1_vs_latency(
    summary_paths: list[Path],
    prefix: Path,
    *,
    title: str | None = None,
) -> Path:
    """F1 from raw (pre-filter) precision/recall vs average vector query latency."""
    return plot_f1_vs_latency(
        summary_paths,
        prefix,
        title=title,
        precision_col="avg_raw_precision",
        recall_col="avg_raw_recall",
        output_suffix="raw_f1_vs_latency",
        ylabel="Raw F1@K",
    )


def plot_raw_hit_f1_vs_latency(
    summary_paths: list[Path],
    prefix: Path,
    *,
    title: str | None = None,
) -> Path:
    """F1 from raw recall and precision with raw_hit_count as the denominator."""
    rows = []
    for summary_path in summary_paths:
        row = _read_summary_row(summary_path)
        per_query_path = _per_query_path_from_summary(summary_path)
        if not per_query_path.is_file():
            raise FileNotFoundError(
                f"Per-query CSV not found for summary {summary_path}: {per_query_path}"
            )
        if "avg_raw_recall" not in row:
            raise RuntimeError(f"Summary missing avg_raw_recall: {summary_path}")
        row["avg_raw_hit_precision"] = str(_avg_raw_hit_precision(per_query_path))
        row["_source_file"] = str(summary_path)
        rows.append(row)
    return _plot_f1_vs_latency_rows(
        rows,
        prefix,
        title=title or "raw hit F1 vs average latency",
        precision_col="avg_raw_hit_precision",
        recall_col="avg_raw_recall",
        output_suffix="raw_hit_f1_vs_latency",
        ylabel="Raw hit F1@K",
    )


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
        if args.f1_latency or args.f1_latency_only:
            saved_paths = [plot_f1_vs_latency(summary_paths, prefix)]
        else:
            saved_paths = []
        if not args.f1_latency_only:
            saved_paths.extend(plot_grouped_bars_from_summaries(summary_paths, prefix))
    else:
        if input_path is None:
            raise RuntimeError("Provide --input for JSON plotting mode.")
        saved_paths = plot_from_json(input_path, prefix)

    for path in saved_paths:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
