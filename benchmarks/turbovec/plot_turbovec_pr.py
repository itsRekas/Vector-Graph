#!/usr/bin/env python3
"""Plot TurboVec precision/recall per bit-width with the Milvus HNSW reference.

Reads the JSON produced by run_turbovec_pr_benchmark.py (per_dimension entries,
one per bit-width) and draws a grouped bar chart of avg precision and recall,
plus dashed reference lines for the stored Milvus dim-8 baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TurboVec PR benchmark outputs.")
    parser.add_argument("--input", required=True, help="turbovec_pr_<ts>.json")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix (default: input stem in same folder).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    rows = payload.get("per_dimension", [])
    if not rows:
        raise RuntimeError("No per_dimension entries in input JSON.")
    rows = sorted(rows, key=lambda r: int(r.get("bit_width", 0)))

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required for plotting.") from exc

    labels = [f"{int(r['bit_width'])}-bit\n({r.get('footprint_bytes', 0):.0f} B/vec)" for r in rows]
    precision = [float(r["avg_precision"]) for r in rows]
    recall = [float(r["avg_recall"]) for r in rows]

    threshold = float(payload.get("accuracy_threshold_pct", 95.0)) / 100.0
    ref = payload.get("milvus_reference", {})
    ref_p = float(ref.get("avg_precision", 0.0))
    ref_r = float(ref.get("avg_recall", 0.0))

    x = np.arange(len(rows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_p = ax.bar(x - width / 2, precision, width, label="Avg precision", color="#4c78a8")
    bars_r = ax.bar(x + width / 2, recall, width, label="Avg recall", color="#54a24b")

    for bars in (bars_p, bars_r):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height * 100:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(ref_p, linestyle="--", linewidth=1.2, color="#4c78a8",
               label=f"Milvus precision {ref_p * 100:.1f}%")
    ax.axhline(ref_r, linestyle="--", linewidth=1.2, color="#54a24b",
               label=f"Milvus recall {ref_r * 100:.1f}%")
    ax.axhline(threshold, linestyle=":", linewidth=1.0, color="#e45756",
               label=f"Threshold {threshold * 100:.0f}%")

    dim = payload.get("dimension", "?")
    k = payload.get("k", "?")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"TurboVec vs Milvus HNSW - precision/recall (dim={dim}, k={k})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    prefix = Path(args.output_prefix) if args.output_prefix else input_path.with_name(input_path.stem)
    out_path = prefix.with_name(f"{prefix.name}_precision_recall_by_bit.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
