#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot string-vs-embedding benchmark times with mean and 95% CI."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to benchmark JSON produced by run_string_vs_embedding_benchmark.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: same folder/input name + _line_mean_ci.png)",
    )
    return parser.parse_args()


def ci95(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr, ddof=1))
    return 1.96 * std / math.sqrt(len(values))


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    if not runs:
        raise RuntimeError("No runs found in input JSON.")

    string_vals = np.asarray([row["string_ms"] for row in runs], dtype=float)
    embedding_vals = np.asarray([row["embedding_ms"] for row in runs], dtype=float)
    run_idx = np.arange(1, len(runs) + 1)

    string_mean = float(np.mean(string_vals))
    embedding_mean = float(np.mean(embedding_vals))
    string_ci = ci95(string_vals)
    embedding_ci = ci95(embedding_vals)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(run_idx, string_vals, marker="o", linewidth=1.8, label="String Match (ms)")
    ax.plot(run_idx, embedding_vals, marker="o", linewidth=1.8, label="Embedding Match (ms)")

    string_lower = np.full_like(run_idx, string_mean - string_ci, dtype=float)
    string_upper = np.full_like(run_idx, string_mean + string_ci, dtype=float)
    embedding_lower = np.full_like(run_idx, embedding_mean - embedding_ci, dtype=float)
    embedding_upper = np.full_like(run_idx, embedding_mean + embedding_ci, dtype=float)

    ax.fill_between(run_idx, string_lower, string_upper, alpha=0.18, label="String mean ±95% CI")
    ax.fill_between(run_idx, embedding_lower, embedding_upper, alpha=0.18, label="Embedding mean ±95% CI")

    ax.axhline(string_mean, linestyle="--", linewidth=1.2)
    ax.axhline(embedding_mean, linestyle="--", linewidth=1.2)

    query = payload.get("config", {}).get("query", "")
    title_query = query[:80] + ("..." if len(query) > 80 else "")
    ax.set_title(f"Post-Filter Time: String vs Embedding\n{title_query}")
    ax.set_xlabel("Run Index")
    ax.set_ylabel("Post-Filter Time (ms)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_line_mean_ci.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
