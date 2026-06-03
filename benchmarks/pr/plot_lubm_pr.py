#!/usr/bin/env python3
"""Plot latest LUBM vector precision/recall per query (vs SPARQL-file GT)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing lubm_pr_<timestamp>.json files",
    )
    p.add_argument(
        "--glob",
        default="lubm_pr_*.json",
        help="JSON glob under results-dir",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <results-dir>/lubm_pr_latest_pr.png)",
    )
    p.add_argument(
        "--threshold-pct",
        type=float,
        default=95.0,
        help="Horizontal threshold line (percent)",
    )
    p.add_argument(
        "--title",
        default="LUBM Q1-Q14: vector precision & recall",
    )
    return p.parse_args()


def load_latest_per_query(results_dir: Path, pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob(pattern)):
        if "_summary" in path.name or "_per_query" in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        ts = payload.get("timestamp_utc") or path.stem.replace("lubm_pr_", "")
        for q in payload.get("queries", []):
            rows.append(
                {
                    "ts": ts,
                    "query_num": int(q["query_num"]),
                    "query_id": q["query_id"],
                    "query_type": q.get("query_type", ""),
                    "vector_count": q.get("vector_count"),
                    "precision": q.get("precision"),
                    "recall": q.get("recall"),
                    "failed": q.get("error") is not None,
                    "source": path.name,
                }
            )

    latest: dict[str, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda x: x["ts"], reverse=True):
        latest.setdefault(r["query_id"], r)

    out = [latest[f"LUBM_Q{i}"] for i in range(1, 15) if f"LUBM_Q{i}" in latest]
    if not out:
        raise RuntimeError(f"No LUBM query rows found under {results_dir}/{pattern}")
    return sorted(out, key=lambda x: x["query_num"])


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    rows = load_latest_per_query(results_dir, args.glob)
    threshold = args.threshold_pct / 100.0

    labels = [f"Q{r['query_num']}" for r in rows]
    n = len(rows)
    x = np.arange(n)
    width = 0.35
    offsets = np.array([-0.5, 0.5]) * width

    vec_p, vec_r = [], []
    for r in rows:
        if r["failed"]:
            vec_p.append(0.0)
            vec_r.append(0.0)
        else:
            vec_p.append(float(r["precision"] or 0.0))
            vec_r.append(float(r["recall"] or 0.0))

    fig, ax = plt.subplots(figsize=(14, 6))
    series = [
        (offsets[0], vec_p, "#4c78a8", "Vector precision"),
        (offsets[1], vec_r, "#54a24b", "Vector recall"),
    ]
    for off, vals, color, label in series:
        bars = ax.bar(x + off, vals, width, label=label, color=color, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if val <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(val + 0.02, 1.08),
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )

    for i, r in enumerate(rows):
        if r["failed"]:
            ax.text(
                x[i],
                0.03,
                "FAIL",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#c0392b",
            )

    ax.axhline(
        threshold,
        color="#e45756",
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold {args.threshold_pct:.0f}%",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.15)
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    out = args.output or (results_dir / "lubm_pr_latest_pr.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print("Latest sources:")
    for r in rows:
        if r["failed"]:
            status = "FAIL"
        else:
            status = f"P={r['precision']:.0%} R={r['recall']:.0%}"
        print(f"  {r['query_id']:10s} {r['ts']}  {status}  ({r['source']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
