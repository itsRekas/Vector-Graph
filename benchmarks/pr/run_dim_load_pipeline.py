#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import List, Sequence

from pymilvus import utility

from vector_endpoint.db.VectorDataBase import VectorDataBase


@dataclass(frozen=True)
class LoadResult:
    dimension: int
    collection: str
    dropped_existing: bool
    duration_seconds: float
    status: str
    return_code: int
    catalog_out: str
    error: str


def parse_dims(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        dim = int(token)
        if dim <= 0:
            raise ValueError(f"Invalid dimension: {dim}")
        values.append(dim)
    if not values:
        raise ValueError("No dimensions provided")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load benchmark data for one or more embedding dims into only the "
            "'dim_benchmark' collection. Existing 'dim_benchmark' is dropped first."
        )
    )
    parser.add_argument("--input-file", required=True, help="Path to .nt file")
    parser.add_argument(
        "--collection",
        default="dim_benchmark",
        help="Benchmark collection name. Must be 'dim_benchmark'.",
    )
    parser.add_argument(
        "--dimensions",
        default="8",
        help="Comma-separated dims to load",
    )
    parser.add_argument("--database-name", default="lubm_db", help="Database label")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument(
        "--dim-adjustment",
        default="truncate",
        choices=["truncate"],
        help="How to adapt model output to target dim (currently only truncate).",
    )
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size")
    parser.add_argument("--max-lines", type=int, default=None, help="Optional quick-run cap")
    parser.add_argument("--out-dir", default="results", help="Output metadata directory")
    parser.add_argument("--log", action="store_true", help="Verbose logs")
    return parser.parse_args()


def require_safe_collection(collection: str) -> None:
    if collection != "dim_benchmark":
        raise ValueError(
            "Safety guard: this script only allows --collection dim_benchmark."
        )


def drop_dim_benchmark_if_exists(
    *,
    collection: str,
    database_name: str,
    host: str,
    port: int,
    embedding_model: str,
    dim_adjustment: str,
    dim: int,
    log: bool,
) -> bool:
    vdb = VectorDataBase(
        database_name=database_name,
        host=host,
        port=port,
        embedding_model=embedding_model,
        target_embedding_dim=dim,
        dim_adjustment=dim_adjustment,
    )
    vdb.connect()
    if collection not in set(utility.list_collections()):
        return False
    ok = vdb.drop_collection(collection, log=log)
    if not ok:
        raise RuntimeError(f"Failed to drop existing collection: {collection}")
    return True


def run_loader_for_dim(dim: int, args: argparse.Namespace, out_dir: Path) -> LoadResult:
    collection = args.collection
    dropped_existing = False
    status = "failed"
    return_code = 1
    error = ""
    start = perf_counter()
    catalog_out = out_dir / f"catalog_dim{dim}.pkl"
    try:
        dropped_existing = drop_dim_benchmark_if_exists(
            collection=collection,
            database_name=args.database_name,
            host=args.host,
            port=args.port,
            embedding_model=args.embedding_model,
            dim_adjustment=args.dim_adjustment,
            dim=dim,
            log=args.log,
        )

        cmd: List[str] = [
            sys.executable,
            "-m",
            "vector_endpoint.load",
            args.input_file,
            "--collection",
            collection,
            "--database-name",
            args.database_name,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--embedding-model",
            args.embedding_model,
            "--target-embedding-dim",
            str(dim),
            "--dim-adjustment",
            args.dim_adjustment,
            "--chunk-size",
            str(args.chunk_size),
            "--catalog-out",
            str(catalog_out),
        ]
        if args.max_lines is not None:
            cmd.extend(["--max-lines", str(args.max_lines)])
        if args.log:
            cmd.append("--log")

        proc = subprocess.run(cmd, capture_output=not args.log, text=True)
        return_code = int(proc.returncode)
        if return_code == 0:
            status = "ok"
        else:
            error = (proc.stderr or proc.stdout or "").strip()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    elapsed = perf_counter() - start
    return LoadResult(
        dimension=dim,
        collection=collection,
        dropped_existing=dropped_existing,
        duration_seconds=elapsed,
        status=status,
        return_code=return_code,
        catalog_out=str(catalog_out),
        error=error,
    )


def write_outputs(rows: Sequence[LoadResult], out_dir: Path, collection: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir / f"dim_load_pipeline_{ts}"
    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")

    payload = {
        "timestamp_utc": ts,
        "collection": collection,
        "results": [asdict(r) for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dimension",
                "collection",
                "dropped_existing",
                "duration_seconds",
                "status",
                "return_code",
                "catalog_out",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"Load metadata JSON: {json_path}")
    print(f"Load metadata CSV:  {csv_path}")
    return json_path


def main() -> int:
    args = parse_args()
    require_safe_collection(args.collection)
    dims = parse_dims(args.dimensions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collection: {args.collection}")
    print(f"Dimensions: {dims}")
    print(f"Input file: {args.input_file}")

    rows: List[LoadResult] = []
    for dim in dims:
        print(f"\n=== Load dim={dim} into {args.collection} ===")
        row = run_loader_for_dim(dim, args, out_dir)
        rows.append(row)
        print(
            f"status={row.status} | dropped_existing={row.dropped_existing} "
            f"| duration={row.duration_seconds:.2f}s"
        )
        if row.error:
            print(f"error={row.error}")

    write_outputs(rows, out_dir, args.collection)
    failures = [r for r in rows if r.status != "ok"]
    if failures:
        print(f"\nCompleted with failures: {len(failures)}/{len(rows)} dims.")
        return 1
    print("\nLoad pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
