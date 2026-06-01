#!/usr/bin/env python3
"""Print Milvus API storage stamp and optional MinIO on-disk usage."""

from __future__ import annotations

import argparse
from pathlib import Path

from vector_endpoint.db.VectorDataBase import VectorDataBase

DEFAULT_COLLECTION = "version_5"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 19530
MINIO_VOLUME = Path(__file__).resolve().parents[1] / "volumes" / "minio"


def dir_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def human_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{n} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Milvus collection stats (API) and MinIO volume size (disk)."
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Milvus collection name")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Milvus host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Milvus port")
    parser.add_argument("--target-embedding-dim", type=int, default=8, help="Per-component embedding dim (fallback if schema missing)")
    parser.add_argument(
        "--minio-path",
        default=str(MINIO_VOLUME),
        help="Path to Milvus MinIO data directory (docker compose volumes/minio)",
    )
    parser.add_argument("--skip-disk", action="store_true", help="Skip MinIO directory size scan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    default_dim = args.target_embedding_dim * 3
    stamp = VectorDataBase.milvus_storage_stamp(
        args.collection,
        host=args.host,
        port=args.port,
        default_embedding_dim=default_dim,
    )
    print("Milvus API storage stamp:")
    print(f"  {stamp}")

    if not args.skip_disk:
        minio_path = Path(args.minio_path)
        disk_bytes = dir_size_bytes(minio_path)
        print(f"\nMinIO volume ({minio_path}):")
        print(f"  total_on_disk={human_bytes(disk_bytes)} ({disk_bytes} bytes)")
        print(
            "  note: includes all collections and indexes in this MinIO bucket, "
            "not just the named collection"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
