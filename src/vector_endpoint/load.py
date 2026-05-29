#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

from vector_endpoint.catalog import Catalog, parse_nt_triple_line
from vector_endpoint.db.VectorDataBase import VectorDataBase


def chunked(iterable: Iterable[str], chunk_size: int) -> Iterator[List[str]]:
    bucket: List[str] = []
    for item in iterable:
        bucket.append(item)
        if len(bucket) >= chunk_size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def iter_nt_lines(file_path: Path, max_lines: Optional[int] = None) -> Iterator[str]:
    yielded = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield line
            yielded += 1
            if max_lines is not None and yielded >= max_lines:
                return


def build_schema(embedding_dimension: int) -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dimension),
    ]
    return CollectionSchema(fields, "Collection for text embeddings with concatenated vector field")


def ensure_collection(vdb: VectorDataBase, collection_name: str, log: bool = False) -> Collection:
    schema = build_schema(vdb.embedding_dimension)
    collection = vdb.add_collection(collection_name, schema, log=log)
    if collection is None:
        raise RuntimeError(f"Unable to create or retrieve collection '{collection_name}'")
    return collection


def process_chunk_with_milvus(
    vdb: VectorDataBase,
    collection: Collection,
    chunk_lines: List[str],
    catalog: Catalog,
    log: bool = False,
) -> Tuple[int, int]:
    normalized_chunk = []
    parsed_for_catalog = []

    for raw_line in chunk_lines:
        parsed = vdb._parse_triple_line(raw_line)
        normalized_chunk.append(vdb._normalize_triple_record(parsed, raw_line))
        if parsed:
            parsed_for_catalog.append(
                (parsed["subject"], parsed["predicate"], parsed["object"])
            )

    catalog_added = catalog.add_batch(parsed_for_catalog)
    chunk_texts = [record["text"] for record in normalized_chunk]
    embeddings = vdb._embed_triple_batch(normalized_chunk, normalize=True)
    entities = [chunk_texts, embeddings.tolist()]
    collection.insert(entities)

    if log:
        print(
            f"Inserted chunk: lines={len(chunk_lines)}, "
            f"catalog_triples={catalog_added}, embedding_shape={embeddings.shape}"
        )
    return len(chunk_lines), catalog_added


def process_chunk_catalog_only(chunk_lines: List[str], catalog: Catalog, log: bool = False) -> Tuple[int, int]:
    triples = []
    for line in chunk_lines:
        triple = parse_nt_triple_line(line)
        if triple:
            triples.append((triple.subject, triple.predicate, triple.object_value))
    catalog_added = catalog.add_batch(triples)
    if log:
        print(f"Catalog-only chunk: lines={len(chunk_lines)}, catalog_triples={catalog_added}")
    return len(chunk_lines), catalog_added


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream triples into Milvus while incrementally building a pickled cardinality catalog."
    )
    parser.add_argument("input_file", help="Path to NT data file")
    parser.add_argument("--catalog-out", default="catalog.pkl", help="Output pickle path")
    parser.add_argument("--collection", default="version_5", help="Milvus collection name")
    parser.add_argument("--database-name", default="lubm_db", help="Database label used by loader")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--target-embedding-dim", type=int, default=8, help="Per-component embedding dimension")
    parser.add_argument(
        "--dim-adjustment",
        default="truncate",
        choices=["truncate"],
        help="How to adapt model output to target dim (currently only truncate)",
    )
    parser.add_argument("--chunk-size", type=int, default=100, help="Ingestion chunk size")
    parser.add_argument("--max-lines", type=int, default=None, help="Optional limit for quick runs")
    parser.add_argument("--checkpoint-every-chunks", type=int, default=0, help="Save checkpoint every N chunks (0 disables)")
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint pickle path")
    parser.add_argument("--track-spo", action="store_true", help="Also track full (s,p,o) counts in catalog")
    parser.add_argument("--skip-index", action="store_true", help="Skip Milvus index creation")
    parser.add_argument("--skip-load", action="store_true", help="Skip loading collection into memory")
    parser.add_argument("--catalog-only", action="store_true", help="Build catalog only without Milvus writes")
    parser.add_argument(
        "--reset-all-collections",
        action="store_true",
        help="Drop all existing Milvus collections before creating the target collection",
    )
    parser.add_argument("--log", action="store_true", help="Verbose output")
    return parser.parse_args()


def reset_all_collections(vdb: VectorDataBase, log: bool = False) -> int:
    """Drop every existing collection and verify Milvus is empty."""
    existing_collections = list(utility.list_collections())
    if log:
        print(f"Reset requested. Found {len(existing_collections)} collection(s) to drop.")

    dropped_count = 0
    for collection_name in existing_collections:
        dropped = vdb.drop_collection(collection_name, log=log)
        if not dropped:
            raise RuntimeError(f"Failed to drop collection '{collection_name}' during reset")
        dropped_count += 1

    remaining_collections = list(utility.list_collections())
    if remaining_collections:
        raise RuntimeError(
            "Reset failed. Remaining collections: "
            + ", ".join(sorted(remaining_collections))
        )

    # Keep internal tracking aligned with Milvus after destructive reset.
    vdb._collections = set()
    if log:
        print("Reset completed. Milvus has no collections.")
    return dropped_count


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    catalog_out = Path(args.catalog_out)
    if catalog_out.parent and not catalog_out.parent.exists():
        catalog_out.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else catalog_out.with_suffix(".checkpoint.pkl")

    catalog = Catalog(track_spo=args.track_spo)
    total_lines_processed = 0
    total_catalog_triples = 0

    vdb: Optional[VectorDataBase] = None
    collection: Optional[Collection] = None
    chunk_count = 0

    if not args.catalog_only:
        vdb = VectorDataBase(
            database_name=args.database_name,
            host=args.host,
            port=args.port,
            embedding_model=args.embedding_model,
            target_embedding_dim=args.target_embedding_dim,
            dim_adjustment=args.dim_adjustment,
        )
        vdb.connect()
        if args.reset_all_collections:
            dropped_count = reset_all_collections(vdb, log=args.log)
            print(f"Reset complete. Dropped {dropped_count} collection(s).")
        collection = ensure_collection(vdb, args.collection, log=args.log)

    total_lines = sum(1 for _ in iter_nt_lines(input_path, max_lines=args.max_lines))
    total_chunks = math.ceil(total_lines / args.chunk_size) if total_lines else 0
    if args.log:
        print(
            f"Ingestion plan: total_lines={total_lines}, chunk_size={args.chunk_size}, "
            f"chunks={total_chunks}"
        )

    line_iter = iter_nt_lines(input_path, max_lines=args.max_lines)
    for chunk in chunked(line_iter, args.chunk_size):
        chunk_count += 1
        if args.catalog_only:
            lines_processed, catalog_added = process_chunk_catalog_only(chunk, catalog, log=args.log)
        else:
            lines_processed, catalog_added = process_chunk_with_milvus(vdb, collection, chunk, catalog, log=args.log)

        total_lines_processed += lines_processed
        total_catalog_triples += catalog_added
        if args.log:
            print(
                f"Chunk {chunk_count}/{total_chunks}: "
                f"lines={lines_processed}, catalog_triples={catalog_added}"
            )

        if args.checkpoint_every_chunks > 0 and chunk_count % args.checkpoint_every_chunks == 0:
            catalog.save_pickle(checkpoint_path)
            if args.log:
                print(f"Checkpoint saved after chunk {chunk_count}: {checkpoint_path}")

    if not args.catalog_only and collection is not None and vdb is not None:
        collection.flush()

        if not args.skip_index:
            vdb.create_index(
                collection_name=args.collection,
                index_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                M=16,
                ef_construction=200,
                log=args.log,
            )

        if not args.skip_load:
            vdb.load_data(args.collection, log=args.log)

    catalog.save_pickle(catalog_out)
    stats = catalog.summary()

    print("Load completed.")
    print(f"Input file: {input_path}")
    print(f"Lines processed: {total_lines_processed}")
    print(f"Catalog triples indexed: {total_catalog_triples}")
    print(f"Chunks processed: {chunk_count}")
    print(f"Catalog pickle: {catalog_out}")
    print("Catalog stats:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    if args.checkpoint_every_chunks > 0:
        print(f"Checkpoint path: {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

