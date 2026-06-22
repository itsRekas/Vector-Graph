#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

from vector_endpoint.catalog import Catalog, parse_nt_triple_line
from vector_endpoint.db.VectorDataBase import VectorDataBase
from vector_endpoint.embedding_disk_cache import (
    DiskEmbeddingCache,
    build_cache_from_nt,
    build_meta,
    collect_component_texts_from_triple,
    embed_triple_batch,
    load_cache,
    save_cache,
    validate_meta,
)


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


def _accumulate_cache_from_triples(
    triples: List[dict],
    vdb: VectorDataBase,
    out_cache: DiskEmbeddingCache,
    log: bool = False,
) -> None:
    """Add newly seen component texts to out_cache using raw (unnormalized) encodings."""
    texts_to_encode: List[str] = []
    for triple in triples:
        for text in collect_component_texts_from_triple(triple):
            if text is not None and out_cache.lookup_raw(text) is None and text not in texts_to_encode:
                texts_to_encode.append(text)

    if not texts_to_encode:
        return

    raw_embeddings = vdb._encode_text_batch(texts_to_encode, normalize=False)
    for text, vec in zip(texts_to_encode, raw_embeddings):
        out_cache.put(text, vec)

    if log:
        print(f"Embedding cache: added {len(texts_to_encode)} components (total={len(out_cache)})")


def process_chunk_with_milvus(
    vdb: VectorDataBase,
    collection: Collection,
    chunk_lines: List[str],
    catalog: Catalog,
    *,
    disk_cache: Optional[DiskEmbeddingCache] = None,
    out_cache: Optional[DiskEmbeddingCache] = None,
    skip_catalog_updates: bool = False,
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

    catalog_added = 0
    if not skip_catalog_updates:
        catalog_added = catalog.add_batch(parsed_for_catalog)

    if out_cache is not None:
        _accumulate_cache_from_triples(normalized_chunk, vdb, out_cache, log=log)

    chunk_texts = [record["text"] for record in normalized_chunk]
    if disk_cache is not None:
        embeddings = embed_triple_batch(
            normalized_chunk,
            disk_cache,
            target_dim=vdb._embedding_dim,
            normalize=True,
            fusion=vdb._component_fusion,
        )
    else:
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
    parser.add_argument("--catalog-in", default=None, help="Existing catalog pickle (skip catalog rebuild)")
    parser.add_argument("--collection", default="version_5", help="Milvus collection name")
    parser.add_argument("--database-name", default="lubm_db", help="Database label used by loader")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    # parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--embedding-model", default="mixedbread-ai/mxbai-embed-xsmall-v1", help="SentenceTransformer model name")
    parser.add_argument("--target-embedding-dim", type=int, default=384, help="Per-component embedding dimension")
    parser.add_argument(
        "--dim-adjustment",
        default="truncate",
        choices=["truncate"],
        help="How to adapt model output to target dim (currently only truncate)",
    )
    parser.add_argument(
        "--component-fusion",
        default="concat",
        choices=["concat", "hadamard"],
        help="Fuse S|P|O embeddings: concat (3d stored) or hadamard (d stored)",
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
        "--embed-cache-only",
        action="store_true",
        help="Build embedding disk cache and catalog only (no Milvus writes)",
    )
    parser.add_argument(
        "--embedding-cache-out",
        default=None,
        help="Write raw component embeddings to this .npz path (requires full model dim)",
    )
    parser.add_argument(
        "--embedding-cache-in",
        default=None,
        help="Load raw component embeddings from .npz instead of calling the model",
    )
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

    vdb._collections = set()
    if log:
        print("Reset completed. Milvus has no collections.")
    return dropped_count


def _resolve_catalog(args: argparse.Namespace, catalog_out: Path) -> Tuple[Catalog, bool]:
    if args.catalog_in:
        catalog_path = Path(args.catalog_in)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog input not found: {catalog_path}")
        catalog = Catalog.load_pickle(catalog_path)
        if args.log:
            print(f"Loaded catalog from {catalog_path}")
        return catalog, True

    return Catalog(track_spo=args.track_spo), False


def _run_embed_cache_only(args: argparse.Namespace, input_path: Path, catalog_out: Path) -> int:
    cache_out = Path(args.embedding_cache_out) if args.embedding_cache_out else None
    if cache_out is None:
        raise ValueError("--embed-cache-only requires --embedding-cache-out")

    vdb = VectorDataBase(
        database_name=args.database_name,
        host=args.host,
        port=args.port,
        embedding_model=args.embedding_model,
        target_embedding_dim=args.target_embedding_dim,
        dim_adjustment=args.dim_adjustment,
        component_fusion=args.component_fusion,
    )

    if args.target_embedding_dim != vdb._model_output_dim:
        raise ValueError(
            f"--embed-cache-only requires --target-embedding-dim={vdb._model_output_dim} "
            f"(model output dim), got {args.target_embedding_dim}"
        )

    if args.log:
        print("Building embedding cache and catalog (no Milvus)...")

    cache, catalog, meta = build_cache_from_nt(
        input_path,
        vdb,
        max_lines=args.max_lines,
        embedding_model=args.embedding_model,
        dim_adjustment=args.dim_adjustment,
    )
    save_cache(cache_out, cache, meta)
    catalog.save_pickle(catalog_out)

    print("Embed-cache-only completed.")
    print(f"Embedding cache: {cache_out}")
    print(f"Catalog pickle: {catalog_out}")
    print(f"Unique components: {len(cache)}")
    return 0


def main() -> int:
    args = parse_args()

    if args.embed_cache_only:
        args.catalog_only = True

    if args.embedding_cache_in and args.embedding_cache_out:
        raise ValueError("Use only one of --embedding-cache-in or --embedding-cache-out")

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    catalog_out = Path(args.catalog_out)
    if catalog_out.parent and not catalog_out.parent.exists():
        catalog_out.parent.mkdir(parents=True, exist_ok=True)

    if args.embed_cache_only:
        return _run_embed_cache_only(args, input_path, catalog_out)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else catalog_out.with_suffix(".checkpoint.pkl")

    catalog, skip_catalog_updates = _resolve_catalog(args, catalog_out)
    total_lines_processed = 0
    total_catalog_triples = 0

    disk_cache: Optional[DiskEmbeddingCache] = None
    out_cache: Optional[DiskEmbeddingCache] = None

    if args.embedding_cache_in:
        disk_cache, cache_meta = load_cache(Path(args.embedding_cache_in))
        validate_meta(
            cache_meta,
            nt_path=input_path,
            embedding_model=args.embedding_model,
            dim_adjustment=args.dim_adjustment,
        )
        if args.log:
            print(
                f"Loaded embedding cache: {args.embedding_cache_in} "
                f"({len(disk_cache)} components, full_dim={disk_cache.full_dim})"
            )

    if args.embedding_cache_out:
        out_cache = DiskEmbeddingCache.from_dict({})

    vdb: Optional[VectorDataBase] = None
    collection: Optional[Collection] = None
    chunk_count = 0

    if not args.catalog_only:
        use_lazy_model = bool(args.embedding_cache_in)
        vdb = VectorDataBase(
            database_name=args.database_name,
            host=args.host,
            port=args.port,
            embedding_model=args.embedding_model,
            target_embedding_dim=args.target_embedding_dim,
            dim_adjustment=args.dim_adjustment,
            lazy_embedding_model=use_lazy_model,
            component_fusion=args.component_fusion,
        )
        vdb.connect()
        if args.reset_all_collections:
            dropped_count = reset_all_collections(vdb, log=args.log)
            print(f"Reset complete. Dropped {dropped_count} collection(s).")

        if args.embedding_cache_out:
            vdb._ensure_embedding_model()
            if args.target_embedding_dim != vdb._model_output_dim:
                raise ValueError(
                    f"--embedding-cache-out requires --target-embedding-dim={vdb._model_output_dim}, "
                    f"got {args.target_embedding_dim}"
                )

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
            lines_processed, catalog_added = process_chunk_with_milvus(
                vdb,
                collection,
                chunk,
                catalog,
                disk_cache=disk_cache,
                out_cache=out_cache,
                skip_catalog_updates=skip_catalog_updates,
                log=args.log,
            )

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

    if out_cache is not None and args.embedding_cache_out:
        meta = build_meta(
            nt_path=input_path,
            embedding_model=args.embedding_model,
            dim_adjustment=args.dim_adjustment,
            cache_full_dim=out_cache.full_dim,
            unique_components=len(out_cache),
        )
        save_cache(Path(args.embedding_cache_out), out_cache, meta)
        if args.log:
            print(f"Saved embedding cache: {args.embedding_cache_out} ({len(out_cache)} components)")

    if args.catalog_in and catalog_out != Path(args.catalog_in):
        shutil.copy2(args.catalog_in, catalog_out)
    else:
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
