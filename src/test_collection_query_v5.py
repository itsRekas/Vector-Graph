#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, Dict, List

from pymilvus import utility

from vector_endpoint.db.VectorDataBase import VectorDataBase


# Simple query style used in V4 testing/benchmark flow.
DEFAULT_QUERY = (
    "SELECT ?X WHERE {"
    " ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
    " <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> }"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a Milvus collection and run one simple V4-style vector query."
    )
    parser.add_argument("--collection", default="version_5", help="Collection to validate/search")
    parser.add_argument("--database-name", default="lubm_db", help="Database label for VectorDataBase")
    parser.add_argument("--host", default="localhost", help="Milvus host")
    parser.add_argument("--port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--target-embedding-dim", type=int, default=384, help="Per-component model dim")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Simple SPARQL query to test")
    parser.add_argument("--k", type=int, default=10, help="Top-K results to fetch")
    parser.add_argument("--log", action="store_true", help="Verbose VectorDataBase logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    vdb = VectorDataBase(
        database_name=args.database_name,
        host=args.host,
        port=args.port,
        embedding_model=args.embedding_model,
        target_embedding_dim=args.target_embedding_dim,
    )
    vdb.connect()

    collections = sorted(utility.list_collections())
    print(f"Milvus collections ({len(collections)}): {collections}")

    if args.collection not in collections:
        print(f"ERROR: Collection '{args.collection}' does not exist.")
        return 1

    collection_obj = vdb.get_collection(args.collection, log=args.log)
    if collection_obj is None:
        print(f"ERROR: Could not retrieve collection '{args.collection}'.")
        return 1

    entity_count = collection_obj.num_entities
    print(f"Collection '{args.collection}' entity count: {entity_count}")
    if entity_count == 0:
        print("WARNING: Collection exists but has no entities.")

    vdb.load_data(args.collection, log=args.log)

    print("\nRunning simple query:")
    print(args.query)
    results = vdb.search(
        collection_name=args.collection,
        query_texts=args.query,
        limit=args.k,
        output_fields=["text"],
        log=args.log,
    )

    if not results:
        print("No search result sets returned.")
        return 0

    first: Dict[str, Any] = results[0]
    matches: List[Dict[str, Any]] = first.get("matches", [])
    print(f"\nMatches returned: {len(matches)}")

    if not matches:
        print("No matches found.")
        return 0

    for idx, match in enumerate(matches[: args.k], start=1):
        print(
            f"{idx}. score={match.get('score'):.6f} | "
            f"text={match.get('text')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
