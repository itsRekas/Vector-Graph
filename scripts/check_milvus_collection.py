#!/usr/bin/env python3
"""Print Milvus collection stats for the running vector-endpoint setup."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from vector-endpoint/src
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pymilvus import Collection, connections, utility  # noqa: E402

COLLECTION = "version_5"
HOST = "localhost"
PORT = 19530


def main() -> int:
    print(f"Connecting to Milvus at {HOST}:{PORT}...")
    try:
        connections.connect(host=HOST, port=PORT)
    except Exception as exc:
        print(f"FAILED to connect: {exc}")
        print("Is Milvus running? (e.g. docker compose up)")
        return 1

    names = sorted(utility.list_collections())
    print(f"\nCollections ({len(names)}): {', '.join(names) or '(none)'}")

    if COLLECTION not in names:
        print(f"\nWARNING: '{COLLECTION}' not found.")
        print("Load RLUBM with:")
        print(
            "  python src/load.py data/nts/RLUBM_cleaned.nt "
            "--collection version_5 --target-embedding-dim 8 --catalog-out catalog.pkl --log"
        )
        return 2

    col = Collection(COLLECTION)
    col.load()
    n = col.num_entities
    emb_field = next((f for f in col.schema.fields if f.name == "embedding"), None)
    dim = emb_field.params.get("dim") if emb_field else "?"

    print(f"\nCollection: {COLLECTION}")
    print(f"  Entities: {n:,}")
    print(f"  Embedding dim: {dim} (expect 24 for target_embedding_dim=8)")

    if n == 0:
        print("\nSTATUS: EMPTY — vector queries will return 0 rows.")
        print("Re-run load.py on RLUBM_cleaned.nt, then restart vectorEndpoint.")
        return 3

    if dim != 24:
        print(f"\nSTATUS: WRONG DIM — endpoint uses target_embedding_dim=8 → dim 24, got {dim}.")
        return 4

    print("\nSTATUS: OK — collection has data and expected dim=24.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
