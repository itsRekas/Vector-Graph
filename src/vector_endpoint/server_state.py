"""Shared Milvus + catalog initialization for HTTP and gRPC entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

from vector_endpoint.auto_k import CatalogKResolver
from vector_endpoint.db.VectorDataBase import VectorDataBase

VECTOR_COLLECTION_NAME = "version_5"
CATALOG_PATH = Path(
    os.getenv("VECTOR_CATALOG_PATH", Path(__file__).resolve().parents[2] / "catalog.pkl")
)

AUTO_K_RESOLVER = CatalogKResolver(catalog_path=CATALOG_PATH)
if AUTO_K_RESOLVER.available:
    print(f"Catalog auto-k enabled: {CATALOG_PATH}")
else:
    print(f"Catalog auto-k disabled: {AUTO_K_RESOLVER.error}")

vdb: VectorDataBase | None = VectorDataBase(
    database_name="lubm_db",
    host=os.getenv("VECTOR_MILVUS_HOST", "localhost"),
    port=int(os.getenv("VECTOR_MILVUS_PORT", "19530")),
    embedding_model=os.getenv("VECTOR_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    target_embedding_dim=int(os.getenv("VECTOR_TARGET_EMBEDDING_DIM", "384")),
)

try:
    vdb.connect()
    print("Connected to Vector Database")
except Exception as e:  # noqa: BLE001
    print(f"Failed to connect to Vector Database: {e}")
    vdb = None
