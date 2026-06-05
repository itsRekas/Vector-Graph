# Vector Endpoint

Milvus-backed vector query endpoint that answers Comunica triple-pattern requests
using per-component S|P|O embeddings (`all-MiniLM-L6-v2`, default per-component
dim **384**, stored as **1152-d** concatenated vectors). The active Milvus
collection is `version_5`.

## Layout

```
vector-endpoint/
├── pyproject.toml              # package definition + dependencies
├── src/
│   └── vector_endpoint/        # the engine (installable package)
│       ├── app.py              # Flask HTTP baseline (POST /vector on :2222)
│       ├── grpc_app.py         # gRPC streaming server (:50051)
│       ├── pattern_query.py    # shared BGP logic (stream + collect)
│       ├── proto/              # (repo root) pattern.proto
│       ├── catalog.py          # cardinality catalog for auto-k
│       ├── auto_k.py           # catalog-driven k resolution
│       ├── load.py             # NT -> Milvus load pipeline
│       ├── clean.py            # NT de-duplication helper
│       ├── adaptive_exp/       # adaptive k-escalation search
│       └── db/VectorDataBase.py
├── benchmarks/
│   ├── pr/                     # precision/recall vs SPARQL ground truth
│   └── string_match/           # string vs embedding part-match timing
├── data/                       # RDF datasets (RLUBM_cleaned.nt, ...)
├── scripts/                    # operational helpers (Milvus checks)
├── volumes/                    # Milvus runtime data (gitignored)
├── catalog.pkl                 # prebuilt catalog (gitignored)
├── configs/milvus.yaml           # Milvus standalone config (mounted in Docker)
├── docker-compose.yml          # Milvus stack (etcd + minio + standalone)
└── archive/                    # frozen legacy code (not part of the pipeline)
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

This installs the `vector_endpoint` package in editable mode, so every script and
benchmark imports it as `from vector_endpoint... import ...` with no `sys.path`
manipulation, regardless of the directory it runs from.

## Run

1. Start the Milvus stack (uses `configs/milvus.yaml` via `docker-compose.yml`):

```bash
docker compose up -d
```

After changing `configs/milvus.yaml`, recreate the Milvus container so the mount
is picked up (`docker compose up -d --force-recreate standalone`).

2. Start an endpoint (two independent processes):

**HTTP baseline** (buffered JSON, `http://localhost:2222/vector`):

```bash
.venv/bin/python -m vector_endpoint.app
```

**gRPC streaming** (`grpc://127.0.0.1:50051`):

```bash
.venv/bin/python -m vector_endpoint.grpc_app
```

Regenerate Python stubs after editing `proto/vector/v1/pattern.proto`:

```bash
bash scripts/generate_grpc.sh
```

Optional environment overrides:

- `VECTOR_CATALOG_PATH` — path to the catalog pickle used for auto-k
  (defaults to `catalog.pkl` at the repo root).
- `VECTOR_JOIN_BOUND_MIN_K` — minimum k for join-extension POSTs with a bound
  subject or object (default `512`).
- `VECTOR_ADAPTIVE_MULTIPLIERS` — comma-separated k ladder (default `1,10,100,1000`).
- `VECTOR_ADAPTIVE_JACCARD` — Jaccard stability threshold (default `0.99`).
- `VECTOR_GRPC_PORT` — gRPC listen port (default `50051`).
- `VECTOR_GRPC_ROW_BATCH` — rows per `row_batch` stream event (default `100`).

## Milvus configuration

`docker-compose.yml` mounts `configs/milvus.yaml` into the standalone service
(`./configs/milvus.yaml:/milvus/configs/milvus.yaml`). The stock Milvus defaults
cap search `topk` too low for LUBM join workloads and adaptive k escalation; this
repo raises the limits that matter for production:

| Setting | Location | Value | Why |
|---------|----------|-------|-----|
| `common.topKLimit` | `configs/milvus.yaml` | `200000` | Matches adaptive k / catalog ladders (see `milvus_safe_k` in `auto_k.py`). |
| `queryNode.grpc.*Max*Size` | `configs/milvus.yaml` | up to 512 MiB send / 256 MiB recv | Large top-k batches over gRPC. |
| `MQ_TYPE` | `docker-compose.yml` | `woodpecker` | Embedded message queue for standalone (no external Pulsar). |

Reload RLUBM after a dim change so `version_5` matches `target_embedding_dim` in
`app.py` (384 → 1152 stored dim):

```bash
.venv/bin/python -m vector_endpoint.load data/nts/RLUBM_cleaned.nt \
  --collection version_5 --target-embedding-dim 384 --catalog-out catalog.pkl --log
```

Check the running collection with `python scripts/check_milvus_collection.py`.

## Comunica integration

Queries reach this endpoint through a forked Comunica engine:

- **Fork:** [`itsRekas/comunica`](https://github.com/itsRekas/comunica) — a
  Comunica 5.2.2 fork with the Colab research changes.

It adds a vector query source (`actor-query-source-identify-hypermedia-vector`)
exposed via the **`comunica-vector`** CLI. Select the pipeline with **`--http`**
(baseline JSON) or **`--grpc`** (streaming):

```bash
# HTTP baseline
comunica-vector --http http://localhost:2222/vector -k 1200 -q 'SELECT ...'

# gRPC streaming (start vector_endpoint.grpc_app first)
comunica-vector --grpc 127.0.0.1:50051 -k 1200 -q 'SELECT ...'
```

Legacy invocations without flags still default to HTTP when the source URL is
`http://...`. The CLI forwards `-k` and adaptive options to Milvus. The
fork also provides **`comunica-sparql-file`**, used as the SPARQL ground-truth
baseline in `benchmarks/pr/`.

Build and link the CLIs (`comunica-vector`, `comunica-sparql-file`) before
running the endpoint queries or the benchmarks:

```bash
git clone https://github.com/itsRekas/comunica.git
cd comunica
yarn install && yarn build && yarn run build:engines
cd engines/query-sparql && yarn link   # puts comunica-vector / comunica-sparql-file on PATH
```

See the fork's `SWITCH.md` for full build/link details and the Milvus cutover steps.

## Benchmarks

Each benchmark folder has its own README:

- `benchmarks/pr/` — embedding-dimension precision/recall against SPARQL ground
  truth (via `comunica-sparql-file`).
- `benchmarks/string_match/` — post-filter comparison time, string match vs
  embedding part-match.

## Catalog compatibility

`catalog.pkl` files written before the engine was packaged referenced the
top-level module path `catalog`. `Catalog.from_bytes` remaps that legacy path to
`vector_endpoint.catalog` on load, so older pickles keep working; new pickles are
written under the current path.
