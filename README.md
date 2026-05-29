# Vector Endpoint

Milvus-backed vector query endpoint that answers Comunica triple-pattern requests
using per-component S|P|O embeddings (`all-MiniLM-L6-v2`, dimension truncated at
load). The active Milvus collection is `version_5`.

## Layout

```
vector-endpoint/
├── pyproject.toml              # package definition + dependencies
├── src/
│   └── vector_endpoint/        # the engine (installable package)
│       ├── app.py              # Flask endpoint (POST /vector on :2222)
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

1. Start the Milvus stack:

```bash
docker compose up -d
```

2. Start the endpoint (listens on `http://localhost:2222`):

```bash
.venv/bin/python -m vector_endpoint.app
```

Optional environment overrides:

- `VECTOR_CATALOG_PATH` — path to the catalog pickle used for auto-k
  (defaults to `catalog.pkl` at the repo root).

## Comunica integration

Queries reach this endpoint through a forked Comunica engine:

- **Fork:** [`itsRekas/comunica`](https://github.com/itsRekas/comunica) — a
  Comunica 5.2.2 fork with the Colab research changes.

It adds a vector query source (`actor-query-source-identify-hypermedia-vector`)
exposed via the **`comunica-vector`** CLI, which sends `POST` requests to
`http://localhost:2222/vector` and forwards the `-k` search limit to Milvus. The
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

With the endpoint running and Milvus collection `version_5` loaded, a vector
query looks like:

```bash
comunica-vector http://localhost:2222/vector -k 1200 -q \
  'SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> }'
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
