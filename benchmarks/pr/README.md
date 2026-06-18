# Precision/Recall Benchmark (`benchmarks/pr`)

This benchmark evaluates embedding dimensions using **precision** and **recall** against SPARQL ground truth, then compares dimensions on:

- `X`: embedding dimension
- `Y`: average precision / average recall (0-1)

It is split into three phases:

1. Query generation (`generate_random_queries.py`)
2. Load + benchmark (`run_vector_dim_accuracy_benchmark.py`, optionally using `run_dim_load_pipeline.py`)
3. Plotting (`plot_vector_dim_accuracy.py`)

Safety behavior:

- This benchmark only allows collection name **`dim_benchmark`**.
- If `dim_benchmark` already exists, it is dropped before each new dimension load.
- Other collections (including your main pipeline collection) are not targeted by these scripts.

## Metrics

The benchmark reports **two** precision/recall tracks per query.

### Post-filter P/R (primary)

Bindings returned after `string_part_match` post-filter (same as the endpoint).
Used for pass/fail threshold, F1@K plots, and dimension selection.

For each query and dimension:

- `TP`, `FP`, `FN`
- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `jaccard` and `exact_match` (secondary diagnostics)

Per dimension:

- `avg_precision` (0-1)
- `avg_recall` (0-1)
- bucket-level averages for `sp*`, `*po`, `s*o`

A dimension passes when both `avg_precision` and `avg_recall` are at least `--accuracy-threshold-pct / 100`.

### Raw retrieval P/R (diagnostic)

Bindings extracted from **all parseable** Milvus top-k hits, with **no**
pattern post-filter. Measures vector-neighbor quality before filtering.

- Per query: `raw_tp`, `raw_fp`, `raw_fn`, `raw_precision`, `raw_recall`,
  `raw_jaccard`, plus `raw_hit_count`, `raw_parseable_count`, `raw_binding_count`
- Per dimension: `avg_raw_precision`, `avg_raw_recall`, `mean_raw_jaccard`

Raw precision is typically lower than post-filter (non-matching neighbors become
FPs). Raw recall can be higher when GT triples are in top-k but post-filter
rejects them.

**gRPC runs:** raw hits are streamed from the server (`include_raw_hits=true`)
from the **same** Milvus search used for post-filter results (final adaptive
round k when escalating).

## Dimension configuration

- `--dimensions` is the **per-component** embedding dim: S, P and O are each
  truncated to this size and concatenated, so the stored vector is `3 × dim`.
- Production endpoint uses `384` (→ 1152-dim stored vectors) in
  `src/vector_endpoint/app.py`. Dim sweeps can still use smaller values
  (e.g. `8` → 24-dim) via `--dimensions` on `dim_benchmark` only.
- `384` is the model-native max for `all-MiniLM-L6-v2`; you can sweep any value
  `<= 384` (e.g. `--dimensions 8,16,32,64,128,256,384`).
- Dim adjustment mode is controlled by `--dim-adjustment` and defaults to `truncate`.
- In truncate mode:
  - `target_embedding_dim <= model_dim` works (e.g. 256 from a 384-d model)
  - `target_embedding_dim > model_dim` fails fast with a clear error

## Result limit (`--k`)

- `--k` defaults to **catalog auto-k**: when omitted, each pattern gets a
  per-query limit derived from the catalog match count (see `auto_k.py`). This is
  the recommended mode.
- The `--k 5000` shown in the commands below is just an illustrative fixed value,
  not a default. Drop the `--k` flag to use auto-k.

## 1) Generate query sets

From `vector-endpoint/benchmarks/pr`:

### Dim-sweep profile (recommended for PR dynamic sweep)

Cardinality-weighted ~800 queries tuned for dimension stress testing:

| Bucket | Count | Rule |
|--------|-------|------|
| `*po` | 500 | All **67** keys with count ≥ 100; remainder split evenly between count **10–24** and **25–99** |
| `sp*` | 250 | Count **2–7** only, evenly across exact counts |
| `s*o` | 50 | All **37** count=2 keys + fill from count=1 |

```bash
cd benchmarks/pr
../../.venv/bin/python generate_random_queries.py \
  --profile dim-sweep \
  --input-file ../../data/nts/RLUBM_cleaned.nt \
  --seed 42 \
  --sp-count 250 --po-count 500 --so-count 50
```

Writes:

- `results/PR_dynamic_sweep/random_queries_dim_sweep.json` (sweep default)
- `results/random_queries_dim_sweep.json` (copy)

Each query includes `expected_count` and `sampling_band` (for `*po` bands:
`large`, `mid_10_24`, `mid_25_99`).

### Legacy stratified profile (archived)

Equal 1000/1000/1000 with even spread across all result-count bins:

```bash
../../.venv/bin/python generate_random_queries.py \
  --profile legacy-stratified \
  --input-file ../../data/nts/RLUBM_cleaned.nt \
  --out results/archive/query_sets/random_queries_3000_stratified.json
```

Uniform random keys (no stratification):

```bash
../../.venv/bin/python generate_random_queries.py \
  --profile legacy-stratified \
  --no-stratify-counts \
  --out results/archive/query_sets/random_queries_3000.json
```

### Query file reference

| File | Purpose |
|------|---------|
| `results/PR_dynamic_sweep/random_queries_dim_sweep.json` | **Active** dim-sweep weighted set |
| `results/archive/query_sets/random_queries_3000_stratified.json` | Archived equal-bucket stratified set |
| `results/archive/query_sets/random_queries_3000.json` | Archived legacy uniform random set |

Benchmark results under `PR_dynamic_sweep/dim*` from before the dim-sweep query
file used the archived stratified set and are **not directly comparable** to new
runs on `random_queries_dim_sweep.json`.

## 2) Load phase (optional standalone)

From `vector-endpoint` root:

```bash
cd benchmarks/pr
../../.venv/bin/python run_dim_load_pipeline.py \
  --input-file ../../data/nts/RLUBM_cleaned.nt \
  --collection dim_benchmark \
  --dimensions 8,16,32,64,128,256,384 \
  --dim-adjustment truncate \
  --out-dir results
```

Outputs:

- `results/dim_load_pipeline_<timestamp>.json`
- `results/dim_load_pipeline_<timestamp>.csv`

## 3) Precision/Recall benchmark (with optional auto-load)

### Option A: benchmark only (assumes collection already loaded)

```bash
../../.venv/bin/python run_vector_dim_accuracy_benchmark.py \
  --collection dim_benchmark \
  --dimensions 8,16,32,64,128,256,384 \
  --dim-adjustment truncate \
  --queries-file results/archive/query_sets/random_queries_3000.json \
  --rdf-file ../../data/nts/RLUBM_cleaned.nt \
  --k 5000 \
  --out-dir results
```

### Option B: full sweep (auto-load each dim, benchmark, aggregate)

```bash
../../.venv/bin/python run_vector_dim_accuracy_benchmark.py \
  --collection dim_benchmark \
  --dimensions 8,16,32,64,128,256,384 \
  --dim-adjustment truncate \
  --run-load-phase \
  --load-input-file ../../data/nts/RLUBM_cleaned.nt \
  --queries-file results/archive/query_sets/random_queries_3000.json \
  --rdf-file ../../data/nts/RLUBM_cleaned.nt \
  --k 5000 \
  --accuracy-threshold-pct 95 \
  --out-dir results
```

Outputs:

- `results/vector_dim_pr_<timestamp>.json`
- `results/vector_dim_pr_<timestamp>_summary.csv`
- `results/vector_dim_pr_<timestamp>_per_query.csv`

## 4) Plot

```bash
../../.venv/bin/python plot_vector_dim_accuracy.py \
  --input results/vector_dim_pr_<timestamp>.json
```

Outputs:

- `results/vector_dim_pr_<timestamp>_precision_recall_vs_dim.png`
- `results/vector_dim_pr_<timestamp>_bucket_precision_recall.png`

## PR dynamic sweep (gRPC + catalog k)

Redo of the dimension P/R sweep using **catalog-based k** (not fixed `k=5000`)
over gRPC. Each dim: drop/reload `dim_benchmark` only, restart `grpc_app`, run
the **dim-sweep weighted** query set (~800 queries, cardinality-heavy `*po`).

**k policy:** `seed_k = max(10, ceil(catalog_count × 1.2))` per query.
`--use-adaptive --adaptive-multipliers 1` runs exactly one search at that k (no
10×/100× escalation).

**Component fusion:** `run_one_dim.sh` defaults to Hadamard fusion
(`COMPONENT_FUSION=hadamard`). Pass `--component-fusion` to the load/benchmark
scripts and set `VECTOR_COMPONENT_FUSION` on `grpc_app`. Concat remains the
library default elsewhere. Hadamard stores **d**-dim vectors (not `3×d`); missing
S/P/O slots use identity (ones) in the product, not zero vectors. Compare recall
at equal storage bytes when writing up results (e.g. Hadamard @ 384 vs concat @ 128
per component).

**Results layout:**

```
results/
  archive/PR_fixed_k5000_sweep/   # old fixed-k5000 dim runs
  archive/query_sets/             # archived query JSON (legacy / stratified)
  PR_dynamic_sweep/
    random_queries_dim_sweep.json
    load_phase/                   # catalog_dim{N}.pkl per dim
    dim8/                           # JSON/CSV + grpc_server.log per dim
    dim16/
    ...
```

### Prerequisites

1. Milvus up (`configs/milvus.yaml`, `common.topKLimit: 200000`).
2. No other `grpc_app` on port 50051 (`pkill -f vector_endpoint.grpc_app` if needed).
3. `comunica-sparql-file` on PATH (SPARQL baseline).

### Single dimension (smoke)

From `vector-endpoint/benchmarks/pr`:

```bash
./run_one_dim.sh 8
```

This runs: load RLUBM → `dim_benchmark` at dim 8 → restart gRPC with
`VECTOR_COLLECTION=dim_benchmark`, `VECTOR_TARGET_EMBEDDING_DIM=8`, matching
catalog → benchmark dim-sweep queries via gRPC.

### Full sweep (8 → 384)

```bash
./run_PR_dynamic_sweep.sh
```

Runs dims `8,16,32,64,128,256,384` sequentially. Production `version_5` is never
touched; only `dim_benchmark` is dropped/reloaded each iteration.

### Manual benchmark (after load + gRPC already running)

```bash
../../.venv/bin/python run_vector_dim_accuracy_benchmark.py \
  --collection dim_benchmark \
  --dimensions 8 \
  --queries-file results/PR_dynamic_sweep/random_queries_dim_sweep.json \
  --rdf-file ../../data/nts/RLUBM_cleaned.nt \
  --catalog-path results/PR_dynamic_sweep/load_phase/catalog_dim8.pkl \
  --catalog-k-scale 1.2 \
  --use-adaptive \
  --adaptive-multipliers 1 \
  --grpc-endpoint 127.0.0.1:50051 \
  --out-dir results/PR_dynamic_sweep/dim8
```

## LUBM Q1–Q14 join benchmark (`run_lubm_pr_benchmark.py`)

For the **official LUBM join queries** in `RLUBM/Query_Types.txt`, use
`run_lubm_pr_benchmark.py`. Unlike the dim sweep above (single-pattern Milvus
search + post-filter), this script runs **end-to-end** queries via
`comunica-vector` and compares binding sets against `comunica-sparql-file`
ground truth.

### Prerequisites

1. **vector-endpoint** running with RLUBM loaded (`version_5`, dim 384 / 1152 stored).
   Milvus must use `configs/milvus.yaml` (`common.topKLimit: 200000`) — see the
   root `README.md` Milvus section.

```bash
cd vector-endpoint
.venv/bin/python -m vector_endpoint.app
```

2. **Comunica CLIs** on PATH (`comunica-vector`, `comunica-sparql-file`).

### Run all Q1–Q14 (auto-k from expected result counts)

From `vector-endpoint/benchmarks/pr`:

```bash
../../.venv/bin/python run_lubm_pr_benchmark.py \
  --rdf-file ../../data/nts/RLUBM_cleaned.nt \
  --vector-endpoint http://localhost:2222/vector \
  --write-queries-json results/lubm_q1_q14.json \
  --out-dir results
```

**Catalog-based k (default when `--k` is omitted):** parses each BGP in the
query, looks up `catalog.pkl` match counts, and sets `seed_k = ceil(count ×
--catalog-k-scale)` per pattern (default scale `1.2`, min `--catalog-min-k`).
Baseline `LIMIT` is the **maximum ladder top** across all BGPs (not the final
tuple count from `Query_Types.txt`). JSON output includes
`pattern_catalog_plans` per query.

Requires `catalog.pkl` from the same load as Milvus (`--catalog-path` defaults
to `vector-endpoint/catalog.pkl`).

### Pagination k (Milvus search iterator, client-driven pages)

Fourth k policy: **`k_mode: "pagination"`** with **`k`** as Milvus page
`batch_size` and optional **`limit`** (default **`2 × catalog_k`** per pattern).
The client loops `QueryPatternPage` (gRPC) or POST `/vector` with `cursor` until
`pagination.done` is true.

Dim accuracy benchmark (requires gRPC server with pagination support):

```bash
../../.venv/bin/python run_vector_dim_accuracy_benchmark.py \
  --collection version_5 \
  --grpc-endpoint 127.0.0.1:50051 \
  --use-pagination \
  --k 500 \
  --catalog-path ../../catalog.pkl \
  --catalog-k-scale 1.2 \
  --out-dir results/pagination_sweep
```

Optional `--pagination-limit` caps total Milvus hits scanned (default
`2 × catalog_k`). Cannot combine with `--use-adaptive`.

### Adaptive k (catalog seed + multipliers per BGP)

Omits `-k` from `comunica-vector`. Each endpoint POST uses catalog seed k for
that BGP pattern, then `adaptive_batch_search` with
`--adaptive-multipliers` / `--adaptive-jaccard` (forwarded in the JSON body).
Baseline `LIMIT` uses the max ladder top over all BGPs in the query.

```bash
../../.venv/bin/python run_lubm_pr_benchmark.py \
  --use-adaptive \
  --adaptive-multipliers 1,10,100,1000 \
  --adaptive-jaccard 0.99 \
  --out-dir results
```

Optional: `--k 500` overrides catalog seed for every BGP in the plan (baseline
and vector when adaptive).

### Quick smoke (small queries only)

```bash
../../.venv/bin/python run_lubm_pr_benchmark.py \
  --query-nums 1,2,3,4,9,11,12,13 \
  --max-expected 200 \
  --out-dir results
```

### Fixed k for every query

```bash
../../.venv/bin/python run_lubm_pr_benchmark.py \
  --k 100 \
  --query-nums 1,2,3 \
  --out-dir results
```

Outputs:

- `results/lubm_pr_<timestamp>.json`
- `results/lubm_pr_<timestamp>_summary.csv`
- `results/lubm_pr_<timestamp>_per_query.csv`

## False-case debug export

Use this when you want to manually inspect false queries from a specific run (no load phase).

### Smoke (first 5 false queries at dim 128)

```bash
../../.venv/bin/python rerun_false_queries_debug.py \
  --collection dim_benchmark \
  --dimension 128 \
  --k 5000 \
  --per-query-csv results/vector_dim_pr_<timestamp>_per_query.csv \
  --queries-file results/archive/query_sets/random_queries_3000.json \
  --limit-queries 5 \
  --out-dir results/debug \
  --log
```

### Full rerun (all false queries at dim 128)

```bash
../../.venv/bin/python rerun_false_queries_debug.py \
  --collection dim_benchmark \
  --dimension 128 \
  --k 5000 \
  --per-query-csv results/vector_dim_pr_<timestamp>_per_query.csv \
  --queries-file results/archive/query_sets/random_queries_3000.json \
  --out-dir results/debug
```

Output:

- `results/debug/false_query_debug_<timestamp>.json`
