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

## Dimension configuration

- `--dimensions` is the **per-component** embedding dim: S, P and O are each
  truncated to this size and concatenated, so the stored vector is `3 × dim`.
- Default is `8` (→ 24-dim vectors), matching the production endpoint
  (`src/vector_endpoint/app.py`, `target_embedding_dim=8`).
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

## 1) Generate 3000 random queries

From `vector-endpoint` root:

```bash
cd benchmarks/pr
../../.venv/bin/python generate_random_queries.py \
  --input-file ../../data/nts/RLUBM_cleaned.nt \
  --seed 42 \
  --sp-count 1000 \
  --po-count 1000 \
  --so-count 1000 \
  --out results/random_queries_3000.json
```

This creates a reproducible mixed query set:

- 1000 `sp*`
- 1000 `*po`
- 1000 `s*o`

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
  --queries-file results/random_queries_3000.json \
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
  --queries-file results/random_queries_3000.json \
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

## False-case debug export

Use this when you want to manually inspect false queries from a specific run (no load phase).

### Smoke (first 5 false queries at dim 128)

```bash
../../.venv/bin/python rerun_false_queries_debug.py \
  --collection dim_benchmark \
  --dimension 128 \
  --k 5000 \
  --per-query-csv results/vector_dim_pr_<timestamp>_per_query.csv \
  --queries-file results/random_queries_3000.json \
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
  --queries-file results/random_queries_3000.json \
  --out-dir results/debug
```

Output:

- `results/debug/false_query_debug_<timestamp>.json`
