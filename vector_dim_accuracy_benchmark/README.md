# Vector Dim Accuracy Benchmark

This benchmark estimates the **lowest plausible embedding dimension** that still matches SPARQL baseline results closely enough.

It is split into two phases:

1. **Load phase** (`run_dim_load_pipeline.py`)
2. **Accuracy phase** (`run_vector_dim_accuracy_benchmark.py`)

Safety behavior:

- This benchmark only allows collection name **`dim_benchmark`**.
- If `dim_benchmark` already exists, it is dropped before each new dimension load.
- Other collections (including your main pipeline collection) are not targeted by these scripts.

## Metrics

For each query and dimension:

- `count_match`: vector result count equals SPARQL baseline count
- `overlap_jaccard`: overlap between result binding sets
- `query_accuracy_pct`: `100` if both count match and overlap threshold pass, else `0`

Per dimension:

- `overall_accuracy_pct = queries_passed / total_queries * 100`
- a dimension passes if `overall_accuracy_pct >= --accuracy-threshold-pct` (default `95`)

The output reports `lowest_passing_dimension`.

## Dimension configuration

- Default component dim is still `384` (model-native for `all-MiniLM-L6-v2`).
- You can set lower dims via `--dimensions` (e.g. `256`) for this benchmark flow.
- Dim adjustment mode is controlled by `--dim-adjustment` and defaults to `truncate`.
- In truncate mode:
  - `target_embedding_dim <= model_dim` works (e.g. 256 from a 384-d model)
  - `target_embedding_dim > model_dim` fails fast with a clear error

## 1) Load phase

From `vector-endpoint` root:

```bash
cd vector_dim_accuracy_benchmark
../.venv/bin/python run_dim_load_pipeline.py \
  --input-file ../data/nts/RLUBM_cleaned.nt \
  --collection dim_benchmark \
  --dimensions 384,256,192,128,96,64,48,32 \
  --dim-adjustment truncate \
  --out-dir results
```

Outputs:

- `results/dim_load_pipeline_<timestamp>.json`
- `results/dim_load_pipeline_<timestamp>.csv`

## 2) Accuracy benchmark (with optional auto-load)

### Option A: benchmark only (assumes collection already loaded for your chosen dim)

```bash
../.venv/bin/python run_vector_dim_accuracy_benchmark.py \
  --collection dim_benchmark \
  --dimensions 384 \
  --dim-adjustment truncate \
  --rdf-file ../data/nts/RLUBM_cleaned.nt \
  --k 1000 \
  --out-dir results
```

### Option B: full sweep (auto-load each dim, benchmark, aggregate)

```bash
../.venv/bin/python run_vector_dim_accuracy_benchmark.py \
  --collection dim_benchmark \
  --dimensions 384,256,192,128,96,64,48,32 \
  --dim-adjustment truncate \
  --run-load-phase \
  --load-input-file ../data/nts/RLUBM_cleaned.nt \
  --rdf-file ../data/nts/RLUBM_cleaned.nt \
  --k 1000 \
  --accuracy-threshold-pct 95 \
  --overlap-threshold 0.95 \
  --out-dir results
```

Outputs:

- `results/vector_dim_accuracy_<timestamp>.json`
- `results/vector_dim_accuracy_<timestamp>.csv`

## 3) Plot

```bash
../.venv/bin/python plot_vector_dim_accuracy.py \
  --input results/vector_dim_accuracy_<timestamp>.json
```

Outputs:

- `results/vector_dim_accuracy_<timestamp>_accuracy_vs_dim.png`
- `results/vector_dim_accuracy_<timestamp>_query_counts.png`

## Custom query set

Use `--queries-file` with JSON entries:

```json
[
  {
    "id": "Q1",
    "query": "SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> }"
  }
]
```
