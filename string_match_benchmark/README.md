# String vs Embedding Part-Match Benchmark

This benchmark compares **post-filter-only** comparison time between:

- **String match baseline**: exact equality checks on requested S/P/O constants.
- **Embedding part-match**: compare requested S/P/O constant parts against corresponding result parts using cosine similarity.

The benchmark measures only the post-filter stage on the same retrieved candidates.

## Files

- `run_string_vs_embedding_benchmark.py`: runs both matchers repeatedly and writes raw timing outputs.
- `plot_string_vs_embedding.py`: builds a line graph with mean and 95% CI.
- `results/`: output JSON/CSV/PNG files.

## Run benchmark (30 runs)

From `vector-endpoint` root:

```bash
cd string_match_benchmark
../.venv/bin/python run_string_vs_embedding_benchmark.py \
  --collection version_5 \
  --query 'SELECT ?X WHERE { ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> }' \
  --k 1000 \
  --runs 30 \
  --similarity-threshold 0.999
```

This writes:

- `results/string_vs_embedding_<timestamp>.json`
- `results/string_vs_embedding_<timestamp>.csv`

## Generate graph (line + mean/CI)

```bash
../.venv/bin/python plot_string_vs_embedding.py \
  --input results/string_vs_embedding_<timestamp>.json
```

This writes:

- `results/string_vs_embedding_<timestamp>_line_mean_ci.png`

## Notes

- Requested wildcard parts (e.g. `?s`, `?p`, `?o`) are skipped in both matchers.
- Embedding part comparison uses a configurable cosine threshold (`--similarity-threshold`).
- For literal objects, embeddings use the same `literal:<value>` convention used by the main vector database implementation.

## Generic standalone micro-benchmark (no system dependency)

Use this when you need a pure micro-benchmark that is not tied to Milvus, catalog data, or the endpoint stack.
It generates large synthetic string pairs and compares:

- exact pattern-style string checks on selected parts (`s`, `p`, `o`)
- embedding self-match checks on the same parts using cosine thresholding

### Run generic benchmark

```bash
cd string_match_benchmark
../.venv/bin/python run_generic_microbenchmark.py \
  --num-records 1000000 \
  --constrained-parts o \
  --embedding-dim 128 \
  --runs 10 \
  --warmup-runs 2 \
  --similarity-threshold 0.999
```

This writes:

- `results/generic_string_vs_embedding_<timestamp>.json`
- `results/generic_string_vs_embedding_<timestamp>.csv`

### Key options

- `--num-records`: synthetic workload size (can be millions)
- `--constrained-parts`: which pattern parts are checked (`s`, `p`, `o`, `sp`, `so`, `po`, `spo`)
- `--mismatch-ratio`: fraction of records forced to mismatch on constrained parts
- `--embedding-dim`: fixed embedding dimension for the run
- `--min-len` / `--max-len`: synthetic string length range

### Reported metrics

- per-run latency (`string_ms`, `embedding_ms`)
- throughput (`string_cmp_per_sec`, `embedding_cmp_per_sec`)
- summary stats (mean, p50, p95, min, max, ci95)
- mean latency ratio (`embedding_over_string_mean_time_ratio`)
