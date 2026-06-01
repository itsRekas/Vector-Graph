# TurboVec Precision/Recall Benchmark

This benchmark measures the **precision** and **recall** of a
[`turbovec`](https://pypi.org/project/turbovec/) (TurboQuant) index as a drop-in
replacement for the Milvus HNSW vector index, at the current production config
(per-component embedding `dim=8` -> concatenated 24-d vectors).

It answers one question: **does TurboQuant's 2-bit / 4-bit quantization preserve
enough accuracy at 24-d to match Milvus HNSW?**

Only the index/search stage changes. Embeddings are produced by the exact same
`VectorDataBase._embed_triple_batch` path as production, and scoring reuses the
`benchmarks/pr/` pipeline (exact string post-filter + comparison against
`comunica-sparql-file` ground truth), so the numbers are directly comparable to
the stored Milvus dim-8 baseline:

- Milvus HNSW (dim 8, k=10): **precision 94.73% / recall 93.48%**
  (`../pr/results/vector_dim_pr_20260529T140118Z_summary.csv`)

## Why recall is the discriminator

The `pr` pipeline applies an **exact string post-filter** on the requested
S/P/O constants before scoring. A quantization-induced bad candidate has the
wrong constants and is dropped, so it never becomes a false positive. Therefore
precision tracks Milvus closely regardless of index; **quantization loss shows up
in recall** (a true triple ranked out of the top-k).

## Fixed k=10

The stored Milvus run used a per-query `seed_k=10`, so this benchmark uses a
fixed `k=10` for both the TurboVec search and the SPARQL ground-truth `LIMIT`.
At k=10 the bar is tight: recall depends on whether the true triple lands in the
top-10 of the quantized flat scan, so any quantization loss is clearly visible.

## Install

`turbovec` is not a core dependency. Install it into the project venv:

```bash
../../.venv/bin/pip install turbovec
```

## Run

From the `vector-endpoint` root:

```bash
cd benchmarks/turbovec
../../.venv/bin/python run_turbovec_pr_benchmark.py \
  --input-file ../../data/nts/RLUBM_cleaned.nt \
  --rdf-file ../../data/nts/RLUBM_cleaned.nt \
  --queries-file ../pr/results/random_queries_3000.json \
  --dimension 8 \
  --bit-widths 2,4 \
  --k 10 \
  --out-dir results
```

Requires `comunica-sparql-file` on `PATH` (same as `benchmarks/pr/`). The SPARQL
ground truth is computed once and cached, then reused across both bit-widths.

Outputs (per timestamp):

- `results/turbovec_pr_<timestamp>_summary.csv` - one row per bit-width.
- `results/turbovec_pr_<timestamp>_per_query.csv` - per-query TP/FP/FN/P/R.
- `results/turbovec_pr_<timestamp>.json` - full config + per-dimension payload
  (same shape as the `pr` benchmark, so `plot_turbovec_pr.py` and the `pr`
  plotter can both read it).

## Plot

```bash
../../.venv/bin/python plot_turbovec_pr.py \
  --input results/turbovec_pr_<timestamp>.json
```

Produces a grouped bar chart of precision/recall per bit-width with the Milvus
HNSW reference line.

## Memory footprint

Reported per row as `footprint_bytes` ~= `24 * bit_width / 8 + 8` (the `+8`
covers the stored norm + length-renormalization scalar). For reference, the
Milvus float32 vector is `24 * 4 = 96` bytes.

## Metrics

Per query and bit-width (identical definitions to `benchmarks/pr/`):

- `TP`, `FP`, `FN`, `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `jaccard`,
  `exact_match`.
- Per bit-width: `avg_precision`, `avg_recall`, bucket averages for `sp*`,
  `*po`, `s*o`, and `passes_threshold` (both >= 95%).

## Results (dim 8, k=10, 230,062 triples, 3000 queries)

| index | precision | recall | footprint/vec |
| --- | --- | --- | --- |
| Milvus HNSW (float32) | 94.73% | 93.48% | 96 B |
| TurboVec 4-bit | 41.50% | 37.69% | 20 B |
| TurboVec 2-bit | 8.01% | 6.65% | 14 B |

At the production config (per-component dim 8 -> 24-d vectors), TurboQuant
quantization collapses accuracy: even 4-bit loses more than half the recall, and
2-bit is unusable. 24-d is far below where
TurboQuant's random-rotation Gaussian assumption holds, and recall@10 among
230k crowded vectors is an unforgiving bar.

Validated as a genuine quantization effect (not a pipeline bug): an exact-float
brute-force search over the **same** 24-d vectors recovers ~98% recall on a
stratified 90-query sample, matching/exceeding Milvus, while TurboVec 4-bit stays
near its full-run value. Conclusion for this config: TurboVec is not a viable
drop-in at dim 8. Any future interest in TurboQuant should test higher embedding
dimensions (its competitive regime), which is Phase 2 in the plan.

## Environment notes

- The embedding model is loaded via `sentence-transformers`; if it is already
  cached locally you can run fully offline with `HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1`.
- For `plot_turbovec_pr.py` in restricted/sandboxed environments, point
  matplotlib at a writable cache with `MPLCONFIGDIR=$PWD/.mplcache`.
