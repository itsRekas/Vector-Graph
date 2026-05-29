# Archive

Frozen legacy code. **Not part of the active pipeline.** Kept for reference only.

The active project lives at the repo root:

- `src/` — the vector endpoint engine and Flask app (`vectorEndpoint.py`, `catalog.py`,
  `auto_k.py`, `load.py`, `adaptive_exp/`, `vector_endpoint/db/VectorDataBase.py`)
- `pr_benchmark/` — current precision/recall benchmark
- `string_match_benchmark/` — string-vs-embedding benchmark

## Contents

| Folder | What it was |
|--------|-------------|
| `versions/` | Old endpoint generations V1–V4 + `v_unnormalized` (imported the old `lib`, now broken) |
| `lib_legacy/` | Stale `VectorDataBase_unnormalized.py`; superseded by `src/vector_endpoint/db/VectorDataBase.py` |
| `backup/` | Backup copy of an old endpoint + DB |
| `v_1/` | Old V1 test + result PNG |
| `others/` | Scratch/debug endpoints (`debug_sparql.py`, `oldvEp.py`, `simple_endpoint.py`) |
| `benchmarks_legacy/` | Old `benchmark_v1`–`v4` flow (`comunica-vector` + `Query_Types.txt`); replaced by `pr_benchmark/` |
| `results_legacy/` | Old `benchmark_results_v*.json` outputs |
| `vector_dim_accuracy_benchmark/` | Earlier standalone copy of the dim-accuracy benchmark; superseded by `pr_benchmark/` |
| `VERSION_MAPPING.md` | Documents the archived V1–V4 versions |

See git history for full context on any file here.
