# Vector Endpoint Version Mapping

This document maps each endpoint version to its corresponding VectorDatabase implementation and collection.

## Version Overview

| Version | Endpoint File | VectorDatabase Class | Collection Name | Embedding Strategy | Model | Embedding Dim |
|---------|--------------|---------------------|----------------|-------------------|-------|--------------|
| V1 | `vectorEndpoint_v_1.py` | `lib.VectorDataBase_v_1.VectorDataBase` | `lubm_graph_v_1` | Raw triple text (no S\|P\|O separation) | all-MiniLM-L6-v2 | 384 |
| V2 | `vectorEndpoint_v_2.py` | `lib.VectorDataBase_v_2.VectorDataBase` | `lubm_graph_v_2` | S\|P\|O separate, concatenated | paraphrase-multilingual-MiniLM-L12-v2 | 1152 (3×384) |
| V3 | `vectorEndpoint_v_3.py` | `lib.VectorDataBase_v_3.VectorDataBase` | `lubm_graph_v_3` | S\|P\|O separate, concatenated, normalized | paraphrase-multilingual-MiniLM-L12-v2 | 1152 (3×384) |
| V4 | `vectorEndpoint.py` | `lib.VectorDataBase.VectorDataBase` | `lubm_graph_v1_normalized` | S\|P\|O separate, concatenated, normalized | all-MiniLM-L6-v2 | 1152 (3×384) |

## Detailed Version Information

### Version 1
- **Endpoint**: `vectorEndpoint_v_1.py`
- **VectorDatabase**: `lib/VectorDataBase_v_1.py`
- **Collection**: `lubm_graph_v_1`
- **Embedding Approach**: Embeds entire triple as raw text string
- **Dimension**: Model dimension only (384 for all-MiniLM-L6-v2)
- **Model**: `all-MiniLM-L6-v2`
- **Benchmark Script**: `v_1/benchmark.py`

### Version 2
- **Endpoint**: `vectorEndpoint_v_2.py`
- **VectorDatabase**: `lib/VectorDataBase_v_2.py`
- **Collection**: `lubm_graph_v_2`
- **Embedding Approach**: Embeds subject, predicate, and object separately, then concatenates
- **Dimension**: 3 × model dimension (1152 for 384-dim model)
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Benchmark Script**: `benchmark_v2.py`

### Version 3
- **Endpoint**: `vectorEndpoint_v_3.py`
- **VectorDatabase**: `lib/VectorDataBase_v_3.py`
- **Collection**: `lubm_graph_v_3`
- **Embedding Approach**: Embeds subject, predicate, and object separately, concatenates, and normalizes embeddings
- **Dimension**: 3 × model dimension (1152 for 384-dim model)
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Benchmark Script**: `benchmark_v3.py`
- **Note**: Uses normalized parts and random vectors for missing parts

### Version 4
- **Endpoint**: `vectorEndpoint.py` (main/default)
- **VectorDatabase**: `lib/VectorDataBase.py` (main implementation)
- **Collection**: `lubm_graph_v1_normalized`
- **Embedding Approach**: Embeds subject, predicate, and object separately, concatenates, and normalizes embeddings
- **Dimension**: 3 × model dimension (1152 for 384-dim model)
- **Model**: `all-MiniLM-L6-v2`
- **Benchmark Script**: `benchmark_v4.py`
- **Special Features**: Supports k parameter forwarding from Comunica to vector endpoint

## Key Differences

1. **V1 vs V2-V4**: V1 embeds triples as raw text, while V2-V4 embed S, P, O components separately
2. **V2 vs V3-V4**: V2 doesn't normalize embeddings, while V3 and V4 normalize during storage
3. **V3 vs V4**: Different models (L12 vs L6) and different collection names, but same embedding strategy
4. **V4**: Latest version with k parameter forwarding support

## Usage

Each endpoint version should be used with its corresponding:
- VectorDatabase implementation class
- Collection name in Milvus
- Benchmark script for testing

Make sure the collection exists in Milvus before running the endpoint!

