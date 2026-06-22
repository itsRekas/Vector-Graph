#!/usr/bin/env bash
# Load dim_benchmark at one embedding dim, restart gRPC, run dim-sweep P/R benchmark.
set -euo pipefail

DIM="${1:?usage: run_one_dim.sh <dim>}"
COMPONENT_FUSION="${COMPONENT_FUSION:-concat}"
# EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-mixedbread-ai/mxbai-embed-xsmall-v1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PR="$(cd "$(dirname "$0")" && pwd)"
SWEEP="$PR/results/PR_dynamic_sweep"
VENV="$ROOT/.venv/bin/python"
NT="$ROOT/data/nts/RLUBM_cleaned.nt"
QUERIES="${QUERIES_FILE:-$SWEEP/random_queries_dim_sweep.json}"
CATALOG="$SWEEP/load_phase/catalog_dim${DIM}.pkl"
OUT="$SWEEP/dim${DIM}"
GRPC_PORT="${VECTOR_GRPC_PORT:-50051}"
CACHE="$SWEEP/load_phase/embedding_cache_full384.npz"
CACHE_META="$SWEEP/load_phase/embedding_cache_full384_meta.json"
SHARED_CATALOG="$SWEEP/load_phase/catalog.pkl"

mkdir -p "$OUT" "$SWEEP/load_phase"

if [[ -f "$CACHE" && -f "$CACHE_META" ]]; then
  cached_model="$("$VENV" -c "import json; print(json.load(open('$CACHE_META'))['embedding_model'])")"
  if [[ "$cached_model" != "$EMBEDDING_MODEL" ]]; then
    echo "Stale embedding cache (model=${cached_model}, want=${EMBEDDING_MODEL}); removing..."
    rm -f "$CACHE" "$CACHE_META"
  fi
fi

if [[ ! -f "$CACHE" ]]; then
  echo "=== PRECOMPUTE embedding cache (dim=384) ==="
  cd "$PR"
  "$VENV" run_dim_load_pipeline.py \
    --input-file "$NT" \
    --collection dim_benchmark \
    --dimensions 384 \
    --embedding-model "$EMBEDDING_MODEL" \
    --dim-adjustment truncate \
    --component-fusion "$COMPONENT_FUSION" \
    --embed-cache-only \
    --embedding-cache-out "$CACHE" \
    --out-dir "$SWEEP/load_phase" \
    --log
fi

echo "=== LOAD dim=${DIM} (from embedding cache) ==="
cd "$PR"
"$VENV" run_dim_load_pipeline.py \
  --input-file "$NT" \
  --collection dim_benchmark \
  --dimensions "$DIM" \
  --embedding-model "$EMBEDDING_MODEL" \
  --dim-adjustment truncate \
  --component-fusion "$COMPONENT_FUSION" \
  --embedding-cache-in "$CACHE" \
  --catalog-in "$SHARED_CATALOG" \
  --out-dir "$SWEEP/load_phase" \
  --log

if [[ ! -f "$CATALOG" ]]; then
  echo "ERROR: expected catalog at $CATALOG" >&2
  exit 1
fi

echo "=== RESTART gRPC (dim=${DIM}) ==="
pkill -f 'vector_endpoint.grpc_app' 2>/dev/null || true
sleep 2

cd "$ROOT"
# VECTOR_EMBEDDING_MODEL="all-MiniLM-L6-v2"
VECTOR_COLLECTION=dim_benchmark \
VECTOR_TARGET_EMBEDDING_DIM="$DIM" \
VECTOR_EMBEDDING_MODEL="$EMBEDDING_MODEL" \
VECTOR_COMPONENT_FUSION="$COMPONENT_FUSION" \
VECTOR_CATALOG_PATH="$CATALOG" \
VECTOR_BGP_LOG=1 \
VECTOR_DEVICE="${VECTOR_DEVICE:-}" \
PYTHONUNBUFFERED=1 \
nohup "$VENV" -m vector_endpoint.grpc_app > "$OUT/grpc_server.log" 2>&1 &
GRPC_PID=$!

echo "Waiting for gRPC server and embedding model (up to 180s)..."
DEADLINE=$((SECONDS + 180))
while (( SECONDS < DEADLINE )); do
  if ! kill -0 "$GRPC_PID" 2>/dev/null; then
    echo "ERROR: gRPC server failed to start. See $OUT/grpc_server.log" >&2
    tail -30 "$OUT/grpc_server.log" >&2 || true
    exit 1
  fi
  if grep -q "Initializing embedding model on device:" "$OUT/grpc_server.log" 2>/dev/null; then
    break
  fi
  sleep 2
done

if ! grep -q "Initializing embedding model on device:" "$OUT/grpc_server.log" 2>/dev/null; then
  echo "ERROR: embedding model did not initialize within 180s" >&2
  tail -30 "$OUT/grpc_server.log" >&2 || true
  exit 1
fi

if ! kill -0 "$GRPC_PID" 2>/dev/null; then
  echo "ERROR: gRPC server failed to start. See $OUT/grpc_server.log" >&2
  tail -20 "$OUT/grpc_server.log" >&2 || true
  exit 1
fi

echo "=== VERIFY embedding device (gRPC log + query embed path) ==="
if [[ -n "${VECTOR_DEVICE:-}" ]]; then
  EXPECTED_DEVICE="$VECTOR_DEVICE"
else
  EXPECTED_DEVICE="$("$VENV" -c "from vector_endpoint.db.VectorDataBase import _resolve_embedding_device; print(_resolve_embedding_device())")"
fi
echo "Expected embedding device: ${EXPECTED_DEVICE}"

if ! grep -q "Initializing embedding model on device: ${EXPECTED_DEVICE}" "$OUT/grpc_server.log"; then
  echo "ERROR: gRPC server did not initialize embedding model on ${EXPECTED_DEVICE}" >&2
  echo "grpc_server.log (last 30 lines):" >&2
  tail -30 "$OUT/grpc_server.log" >&2 || true
  exit 1
fi
grep "Initializing embedding model on device:" "$OUT/grpc_server.log" | tail -1

if ! VECTOR_EMBEDDING_MODEL="$EMBEDDING_MODEL" \
     VECTOR_TARGET_EMBEDDING_DIM="$DIM" \
     VECTOR_DEVICE="${VECTOR_DEVICE:-}" \
     "$VENV" "$ROOT/scripts/verify_query_embedding_device.py"; then
  echo "ERROR: query embedding device check failed" >&2
  exit 1
fi

echo "=== BENCHMARK dim=${DIM} fusion=${COMPONENT_FUSION} (gRPC + catalog k x1) ==="
cd "$PR"
"$VENV" run_vector_dim_accuracy_benchmark.py \
  --collection dim_benchmark \
  --dimensions "$DIM" \
  --dim-adjustment truncate \
  --component-fusion "$COMPONENT_FUSION" \
  --queries-file "$QUERIES" \
  --rdf-file "$NT" \
  --catalog-path "$CATALOG" \
  --catalog-k-scale 1.2 \
  --catalog-min-k 10 \
  --use-adaptive \
  --adaptive-multipliers 1 \
  --adaptive-jaccard 0.99 \
  --grpc-endpoint "127.0.0.1:${GRPC_PORT}" \
  --out-dir "$OUT" \
  --log

echo "Done dim=${DIM}. Results in $OUT"
