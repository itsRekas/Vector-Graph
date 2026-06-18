#!/usr/bin/env bash
# Full dimension sweep: 8,16,32,64,128,256,384 via run_one_dim.sh
set -euo pipefail

PR="$(cd "$(dirname "$0")" && pwd)"
DIMS=(384 256 128 64 32 16 8)

echo "PR dynamic sweep: gRPC + catalog k (scale=1.2, multiplier=1, fusion=hadamard)"
echo "Embedding cache: $PR/results/PR_dynamic_sweep/load_phase/embedding_cache_full384.npz"
echo "Dims: ${DIMS[*]}"
echo "Results: $PR/results/PR_dynamic_sweep/"
echo ""

for DIM in "${DIMS[@]}"; do
  echo "########## dim=${DIM} ##########"
  "$PR/run_one_dim.sh" "$DIM"
  echo ""
done

echo "Sweep complete."
