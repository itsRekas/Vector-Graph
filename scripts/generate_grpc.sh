#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="${ROOT}/proto"
OUT_DIR="${ROOT}/src/vector_endpoint/grpc_gen"
mkdir -p "${OUT_DIR}"
python -m grpc_tools.protoc \
  -I"${PROTO_DIR}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  --pyi_out="${OUT_DIR}" \
  vector/v1/pattern.proto
touch "${OUT_DIR}/__init__.py"
touch "${OUT_DIR}/vector/__init__.py"
touch "${OUT_DIR}/vector/v1/__init__.py"
GRPC_FILE="${OUT_DIR}/vector/v1/pattern_pb2_grpc.py"
if grep -q '^from vector.v1 import pattern_pb2' "${GRPC_FILE}"; then
  if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' 's/^from vector.v1 import pattern_pb2/from vector_endpoint.grpc_gen.vector.v1 import pattern_pb2/' "${GRPC_FILE}"
  else
    sed -i 's/^from vector.v1 import pattern_pb2/from vector_endpoint.grpc_gen.vector.v1 import pattern_pb2/' "${GRPC_FILE}"
  fi
fi
echo "Generated gRPC stubs in ${OUT_DIR}"
