#!/usr/bin/env bash
# End-to-end smoke test driver. Run from the repo root on the host:
#   ./docker/e2e/run_e2e.sh [image-tag]
# Requires: docker (image already built via docker/Dockerfile.devel), python3.
set -euo pipefail

IMAGE="${1:-catboost-serving:devel}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="$(mktemp -d /tmp/catboost-e2e.XXXXXX)"
PORT="${PORT:-8500}"
CONTAINER=""

cleanup() {
  [[ -n "$CONTAINER" ]] && docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo ">> staging models and configs in $WORK"
for m in numeric_only categorical_only mixed_multiclass; do
  mkdir -p "$WORK/models/$m/1"
  cp "$REPO_ROOT/cb/testdata/$m.cbm" "$WORK/models/$m/1/model.cbm"
done
mkdir -p "$WORK/config"
cp "$REPO_ROOT/docker/e2e/models.conf" "$REPO_ROOT/docker/e2e/platform.conf" "$WORK/config/"

echo ">> starting server ($IMAGE) on :$PORT"
CONTAINER=$(docker run -d --platform linux/amd64 -p "$PORT:8500" \
  -v "$WORK/models:/models" -v "$WORK/config:/config" "$IMAGE")

echo ">> python venv + stubs"
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" -q install grpcio grpcio-tools
mkdir -p "$WORK/stubs"
"$WORK/venv/bin/python" -m grpc_tools.protoc \
  -I "$REPO_ROOT" -I "$REPO_ROOT/third_party/tensorflow_serving_apis" \
  --python_out="$WORK/stubs" --grpc_python_out="$WORK/stubs" \
  tfs/proto/catboost_predict.proto \
  tensorflow_serving/apis/model.proto
touch "$WORK/stubs/tfs/__init__.py" "$WORK/stubs/tfs/proto/__init__.py"

echo ">> running e2e assertions"
"$WORK/venv/bin/python" "$REPO_ROOT/docker/e2e/e2e_test.py" \
  --target "localhost:$PORT" \
  --stub_dir "$WORK/stubs" \
  --models_dir "$WORK/models" \
  --golden "$REPO_ROOT/cb/testdata/expected_predictions.json"
