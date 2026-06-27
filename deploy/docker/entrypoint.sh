#!/bin/sh
set -e
MODE="${SERVE_MODE:-grpc}"
MODEL="${MODEL_PATH:-/models/model.onnx}"
PORT="${PORT:-50051}"

if [ "$MODE" = "fastapi" ]; then
  exec python -m apps.serve.serve_fastapi --model "$MODEL" --port "${FASTAPI_PORT:-8000}" --host 0.0.0.0 --device auto
fi

exec python -m apps.serve.serve_grpc --model "$MODEL" --port "$PORT" --device auto
