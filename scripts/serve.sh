#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to a local model directory}"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU explicitly}"
exec "${PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name "${SERVED_MODEL:-masbench-model}" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" "$@"
