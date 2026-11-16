#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${RAG_PYTHON:-${ROOT}/../.venv/bin/python}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export RAG_RUNTIME=minicpm
export RAG_ASR="${RAG_ASR:-whisper}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${RAG_ASR_MODEL:-}" && -f "${ROOT}/models/whisper-small/model.safetensors" ]]; then
    export RAG_ASR_MODEL="${ROOT}/models/whisper-small"
fi

exec "${PYTHON}" -m uvicorn server:app \
    --host "${RAG_HOST:-0.0.0.0}" \
    --port "${RAG_PORT:-8010}"
