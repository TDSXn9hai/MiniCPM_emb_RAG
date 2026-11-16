#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${RAG_PYTHON:-${ROOT}/../.venv/bin/python}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export RAG_RUNTIME=mock
export RAG_ASR="${RAG_ASR:-manual}"

exec "${PYTHON}" -m uvicorn server:app \
    --host "${RAG_HOST:-0.0.0.0}" \
    --port "${RAG_PORT:-8010}"
