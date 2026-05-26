#!/bin/zsh
set -euo pipefail

REAL_LLAMA="${QPU_MCP_LAB_LLAMA_BIN:-$HOME/src/ik_llama.cpp/build-air-iqk-lean/bin/llama-cli}"

if [[ -n "${LLAMA_TASKPOLICY_ARGS:-}" ]]; then
  exec /usr/sbin/taskpolicy ${(z)LLAMA_TASKPOLICY_ARGS} "$REAL_LLAMA" "$@"
fi

exec "$REAL_LLAMA" "$@"
