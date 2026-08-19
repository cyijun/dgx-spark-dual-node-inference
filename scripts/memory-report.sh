#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config

pattern='Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|CUDA graph pool memory|Free memory on device|Actual usage'

echo "Head Linux unified memory"
free -h
echo
echo "Head CUDA process accounting"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
echo
echo "Head vLLM memory profile"
docker logs "$HEAD_CONTAINER" 2>&1 | grep -E "$pattern" | tail -n 20
echo

echo "Worker Linux unified memory"
remote_exec free -h
echo
echo "Worker CUDA process accounting"
remote_exec nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
echo
echo "Worker vLLM memory profile"
remote_exec docker logs "$WORKER_CONTAINER" 2>&1 | grep -E "$pattern" | tail -n 20
