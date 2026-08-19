#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config

echo "Head container"
docker ps -a --filter "name=^/${HEAD_CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
echo
echo "Worker container"
remote_exec docker ps -a --filter "name=^/${WORKER_CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
echo

if curl_api --max-time 3 "${API_URL}/health" >/dev/null; then
  echo "API healthy: ${API_URL}"
  curl_api --max-time 5 "${API_URL}/v1/models"
  echo
else
  warn "API is not healthy"
  exit 1
fi
