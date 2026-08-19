#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
attempt=0

while (( SECONDS < deadline )); do
  attempt=$((attempt + 1))
  if curl_api --max-time 3 "${API_URL}/health" >/dev/null 2>&1; then
    echo "API ready after ${attempt} checks: ${API_URL}"
    curl_api --max-time 5 "${API_URL}/v1/models"
    echo
    exit 0
  fi

  head_state="$(docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$HEAD_CONTAINER" 2>/dev/null || true)"
  worker_state="$(remote_exec docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$WORKER_CONTAINER" 2>/dev/null || true)"
  printf 'check=%03d head=%s worker=%s\n' "$attempt" "$head_state" "$worker_state"

  if [[ "$head_state" != running* || "$worker_state" != running* ]]; then
    warn "a container exited during startup"
    docker logs --tail 100 "$HEAD_CONTAINER" 2>&1 || true
    remote_exec docker logs --tail 100 "$WORKER_CONTAINER" 2>&1 || true
    exit 1
  fi
  sleep 5
done

warn "API did not become ready within ${STARTUP_TIMEOUT_SECONDS} seconds"
docker logs --tail 100 "$HEAD_CONTAINER" 2>&1 || true
remote_exec docker logs --tail 100 "$WORKER_CONTAINER" 2>&1 || true
exit 1
