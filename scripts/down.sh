#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config

echo "Stopping head"
docker stop --time 30 "$HEAD_CONTAINER" >/dev/null 2>&1 || true
docker rm "$HEAD_CONTAINER" >/dev/null 2>&1 || true

echo "Stopping worker"
remote_exec docker stop --time 30 "$WORKER_CONTAINER" >/dev/null 2>&1 || true
remote_exec docker rm "$WORKER_CONTAINER" >/dev/null 2>&1 || true

echo "Stopped"
