#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
command -v rsync >/dev/null || die "rsync is required"

[[ -d "$MODEL_REPO" ]] || die "missing source model cache: ${MODEL_REPO}"
model_parent="$(dirname -- "$MODEL_REPO")"
remote_exec mkdir -p "$model_parent"

echo "Syncing ${MODEL_REPO} to ${WORKER_SSH}:${model_parent}/"
rsync -a --partial --info=progress2,stats2 \
  -e 'ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6' \
  "$MODEL_REPO" "${WORKER_SSH}:${model_parent}/"

echo "Sync complete; run make verify-model"
