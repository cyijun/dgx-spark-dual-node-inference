#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
command -v rsync >/dev/null || die "rsync is required"

model_parent="$(dirname -- "$MODEL_REPO")"
echo "Checksumming both model caches; this reads every model file."
differences="$(rsync -a --dry-run --checksum --itemize-changes \
  -e 'ssh -o BatchMode=yes' \
  "$MODEL_REPO" "${WORKER_SSH}:${model_parent}/")"

if [[ -n "$differences" ]]; then
  echo "Model caches differ:" >&2
  printf '%s\n' "$differences" >&2
  exit 1
fi

echo "Model caches match: ${MODEL_REV}"
