#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config

metrics="$(curl_api --max-time 5 "${API_URL}/metrics")"
matched="$(printf '%s\n' "$metrics" | grep -E '^vllm:spec_decode_(num_drafts|num_draft_tokens|num_accepted_tokens)(_total)?\{' || true)"

if [[ -z "$matched" ]]; then
  echo "No speculative-decoding metrics found. MTP may be disabled or no request has completed yet."
  exit 1
fi

printf '%s\n' "$matched"
