#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_BASE_ID="sha256:64c58f100438fa5f036bdfbeb3edd3136fb12c5d22d8ae52786c4a701263c55d"
BASE_IMAGE="lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1"
OUTPUT_IMAGE="qwen38-flash-next-gb10:sm121-qsa-reference"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

actual_base_id="$(docker image inspect --format '{{.Id}}' "$BASE_IMAGE")"
[[ "$actual_base_id" == "$EXPECTED_BASE_ID" ]] || {
  echo "unexpected base image: ${actual_base_id}" >&2
  exit 1
}

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --tag "$OUTPUT_IMAGE" \
  "$SCRIPT_DIR"

docker run --rm -i --entrypoint python3 "$OUTPUT_IMAGE" - <<'PY'
from pathlib import Path

path = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py")
source = path.read_text()
marker = "flash-attn-4's CuTe varlen epilogue rejects Qwen4-Exp's packed QSA"
assert marker in source
print("SM121 QSA reference fallback verified")
PY

docker image inspect --format '{{.Id}}' "$OUTPUT_IMAGE"
