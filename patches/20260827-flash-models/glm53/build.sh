#!/usr/bin/env bash
set -Eeuo pipefail

docker build --progress=plain \
  --tag glm53-flash-nvfp4-gb10:sglang-sm121-tile32 \
  "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

docker image inspect glm53-flash-nvfp4-gb10:sglang-sm121-tile32 \
  --format 'image={{.Id}}'
