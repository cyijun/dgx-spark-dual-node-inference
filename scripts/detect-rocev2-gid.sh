#!/usr/bin/env bash
set -Eeuo pipefail

HCA="${1:?usage: detect-rocev2-gid.sh HCA NIC}"
NIC="${2:?usage: detect-rocev2-gid.sh HCA NIC}"
BASE="/sys/class/infiniband/${HCA}/ports/1"

[[ -d "$BASE" ]] || exit 1

for type_file in "${BASE}/gid_attrs/types/"*; do
  idx="${type_file##*/}"
  type="$(<"$type_file")"
  ndev="$(<"${BASE}/gid_attrs/ndevs/${idx}")"
  gid="$(<"${BASE}/gids/${idx}")"
  if [[ "$type" == "RoCE v2" && "$ndev" == "$NIC" && "$gid" == *":ffff:"* ]]; then
    printf '%s\n' "$idx"
    exit 0
  fi
done

exit 1
