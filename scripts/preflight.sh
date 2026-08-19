#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config

echo "[1/8] Local tools and SSH"
for command in docker ssh python3 ip ping find nvidia-smi; do
  command -v "$command" >/dev/null || die "${command} is not installed"
done
remote_exec true >/dev/null || die "cannot SSH to ${WORKER_SSH}"

echo "[2/8] Architecture and image identity"
head_arch="$(uname -m)"
worker_arch="$(remote_exec uname -m)"
[[ "$head_arch" == "aarch64" ]] || die "head architecture is ${head_arch}, expected aarch64"
[[ "$worker_arch" == "aarch64" ]] || die "worker architecture is ${worker_arch}, expected aarch64"

head_gpu="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader)"
worker_gpu="$(remote_exec nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader)"
[[ "$head_gpu" == NVIDIA\ GB10,\ 12.1,* ]] || die "head GPU is ${head_gpu}, expected NVIDIA GB10 SM121"
[[ "$worker_gpu" == NVIDIA\ GB10,\ 12.1,* ]] || die "worker GPU is ${worker_gpu}, expected NVIDIA GB10 SM121"
[[ "$head_gpu" == "$worker_gpu" ]] || die "GPU/driver snapshots differ: head=${head_gpu}, worker=${worker_gpu}"

head_image="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
worker_image="$(remote_exec docker image inspect --format '{{.Id}}' "$IMAGE")"
[[ "$head_image" == "$worker_image" ]] || die "image IDs differ: head=${head_image}, worker=${worker_image}"
if [[ -n "$EXPECTED_IMAGE_ID" ]]; then
  [[ "$head_image" == "$EXPECTED_IMAGE_ID" ]] || die "image ID ${head_image} does not match EXPECTED_IMAGE_ID"
fi
image_arch="$(docker image inspect --format '{{.Architecture}}' "$IMAGE")"
[[ "$image_arch" == "arm64" ]] || die "image architecture is ${image_arch}, expected arm64"

echo "[3/8] Model cache and revision"
snapshot="${MODEL_REPO}/snapshots/${MODEL_REV}"
[[ -f "${snapshot}/config.json" ]] || die "missing ${snapshot}/config.json on head"
remote_exec test -f "${snapshot}/config.json" || die "missing ${snapshot}/config.json on worker"

head_rev="$(<"${MODEL_REPO}/refs/main")"
worker_rev="$(remote_exec cat "${MODEL_REPO}/refs/main")"
[[ "$head_rev" == "$MODEL_REV" ]] || die "head main revision is ${head_rev}, expected ${MODEL_REV}"
[[ "$worker_rev" == "$MODEL_REV" ]] || die "worker main revision is ${worker_rev}, expected ${MODEL_REV}"

head_incomplete="$(find "$MODEL_REPO" -type f \( -name '*.incomplete' -o -name '*.lock' \) | wc -l)"
worker_incomplete="$(remote_exec find "$MODEL_REPO" -type f \( -name '*.incomplete' -o -name '*.lock' \) | wc -l)"
[[ "$head_incomplete" == 0 ]] || die "head cache contains incomplete or lock files"
[[ "$worker_incomplete" == 0 ]] || die "worker cache contains incomplete or lock files"
[[ -z "$(find "$MODEL_REPO" -xtype l -print -quit)" ]] || die "head snapshot contains broken symlinks"
[[ -z "$(remote_exec find "$MODEL_REPO" -xtype l -print -quit)" ]] || die "worker snapshot contains broken symlinks"

echo "[4/8] RoCE link, interface, and peer reachability"
head_state="$(<"/sys/class/infiniband/${FABRIC_HCA}/ports/1/state")"
worker_state="$(remote_exec cat "/sys/class/infiniband/${FABRIC_HCA}/ports/1/state")"
[[ "$head_state" == "4: ACTIVE" ]] || die "head RDMA state is ${head_state}"
[[ "$worker_state" == "4: ACTIVE" ]] || die "worker RDMA state is ${worker_state}"

ip -4 -o addr show dev "$FABRIC_NIC" | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$HEAD_IP" \
  || die "${HEAD_IP} is not assigned to head interface ${FABRIC_NIC}"
remote_exec ip -4 -o addr show dev "$FABRIC_NIC" | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$WORKER_IP" \
  || die "${WORKER_IP} is not assigned to worker interface ${FABRIC_NIC}"
ping -c 2 -W 2 -I "$FABRIC_NIC" "$WORKER_IP" >/dev/null || die "cannot reach ${WORKER_IP} through ${FABRIC_NIC}"

echo "[5/8] RoCEv2 GID discovery"
head_gid="$(detect_head_gid)" || die "cannot find head IPv4 RoCEv2 GID"
worker_gid="$(detect_worker_gid)" || die "cannot find worker IPv4 RoCEv2 GID"

echo "[6/8] RDMA device nodes"
mapfile -t head_devices < <(head_rdma_devices)
mapfile -t worker_devices < <(worker_rdma_devices)
(( ${#head_devices[@]} >= 2 )) || die "head has insufficient /dev/infiniband device nodes"
(( ${#worker_devices[@]} >= 2 )) || die "worker has insufficient /dev/infiniband device nodes"

echo "[7/8] Container and port conflicts"
if docker container inspect "$HEAD_CONTAINER" >/dev/null 2>&1; then
  die "container ${HEAD_CONTAINER} already exists; run make down first"
fi
if remote_exec docker container inspect "$WORKER_CONTAINER" >/dev/null 2>&1; then
  die "container ${WORKER_CONTAINER} already exists; run make down first"
fi
if ss -lnt "sport = :${MASTER_PORT}" | tail -n +2 | grep -q .; then
  die "master port ${MASTER_PORT} is already in use on the head"
fi
if ! python3 - "$API_BIND" "$API_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
family = socket.AF_INET6 if ":" in host else socket.AF_INET
with socket.socket(family) as sock:
    sock.bind((host, port))
PY
then
  die "API address ${API_BIND}:${API_PORT} is already in use on the head"
fi

echo "[8/8] Capacity snapshot"
head_model_size="$(du -sh "$MODEL_REPO" | awk '{print $1}')"
worker_model_size="$(remote_exec du -sh "$MODEL_REPO" | awk '{print $1}')"
head_available="$(df -h --output=avail "$MODEL_REPO" | tail -n 1 | xargs)"
worker_available="$(remote_exec df -h --output=avail "$MODEL_REPO" | tail -n 1 | xargs)"

cat <<EOF
Preflight passed
  image=${head_image}
  gpu=${head_gpu}
  revision=${MODEL_REV}
  model_size: head=${head_model_size}, worker=${worker_model_size}
  disk_available: head=${head_available}, worker=${worker_available}
  RoCEv2 GID: head=${head_gid}, worker=${worker_gid}
  API=${API_BIND}:${API_PORT}
EOF
