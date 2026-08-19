#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config

"${SCRIPT_DIR}/preflight.sh"

head_gid="$(detect_head_gid)"
worker_gid="$(detect_worker_gid)"
mapfile -t head_devices < <(head_rdma_devices)
mapfile -t worker_devices < <(worker_rdma_devices)

vllm_args=(
  serve "$MODEL_PATH"
  --tensor-parallel-size 2
  --pipeline-parallel-size 1
  --distributed-executor-backend mp
  --nnodes 2
  --master-addr "$HEAD_IP"
  --master-port "$MASTER_PORT"
  --served-model-name "$SERVED_MODEL"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --disable-custom-all-reduce
  --distributed-timeout-seconds 1800
)

[[ -z "$REASONING_PARSER" ]] || vllm_args+=(--reasoning-parser "$REASONING_PARSER")
[[ -z "$QUANTIZATION" ]] || vllm_args+=(--quantization "$QUANTIZATION")
if [[ "$ENABLE_MTP" == "true" ]]; then
  vllm_args+=(--spec-method "$SPEC_METHOD" --spec-tokens "$SPEC_TOKENS")
fi
if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "true" ]]; then
  [[ -n "$TOOL_CALL_PARSER" ]] || die "TOOL_CALL_PARSER is required when ENABLE_AUTO_TOOL_CHOICE=true"
  vllm_args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi
[[ -z "$API_KEY" ]] || vllm_args+=(--api-key "$API_KEY")

common_docker=(
  --gpus all
  --network host
  --ipc host
  --cap-add IPC_LOCK
  --ulimit memlock=-1:-1
  --mount "type=bind,src=${MODEL_REPO},dst=/model,readonly"
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  --env NCCL_IB_DISABLE=0
  --env NCCL_NET=IB
  --env "NCCL_DEBUG=${NCCL_DEBUG}"
  --env "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --entrypoint vllm
)

head_docker=(docker run -d --name "$HEAD_CONTAINER" --label io.github.dgx-spark-dual-node-inference=service)
worker_docker=(docker run -d --name "$WORKER_CONTAINER" --label io.github.dgx-spark-dual-node-inference=service)

for device in "${head_devices[@]}"; do
  head_docker+=(--device "$device")
done
for device in "${worker_devices[@]}"; do
  worker_docker+=(--device "$device")
done

head_docker+=(
  "${common_docker[@]}"
  --env "VLLM_HOST_IP=${HEAD_IP}"
  --env "GLOO_SOCKET_IFNAME=${FABRIC_NIC}"
  --env "NCCL_SOCKET_IFNAME=${FABRIC_NIC}"
  --env "NCCL_IB_HCA=${FABRIC_HCA}"
  --env "NCCL_IB_GID_INDEX=${head_gid}"
  "$IMAGE" "${vllm_args[@]}"
  --node-rank 0
  --host "$API_BIND"
  --port "$API_PORT"
)

worker_docker+=(
  "${common_docker[@]}"
  --env "VLLM_HOST_IP=${WORKER_IP}"
  --env "GLOO_SOCKET_IFNAME=${FABRIC_NIC}"
  --env "NCCL_SOCKET_IFNAME=${FABRIC_NIC}"
  --env "NCCL_IB_HCA=${FABRIC_HCA}"
  --env "NCCL_IB_GID_INDEX=${worker_gid}"
  "$IMAGE" "${vllm_args[@]}"
  --node-rank 1
  --headless
)

cleanup_on_error() {
  local status=$?
  if (( status != 0 )); then
    warn "deployment failed; removing containers created by this run"
    docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1 || true
    remote_exec docker rm -f "$WORKER_CONTAINER" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup_on_error EXIT

echo "Starting worker rank 1"
remote_exec "${worker_docker[@]}"

echo "Starting head rank 0"
"${head_docker[@]}"

trap - EXIT
echo "Containers started; run make wait"
