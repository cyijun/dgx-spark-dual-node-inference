#!/usr/bin/env bash

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

warn() {
  echo "WARNING: $*" >&2
}

require_var() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "${name} is required in ${ENV_FILE}"
}

validate_bool() {
  local name="$1"
  case "${!name}" in
    true|false) ;;
    *) die "${name} must be true or false" ;;
  esac
}

load_config() {
  [[ -f "$ENV_FILE" ]] || die "missing ${ENV_FILE}; copy .env.example to .env and edit it"

  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a

  : "${MASTER_PORT:=29501}"
  : "${API_BIND:=127.0.0.1}"
  : "${API_PORT:=8888}"
  : "${API_KEY:=}"
  : "${ALLOW_REMOTE_API:=false}"
  : "${MAX_MODEL_LEN:=131072}"
  : "${GPU_MEMORY_UTILIZATION:=0.75}"
  : "${MAX_NUM_SEQS:=8}"
  : "${MAX_NUM_BATCHED_TOKENS:=8192}"
  : "${REASONING_PARSER:=}"
  : "${QUANTIZATION:=}"
  : "${ENABLE_MTP:=false}"
  : "${SPEC_METHOD:=mtp}"
  : "${SPEC_TOKENS:=1}"
  : "${ENABLE_AUTO_TOOL_CHOICE:=false}"
  : "${TOOL_CALL_PARSER:=}"
  : "${NCCL_DEBUG:=WARN}"
  : "${NCCL_DEBUG_SUBSYS:=INIT,NET}"
  : "${STARTUP_TIMEOUT_SECONDS:=720}"
  : "${SMOKE_PROMPT:=用一句话介绍张量并行。}"
  : "${SMOKE_MAX_TOKENS:=96}"
  : "${SMOKE_TEMPERATURE:=0}"
  : "${DISABLE_THINKING:=true}"
  : "${EXPECTED_IMAGE_ID:=}"

  local required=(
    WORKER_SSH HEAD_IP WORKER_IP FABRIC_NIC FABRIC_HCA IMAGE MODEL_REPO
    MODEL_REV SERVED_MODEL CONTAINER_PREFIX
  )
  local name
  for name in "${required[@]}"; do
    require_var "$name"
  done

  validate_bool ALLOW_REMOTE_API
  validate_bool ENABLE_MTP
  validate_bool ENABLE_AUTO_TOOL_CHOICE
  validate_bool DISABLE_THINKING

  [[ "$API_PORT" =~ ^[0-9]+$ ]] || die "API_PORT must be numeric"
  [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "MASTER_PORT must be numeric"
  [[ "$SPEC_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "SPEC_TOKENS must be a positive integer"

  if [[ "$API_BIND" != "127.0.0.1" && "$API_BIND" != "::1" ]]; then
    [[ "$ALLOW_REMOTE_API" == "true" ]] || die "non-loopback API_BIND requires ALLOW_REMOTE_API=true"
    (( ${#API_KEY} >= 24 )) || die "non-loopback API_BIND requires an API_KEY of at least 24 characters"
  fi

  MODEL_PATH="/model/snapshots/${MODEL_REV}"
  HEAD_CONTAINER="${CONTAINER_PREFIX}-head"
  WORKER_CONTAINER="${CONTAINER_PREFIX}-worker"
  if [[ "$API_BIND" == *:* ]]; then
    API_URL="http://[${API_BIND}]:${API_PORT}"
  else
    API_URL="http://${API_BIND}:${API_PORT}"
  fi
  SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
}

remote_exec() {
  local command_string="" arg quoted
  for arg in "$@"; do
    printf -v quoted '%q' "$arg"
    command_string+="${quoted} "
  done
  "${SSH[@]}" "$WORKER_SSH" "$command_string"
}

curl_api() {
  local args=(--silent --show-error --fail)
  if [[ -n "$API_KEY" ]]; then
    args+=(-H "Authorization: Bearer ${API_KEY}")
  fi
  curl "${args[@]}" "$@"
}

head_rdma_devices() {
  find /dev/infiniband -maxdepth 1 -type c \( -name rdma_cm -o -name 'uverbs*' \) -print 2>/dev/null | sort
}

worker_rdma_devices() {
  remote_exec find /dev/infiniband -maxdepth 1 -type c \( -name rdma_cm -o -name 'uverbs*' \) -print 2>/dev/null | sort
}

detect_head_gid() {
  "${REPO_ROOT}/scripts/detect-rocev2-gid.sh" "$FABRIC_HCA" "$FABRIC_NIC"
}

detect_worker_gid() {
  "${SSH[@]}" "$WORKER_SSH" bash -s -- "$FABRIC_HCA" "$FABRIC_NIC" \
    < "${REPO_ROOT}/scripts/detect-rocev2-gid.sh"
}
