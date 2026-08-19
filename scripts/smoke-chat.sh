#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
load_config
command -v jq >/dev/null || die "jq is required"

payload="$(jq -n \
  --arg model "$SERVED_MODEL" \
  --arg prompt "$SMOKE_PROMPT" \
  --argjson max_tokens "$SMOKE_MAX_TOKENS" \
  --argjson temperature "$SMOKE_TEMPERATURE" \
  --argjson disable_thinking "$DISABLE_THINKING" \
  '{
    model: $model,
    messages: [{role: "user", content: $prompt}],
    max_tokens: $max_tokens,
    temperature: $temperature
  } + if $disable_thinking then {
    chat_template_kwargs: {enable_thinking: false}
  } else {} end')"

response="$(curl_api --max-time 180 \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "${API_URL}/v1/chat/completions")"

echo "$response" | jq -e --arg model "$SERVED_MODEL" '
  .model == $model and
  (.choices | length) > 0 and
  (.choices[0].message.content | type == "string")
' >/dev/null

echo "$response" | jq '{
  model,
  finish_reason: .choices[0].finish_reason,
  content: .choices[0].message.content,
  usage
}'
