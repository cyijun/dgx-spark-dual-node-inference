#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/5] Shell syntax"
while IFS= read -r script; do
  bash -n "$script"
done < <(find scripts patches -type f -name '*.sh' -print | sort)

echo "[2/5] Optional ShellCheck"
if command -v shellcheck >/dev/null; then
  while IFS= read -r script; do
    shellcheck -x "$script"
  done < <(find scripts patches -type f -name '*.sh' -print | sort)
else
  echo "shellcheck not installed; skipped"
fi

echo "[3/5] Secret and host-identity scan"
command -v rg >/dev/null || {
  echo "ripgrep (rg) is required for the public audit" >&2
  exit 1
}
if rg -n \
  --hidden \
  --glob '!scripts/public-audit.sh' \
  --glob '!.git/**' \
  '(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|spark-b4a6|spark-c829|tailnet\.cyjason)' \
  .; then
  echo "public audit found a possible secret or private host identity" >&2
  exit 1
fi
if rg -n \
  --pcre2 \
  --hidden \
  --glob '!scripts/public-audit.sh' \
  --glob '!.git/**' \
  '(?<!github\.com/)(?<!ghcr\.io/)cyijun' \
  .; then
  echo "public audit found a private user identity outside an allowed public GitHub/registry URL" >&2
  exit 1
fi

echo "[4/5] Large tracked-artifact scan"
large_files="$(find . -path './.git' -prune -o -type f -size +5M -print)"
if [[ -n "$large_files" ]]; then
  echo "files larger than 5 MiB are not allowed:" >&2
  printf '%s\n' "$large_files" >&2
  exit 1
fi

echo "[5/5] Local environment exclusion"
if [[ -e .env ]]; then
  if ! git check-ignore -q .env 2>/dev/null; then
    echo ".env exists but is not ignored" >&2
    exit 1
  fi
fi
if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo ".env must not be tracked" >&2
  exit 1
fi

echo "Public audit passed"
