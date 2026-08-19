SHELL := /usr/bin/env bash

.PHONY: audit preflight sync-model verify-model up wait status smoke memory mtp-metrics logs down

audit:
	./scripts/public-audit.sh

preflight:
	./scripts/preflight.sh

sync-model:
	./scripts/sync-model-cache.sh

verify-model:
	./scripts/verify-model-sync.sh

up:
	./scripts/up.sh

wait:
	./scripts/wait-ready.sh

status:
	./scripts/status.sh

smoke:
	./scripts/smoke-chat.sh

memory:
	./scripts/memory-report.sh

mtp-metrics:
	./scripts/mtp-metrics.sh

logs:
	docker logs -f "$$(. ./scripts/common.sh; load_config; printf '%s' "$$HEAD_CONTAINER")"

down:
	./scripts/down.sh
