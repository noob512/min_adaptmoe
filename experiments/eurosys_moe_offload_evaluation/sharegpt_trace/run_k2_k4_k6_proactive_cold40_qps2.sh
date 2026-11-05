#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Request-pressure-driven adjacent growth at the existing 80% high watermark.
# This does not set a manual target or inject a synthetic request.
export ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW=1
exec bash "${SCRIPT_DIR}/run_k2_k4_k6_cold40_qps2.sh" "${1:?run root required}"
