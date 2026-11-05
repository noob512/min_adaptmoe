#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OFFERED_QPS=1 TRACE_NAMES=short
bash "${SCRIPT_DIR}/run_mixed_2g_qps05_uva_adaptmoe.sh" "${1:?new absolute artifact root required}"
