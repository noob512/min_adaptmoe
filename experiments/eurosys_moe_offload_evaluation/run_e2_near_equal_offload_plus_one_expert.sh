#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e2-near-equal-offload-plus-one-expert-$(date -u +%Y%m%dT%H%M%SZ)}"

E2_EQUAL_C1_PROFILE="${SCRIPT_DIR}/configs/e2_c1_near_equal_offload_plus_one_expert.json" \
E2_EQUAL_C16_PROFILE="${SCRIPT_DIR}/configs/e2_c16_near_equal_offload_plus_one_expert.json" \
E2_EQUAL_VERIFY_MODE=realized_within_10_percent \
E2_EQUAL_NATIVE_CUDAGRAPH_MODE=FULL_DECODE_ONLY \
ALLOW_GPU_RUN="${ALLOW_GPU_RUN:-0}" \
bash "${SCRIPT_DIR}/run_e2_equal_offload_native_uva_graph.sh" "${RUN_ROOT}"
