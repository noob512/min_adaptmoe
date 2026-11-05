#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPARE_DIR="${REPO_ROOT}/experiments/vllm_prefill_p3_pinned_uva_compare"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_clock_gap_PCU_v1}"

reset_clocks() {
  sudo -n nvidia-smi -i "${GPU_INDEX:-0}" -rgc >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i "${GPU_INDEX:-0}" -rmc >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i "${GPU_INDEX:-0}" -pm 0 >/dev/null 2>&1 || true
}

on_exit() {
  local rc="$1"
  trap - EXIT INT TERM
  reset_clocks
  exit "${rc}"
}

trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
reset_clocks

env \
  RUN_ROOT="${RUN_ROOT}" \
  ORDER=PCU \
  CONCURRENCY=4 \
  WARMUP_REQUESTS=4 \
  DISABLE_CHUNKED_PREFILL=1 \
  MAX_MODEL_LEN=512 \
  MAX_NUM_BATCHED_TOKENS=2048 \
  MAX_NUM_SEQS=4 \
  COPY_TRACE_ENABLE=0 \
  ROUTING_TRACE_ENABLE=0 \
  BENCHMARK="${SCRIPT_DIR}/benchmark_clock_phases.py" \
  VERIFY="${SCRIPT_DIR}/verify_clock_diagnosis.py" \
  RUNNER_SCRIPT="${COMPARE_DIR}/run_compare.sh" \
  "${COMPARE_DIR}/run_with_module_cycle.sh"

echo "RUN_ROOT=${RUN_ROOT}"
