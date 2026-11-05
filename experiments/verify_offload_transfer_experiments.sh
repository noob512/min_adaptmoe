#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_LIVE_SMOKE="${RUN_LIVE_SMOKE:-0}"
CHECK_LIVE_STATE="${CHECK_LIVE_STATE:-1}"

case "${RUN_LIVE_SMOKE}" in
  0|1) ;;
  *) echo "RUN_LIVE_SMOKE must be 0 or 1" >&2; exit 2 ;;
esac
case "${CHECK_LIVE_STATE}" in
  0|1) ;;
  *) echo "CHECK_LIVE_STATE must be 0 or 1" >&2; exit 2 ;;
esac

cd "${ROOT_DIR}"

if [[ "${RUN_LIVE_SMOKE}" == 1 ]]; then
  group_result="/tmp/vllm_group_smoke_$(date +%Y%m%d_%H%M%S).txt"
  transfer_result="/tmp/uvm_transfer_smoke_$(date +%Y%m%d_%H%M%S).txt"
  RESULT_FILE="${group_result}" \
    bash experiments/vllm_group_prefetch_baseline/run_group_benchmark.sh
  RESULT_FILE="${transfer_result}" \
    bash experiments/uvm_transfer_microbench/run_benchmark.sh
  make -C experiments/vllm_group_prefetch_baseline clean
  make -C experiments/uvm_transfer_microbench clean
fi

python3 experiments/vllm_group_prefetch_baseline/verify_group_results.py
uvm_verify_args=()
if [[ "${CHECK_LIVE_STATE}" == 1 ]]; then
  uvm_verify_args+=(--check-live-state)
fi
python3 experiments/uvm_transfer_microbench/verify_results.py \
  "${uvm_verify_args[@]}"
python3 experiments/uvm_prefetch_diagnostics/uvm_profile.py self-test

echo "validation_result success=1 live_smoke=${RUN_LIVE_SMOKE} live_state=${CHECK_LIVE_STATE} vllm_started=0"
