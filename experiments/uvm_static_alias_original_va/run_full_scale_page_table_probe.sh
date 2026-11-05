#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
PROBE="${SCRIPT_DIR}/full_scale_page_table_probe.py"
VERIFY="${SCRIPT_DIR}/verify_full_scale_page_table_probe.py"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e10-synthetic-page-table-34560-20260830-attempt-01-qualification}"
OUTPUT="${RUN_ROOT}/result.json"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ACCESS_COUNTER_MIGRATION_ENABLE=0
export UVM_PREFETCH_ENABLE=0
export UVM_THRASHING_ENABLE=0
export STATIC_ALIAS_DUAL_VIEW_ENABLE=0
export SLOT_BUDGET_BYTES=1207959552
export TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
# shellcheck source=/dev/null
source "${BASE}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

require_file "${PROBE}"
require_file "${VERIFY}"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
"${VLLM_PY}" -m py_compile "${PROBE}" "${VERIFY}"
runtime_preflight
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
OUTPUT="${RUN_ROOT}/result.json"

git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/git_status.txt"
nvidia-smi -i 0 --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader,nounits >"${RUN_ROOT}/hardware.csv"
sha256sum "${PROBE}" "${VERIFY}" "${BASH_SOURCE[0]}" "${CUSTOM_UVM}" "${STOCK_UVM}" \
  >"${RUN_ROOT}/input_sha256.txt"

module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
[[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_static_alias_dual_view_enable)" == 0 ]] \
  || die "dual-parent fast path must be disabled for the raw PTE probe"
wait_gpu_clean || die "GPU is not clean before the synthetic probe"
gpu_state_json "${RUN_ROOT}/gpu.before.json"
cp -- "${STATS_PATH}" "${RUN_ROOT}/latency_stats.before.txt"

kernel_epoch="$(date +%s)"
set +e
(
  cd "${VLLM_ROOT}"
  env \
    PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=14 \
    MKL_NUM_THREADS=14 \
    timeout --kill-after=30s "${TIMEOUT_SEC}s" \
    taskset -c 0-13 \
    "${VLLM_PY}" "${PROBE}" --output "${OUTPUT}"
) >"${RUN_ROOT}/run.log" 2>&1
runner_rc=$?
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err"
journal_rc=$?
set -e
printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/returncode.txt"
printf '%s\n' "${journal_rc}" >"${RUN_ROOT}/kernel_log_returncode.txt"
cp -- "${STATS_PATH}" "${RUN_ROOT}/latency_stats.after.txt"

wait_gpu_clean || die "GPU/client state did not clean after the synthetic probe"
gpu_state_json "${RUN_ROOT}/gpu.after.json"
restore_stock || die "failed to restore stock nvidia_uvm"
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters

(( journal_rc == 0 )) || die "kernel log collection failed"
if (( runner_rc != 0 )); then
  tail -n 300 "${RUN_ROOT}/run.log" >&2 || true
  die "the one-shot synthetic page-table probe failed"
fi

"${VLLM_PY}" "${VERIFY}" \
  --result "${OUTPUT}" \
  --kernel-log "${RUN_ROOT}/kernel.since.log" \
  --output "${RUN_ROOT}/verification.json"

echo "RUN_ROOT=${RUN_ROOT}"
echo "verification=${RUN_ROOT}/verification.json"
