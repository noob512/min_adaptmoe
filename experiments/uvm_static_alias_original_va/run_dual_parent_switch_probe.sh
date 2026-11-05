#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
PROBE="${SCRIPT_DIR}/dual_parent_switch_probe.py"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/artifacts/dual-parent-switch-20260830-attempt-05-full-policy-partial-block/result.json}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ACCESS_COUNTER_MIGRATION_ENABLE=0
export UVM_PREFETCH_ENABLE=0
export UVM_THRASHING_ENABLE=0
export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
export SLOT_BUDGET_BYTES=1207959552
# shellcheck source=/dev/null
source "${BASE}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

require_file "${PROBE}"
[[ ! -e "${OUTPUT}" ]] || die "output already exists: ${OUTPUT}"
"${VLLM_PY}" -m py_compile "${PROBE}"
runtime_preflight
RUN_ROOT_ACTIVE="$(dirname "${OUTPUT}")"
mkdir -p "${RUN_ROOT_ACTIVE}"
RUN_ROOT_ACTIVE="$(cd "${RUN_ROOT_ACTIVE}" && pwd)"
OUTPUT="${RUN_ROOT_ACTIVE}/$(basename "${OUTPUT}")"
module_snapshot "${RUN_ROOT_ACTIVE}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT_ACTIVE}/module.custom.json" "${CUSTOM_UVM}"
cp -- "${STATS_PATH}" "${RUN_ROOT_ACTIVE}/latency_stats.before.txt"

kernel_epoch="$(date +%s)"
set +e
(
    cd "${VLLM_ROOT}"
    env PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        CUDA_VISIBLE_DEVICES=0 \
        "${VLLM_PY}" "${PROBE}" --output "${OUTPUT}"
) >"${RUN_ROOT_ACTIVE}/run.log" 2>&1
runner_rc=$?
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${RUN_ROOT_ACTIVE}/kernel.since.log" 2>"${RUN_ROOT_ACTIVE}/kernel.since.err"
journal_rc=$?
set -e
printf '%s\n' "${runner_rc}" >"${RUN_ROOT_ACTIVE}/returncode.txt"
cp -- "${STATS_PATH}" "${RUN_ROOT_ACTIVE}/latency_stats.after.txt"
wait_gpu_clean || die "GPU did not clean after dual-parent probe"
restore_stock || die "failed to restore stock UVM"
module_snapshot "${RUN_ROOT_ACTIVE}/module.after.json" "${STOCK_UVM}"
restart_exporters
(( journal_rc == 0 )) || die "kernel log collection failed"
if (( runner_rc != 0 )); then
    tail -n 200 "${RUN_ROOT_ACTIVE}/run.log" >&2 || true
    die "dual-parent probe failed"
fi
echo "result=${OUTPUT}"
