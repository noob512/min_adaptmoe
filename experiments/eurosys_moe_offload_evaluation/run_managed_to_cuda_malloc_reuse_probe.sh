#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
PROBE="${REPO_ROOT}/workloads/vllm/vllm/uvm_test/managed_to_cuda_malloc_reuse_probe"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/managed_to_cuda_malloc_reuse_$(date -u +%Y%m%dT%H%M%SZ)}"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 2
}
[[ -x "${PROBE}" ]] || {
  echo "probe binary is missing: ${PROBE}" >&2
  exit 2
}

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ALLOW_GPU_RUN=1
export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1207959552}"
# shellcheck source=/dev/null
source "${BASE_HARNESS}"

cleanup_probe() {
  local original_rc="$1" recovery_rc=0
  trap - EXIT INT TERM
  wait_gpu_clean || recovery_rc=1
  restore_stock || recovery_rc=1
  if [[ -d "${RUN_ROOT}" && "${MODULE_RESTORED}" == 1 ]]; then
    module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}" \
      || recovery_rc=1
  fi
  restart_exporters
  (( original_rc == 0 )) || exit "${original_rc}"
  exit "${recovery_rc}"
}

trap 'cleanup_probe $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

runtime_preflight
mkdir -p "${RUN_ROOT}"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"

KERNEL_EPOCH="$(date +%s)"
COMMAND=(
  "${PROBE}"
  --device "${GPU_INDEX}"
  --hot-gib "${HOT_GIB:-20}"
  --expert-gib "${EXPERT_GIB:-8}"
  --native-gib "${NATIVE_GIB:-6}"
)
printf '%q ' "${COMMAND[@]}" >"${RUN_ROOT}/command.txt"
printf '\n' >>"${RUN_ROOT}/command.txt"

set +e
"${COMMAND[@]}" >"${RUN_ROOT}/probe.stdout" 2>"${RUN_ROOT}/probe.stderr"
PROBE_RC="$?"
set -e
printf '%s\n' "${PROBE_RC}" >"${RUN_ROOT}/probe.returncode.txt"

sudo journalctl -k --since "@${KERNEL_EPOCH}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err" || true
cat "${RUN_ROOT}/probe.stdout"
exit "${PROBE_RC}"
