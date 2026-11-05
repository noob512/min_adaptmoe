#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
PROBE="${SCRIPT_DIR}/fragmentation_cuda_malloc_probe.py"
FRAGMENT_VERIFY="${SCRIPT_DIR}/verify_fragmentation_cuda_malloc_probe.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/fragmentation_cuda_malloc_$(date -u +%Y%m%dT%H%M%SZ)}"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 2
}
for required in "${PROBE}" "${FRAGMENT_VERIFY}" "${BASE_HARNESS}"; do
  [[ -f "${required}" ]] || {
    echo "missing required file: ${required}" >&2
    exit 2
  }
done

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
mkdir -p "${RUN_ROOT}/contiguous" "${RUN_ROOT}/fragmented"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
KERNEL_EPOCH="$(date +%s)"

run_case() {
  local layout="$1" case_root="${RUN_ROOT}/$1"
  local -a command=(
    "${VLLM_PY}" "${PROBE}"
    --layout "${layout}" --device "${GPU_INDEX}"
    --arena-gib "${ARENA_GIB:-28}"
    --release-gib "${RELEASE_GIB:-4}"
    --native-gib "${NATIVE_GIB:-4}"
    --stripe-mib "${STRIPE_MIB:-16}"
  )
  printf '%q ' "${command[@]}" >"${case_root}/command.txt"
  printf '\n' >>"${case_root}/command.txt"
  set +e
  (
    cd "${VLLM_ROOT}"
    PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${command[@]}"
  ) >"${case_root}/probe.stdout" 2>"${case_root}/probe.stderr"
  local rc=$?
  set -e
  printf '%s\n' "${rc}" >"${case_root}/probe.returncode.txt"
  wait_gpu_clean || return 1
  return "${rc}"
}

load_custom_module
module_snapshot "${RUN_ROOT}/contiguous/module.custom.json" "${CUSTOM_UVM}"
CONTIGUOUS_RC=0
run_case contiguous || CONTIGUOUS_RC=$?

# Reload UVM between cases so the PMA starts from a fresh allocator state.
restore_stock
module_snapshot "${RUN_ROOT}/contiguous/module.stock_after.json" "${STOCK_UVM}"
MODULE_RESTORED=0
load_custom_module
module_snapshot "${RUN_ROOT}/fragmented/module.custom.json" "${CUSTOM_UVM}"
FRAGMENTED_RC=0
run_case fragmented || FRAGMENTED_RC=$?

VERIFY_RC=0
if (( CONTIGUOUS_RC == 0 && FRAGMENTED_RC == 0 )); then
  set +e
  "${VLLM_PY}" "${FRAGMENT_VERIFY}" \
    --contiguous "${RUN_ROOT}/contiguous/probe.stdout" \
    --fragmented "${RUN_ROOT}/fragmented/probe.stdout" \
    --output "${RUN_ROOT}/comparison.json" \
    >"${RUN_ROOT}/verify.stdout" 2>"${RUN_ROOT}/verify.stderr"
  VERIFY_RC=$?
  set -e
else
  VERIFY_RC=1
fi
printf '%s\n' "${VERIFY_RC}" >"${RUN_ROOT}/verify.returncode.txt"

sudo journalctl -k --since "@${KERNEL_EPOCH}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err" || true
if [[ -f "${RUN_ROOT}/comparison.json" ]]; then
  jq '{status, supports_fragmentation_hypothesis, checks, comparison}' \
    "${RUN_ROOT}/comparison.json"
fi
(( CONTIGUOUS_RC == 0 && FRAGMENTED_RC == 0 && VERIFY_RC == 0 ))
