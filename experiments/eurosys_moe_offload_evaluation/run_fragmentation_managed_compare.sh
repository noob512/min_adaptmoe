#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
MANAGED_PROBE="${SCRIPT_DIR}/fragmentation_cuda_malloc_probe.py"
MANAGED_VERIFY="${SCRIPT_DIR}/verify_fragmentation_managed_compare.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/fragmentation_managed_compare_$(date -u +%Y%m%dT%H%M%SZ)}"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 2
}
for required in "${MANAGED_PROBE}" "${MANAGED_VERIFY}" "${BASE_HARNESS}"; do
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

cleanup_compare() {
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

trap 'cleanup_compare $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

runtime_preflight
RUN_ROOT_ACTIVE="${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
KERNEL_EPOCH="$(date +%s)"

run_case() {
  local allocation="$1" layout="$2" case_root="${RUN_ROOT}/$1_$2"
  local -a command=(
    "${VLLM_PY}" "${MANAGED_PROBE}"
    --allocation-kind "${allocation}" --layout "${layout}"
    --device "${GPU_INDEX}"
    --arena-gib "${ARENA_GIB:-28}"
    --release-gib "${RELEASE_GIB:-4}"
    --native-gib "${NATIVE_GIB:-4}"
    --stripe-mib "${STRIPE_MIB:-16}"
    --bandwidth-repeats "${BANDWIDTH_REPEATS:-8}"
  )
  mkdir -p "${case_root}"
  module_snapshot "${case_root}/module.custom.json" "${CUSTOM_UVM}"
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

CASES=(
  "native contiguous"
  "managed contiguous"
  "native fragmented"
  "managed fragmented"
)
CASE_FAILURES=0
for index in "${!CASES[@]}"; do
  read -r allocation layout <<<"${CASES[index]}"
  if (( index == 0 )); then
    load_custom_module
  else
    restore_stock
    MODULE_RESTORED=0
    load_custom_module
  fi
  run_case "${allocation}" "${layout}" || CASE_FAILURES=$((CASE_FAILURES + 1))
done

VERIFY_RC=1
if (( CASE_FAILURES == 0 )); then
  set +e
  "${VLLM_PY}" "${MANAGED_VERIFY}" \
    --native-contiguous "${RUN_ROOT}/native_contiguous/probe.stdout" \
    --managed-contiguous "${RUN_ROOT}/managed_contiguous/probe.stdout" \
    --native-fragmented "${RUN_ROOT}/native_fragmented/probe.stdout" \
    --managed-fragmented "${RUN_ROOT}/managed_fragmented/probe.stdout" \
    --output "${RUN_ROOT}/comparison.json" \
    >"${RUN_ROOT}/verify.stdout" 2>"${RUN_ROOT}/verify.stderr"
  VERIFY_RC=$?
  set -e
fi
printf '%s\n' "${VERIFY_RC}" >"${RUN_ROOT}/verify.returncode.txt"

sudo journalctl -k --since "@${KERNEL_EPOCH}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err" || true
if [[ -f "${RUN_ROOT}/comparison.json" ]]; then
  jq '{status, managed_avoids_fragmentation_eviction, checks,
       residency_comparison, performance}' "${RUN_ROOT}/comparison.json"
fi
(( CASE_FAILURES == 0 && VERIFY_RC == 0 ))
