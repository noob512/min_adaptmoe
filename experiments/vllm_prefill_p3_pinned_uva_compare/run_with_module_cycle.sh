#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUSTOM_UVM="${CUSTOM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${SCRIPT_DIR}/run_compare.sh}"
POOL_BUDGET_BYTES="${POOL_BUDGET_BYTES:-1207959552}"
EPOCH_VALIDATION="${EPOCH_VALIDATION:-0}"
BANK_LEVEL_COMMIT="${BANK_LEVEL_COMMIT:-0}"
TRACE_MODE="${TRACE_MODE:-full}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_PCU_v1}"
CYCLE_LOG="${RUN_ROOT}.module_cycle.log"
STOCK_UVM="$(modinfo -F filename nvidia-uvm)"
STOCK_SRCVERSION="$(modinfo -F srcversion nvidia-uvm)"
NODE_WAS_ACTIVE="$(systemctl is-active node-exporter.service || true)"
GPU_WAS_ACTIVE="$(systemctl is-active gpu-exporter.service || true)"
RECOVERY_DONE=0

case "${EPOCH_VALIDATION}" in
  0|1) ;;
  *) echo "error: EPOCH_VALIDATION must be 0 or 1" >&2; exit 1 ;;
esac
case "${BANK_LEVEL_COMMIT}" in
  0|1) ;;
  *) echo "error: BANK_LEVEL_COMMIT must be 0 or 1" >&2; exit 1 ;;
esac
case "${TRACE_MODE}" in
  full|compact|errors|off) ;;
  *) echo "error: TRACE_MODE must be full, compact, errors, or off" >&2; exit 1 ;;
esac
gpu_clean() {
  local used pids
  used="$(nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
  [[ "${used}" == "0" && -z "${pids}" ]]
}

restore_stock() {
  local rc=0 loaded attempt
  [[ "${RECOVERY_DONE}" == 0 ]] || return 0
  RECOVERY_DONE=1
  set +e
  for ((attempt = 0; attempt < 240; ++attempt)); do
    gpu_clean && break
    sleep 1
  done
  loaded="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion 2>/dev/null)"
  if ! gpu_clean; then
    echo "recovery: GPU is not clean; refusing unsafe UVM unload" >&2
    rc=1
  elif [[ "${loaded}" != "${STOCK_SRCVERSION}" ]] && sudo fuser /dev/nvidia-uvm >/dev/null 2>&1; then
    echo "recovery: /dev/nvidia-uvm still has a client; refusing unload" >&2
    rc=1
  elif [[ "${loaded}" != "${STOCK_SRCVERSION}" ]]; then
    sudo modprobe -r nvidia_uvm || rc=1
    sudo modprobe nvidia || rc=1
    sudo modprobe nvidia_uvm || rc=1
  fi
  [[ "${NODE_WAS_ACTIVE}" == "active" ]] && sudo systemctl start node-exporter.service || true
  [[ "${GPU_WAS_ACTIVE}" == "active" ]] && sudo systemctl start gpu-exporter.service || true
  loaded="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion 2>/dev/null)"
  [[ "${loaded}" == "${STOCK_SRCVERSION}" ]] || rc=1
  [[ "$(readlink -f "$(modinfo -F filename nvidia-uvm)")" == "$(readlink -f "${STOCK_UVM}")" ]] || rc=1
  gpu_clean || rc=1
  echo "recovery: loaded_srcversion=${loaded} gpu_clean=$(gpu_clean && echo yes || echo no) rc=${rc}"
  set -e
  return "${rc}"
}

on_exit() {
  local original_rc="$1" recovery_rc=0
  trap - EXIT INT TERM
  restore_stock || recovery_rc=$?
  if [[ -d "${RUN_ROOT}" && -f "${CYCLE_LOG}" ]]; then
    mv -- "${CYCLE_LOG}" "${RUN_ROOT}/module_cycle.log"
  fi
  (( original_rc == 0 )) || exit "${original_rc}"
  exit "${recovery_rc}"
}

trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$(dirname "${CYCLE_LOG}")"
exec > >(tee -a "${CYCLE_LOG}") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "stock_module=${STOCK_UVM}"
echo "stock_srcversion=${STOCK_SRCVERSION}"
echo "custom_module=${CUSTOM_UVM}"
echo "custom_srcversion=$(modinfo -F srcversion "${CUSTOM_UVM}")"
echo "runner_script=${RUNNER_SCRIPT}"
echo "pool_budget_bytes=${POOL_BUDGET_BYTES}"
echo "epoch_validation=${EPOCH_VALIDATION}"
echo "bank_level_commit=${BANK_LEVEL_COMMIT}"
echo "trace_mode=${TRACE_MODE}"
echo "run_root=${RUN_ROOT}"
[[ -f "${CUSTOM_UVM}" && -x "${RUNNER_SCRIPT}" && ! -e "${RUN_ROOT}" ]]
gpu_clean
sudo systemctl stop node-exporter.service gpu-exporter.service
if sudo fuser /dev/nvidia-uvm >/dev/null 2>&1; then
  echo "preflight: /dev/nvidia-uvm has a live client" >&2
  exit 1
fi

sudo modprobe -r nvidia_uvm
sudo modprobe nvidia
sudo insmod "${CUSTOM_UVM}" \
  uvm_moe_slot_pte_remap_enable=1 \
  uvm_moe_slot_pool_vidmem_budget_bytes="${POOL_BUDGET_BYTES}" \
  uvm_moe_slot_plan_overlap_copy_prepare=0 \
  uvm_moe_slot_plan_register_pte_prewarm=1 \
  uvm_moe_slot_plan_cache_copy_runs=1 \
  uvm_moe_slot_plan_cache_metadata_recipe=1 \
  uvm_moe_slot_transition_fast_index=1 \
  uvm_moe_slot_transition_direct_cutover=1 \
  uvm_moe_slot_plan_fast_validation=1 \
  uvm_moe_slot_plan_epoch_validation="${EPOCH_VALIDATION}" \
  uvm_moe_slot_plan_shadow_pte_prefill=1 \
  uvm_moe_slot_plan_dual_bank_event=1 \
  uvm_moe_slot_plan_cuda_fence=1

custom_src="$(modinfo -F srcversion "${CUSTOM_UVM}")"
loaded_src="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)"
[[ -n "${custom_src}" && "${loaded_src}" == "${custom_src}" ]]
echo "custom_load: srcversion=${loaded_src}"

ALLOW_GPU_RUN=1 ORDER="${ORDER:-PCU}" RUN_ROOT="${RUN_ROOT}" \
  EPOCH_VALIDATION="${EPOCH_VALIDATION}" \
  BANK_LEVEL_COMMIT="${BANK_LEVEL_COMMIT}" \
  TRACE_MODE="${TRACE_MODE}" \
  "${RUNNER_SCRIPT}" run

restore_stock
echo "completed_at=$(date --iso-8601=seconds)"
