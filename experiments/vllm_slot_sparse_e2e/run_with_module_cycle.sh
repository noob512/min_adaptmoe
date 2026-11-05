#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUSTOM_UVM="${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_sparse48x12_fence_v1}"
CYCLE_LOG="${RUN_ROOT}.module_cycle.log"
EXPECTED_STOCK_SRCVERSION="${EXPECTED_STOCK_SRCVERSION:-85A79790636BBD99BA3E43B}"
SHADOW_PTE_PREFILL="${SHADOW_PTE_PREFILL:-1}"
STOCK_UVM="$(modinfo -F filename nvidia-uvm)"
NODE_WAS_ACTIVE="$(systemctl is-active node-exporter.service || true)"
GPU_WAS_ACTIVE="$(systemctl is-active gpu-exporter.service || true)"
RECOVERY_DONE=0

case "${SHADOW_PTE_PREFILL}" in
  0|1) ;;
  *) echo "error: SHADOW_PTE_PREFILL must be 0 or 1, got ${SHADOW_PTE_PREFILL}" >&2; exit 1 ;;
esac

gpu_clean() {
  local used pids
  used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
  [[ "${used}" == "0" && -z "${pids}" ]]
}

restore_stock() {
  local rc=0 loaded attempt
  [[ "${RECOVERY_DONE}" == 0 ]] || return 0
  RECOVERY_DONE=1
  set +e
  for ((attempt = 0; attempt < 180; ++attempt)); do
    gpu_clean && break
    sleep 1
  done
  loaded="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion 2>/dev/null)"
  if ! gpu_clean; then
    echo "recovery: GPU is not clean; refusing unsafe UVM unload" >&2
    rc=1
  elif [[ "${loaded}" != "${EXPECTED_STOCK_SRCVERSION}" ]] && \
       sudo fuser /dev/nvidia-uvm >/dev/null 2>&1; then
    echo "recovery: /dev/nvidia-uvm still has a client; refusing unload" >&2
    rc=1
  else
    if [[ "${loaded}" != "${EXPECTED_STOCK_SRCVERSION}" ]]; then
      sudo modprobe -r nvidia_uvm || rc=1
      sudo modprobe nvidia || rc=1
      sudo modprobe nvidia_uvm || rc=1
    fi
  fi
  [[ "${NODE_WAS_ACTIVE}" == "active" ]] && sudo systemctl start node-exporter.service || true
  [[ "${GPU_WAS_ACTIVE}" == "active" ]] && sudo systemctl start gpu-exporter.service || true
  loaded="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion 2>/dev/null)"
  [[ "${loaded}" == "${EXPECTED_STOCK_SRCVERSION}" ]] || rc=1
  [[ "$(readlink -f "$(modinfo -F filename nvidia-uvm)")" == "$(readlink -f "${STOCK_UVM}")" ]] || rc=1
  gpu_clean || rc=1
  [[ "${NODE_WAS_ACTIVE}" != "active" ]] || systemctl is-active --quiet node-exporter.service || rc=1
  [[ "${GPU_WAS_ACTIVE}" != "active" ]] || systemctl is-active --quiet gpu-exporter.service || rc=1
  echo "recovery: loaded_srcversion=${loaded} gpu_clean=$(gpu_clean && echo yes || echo no) rc=${rc}"
  set -e
  return "${rc}"
}

on_exit() {
  local original_rc="$1"
  local recovery_rc=0
  trap - EXIT INT TERM
  restore_stock || recovery_rc=$?
  if [[ -d "${RUN_ROOT}" && -f "${CYCLE_LOG}" ]]; then
    mv -- "${CYCLE_LOG}" "${RUN_ROOT}/module_cycle.log"
  fi
  if (( original_rc != 0 )); then
    exit "${original_rc}"
  fi
  exit "${recovery_rc}"
}

trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

exec > >(tee -a "${CYCLE_LOG}") 2>&1
echo "started_at=$(date --iso-8601=seconds)"
echo "stock_module=${STOCK_UVM}"
echo "stock_srcversion=${EXPECTED_STOCK_SRCVERSION}"
echo "custom_module=${CUSTOM_UVM}"
echo "custom_srcversion=$(modinfo -F srcversion "${CUSTOM_UVM}")"
echo "run_root=${RUN_ROOT}"
[[ -f "${CUSTOM_UVM}" ]]
[[ ! -e "${RUN_ROOT}" ]]
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
  uvm_moe_slot_pool_vidmem_budget_bytes=113246208 \
  uvm_moe_slot_plan_overlap_copy_prepare=0 \
  uvm_moe_slot_plan_register_pte_prewarm=1 \
  uvm_moe_slot_plan_cache_copy_runs=1 \
  uvm_moe_slot_plan_cache_metadata_recipe=1 \
  uvm_moe_slot_transition_fast_index=1 \
  uvm_moe_slot_transition_direct_cutover=1 \
  uvm_moe_slot_plan_fast_validation=1 \
  uvm_moe_slot_plan_shadow_pte_prefill="${SHADOW_PTE_PREFILL}" \
  uvm_moe_slot_plan_dual_bank_event=1 \
  uvm_moe_slot_plan_cuda_fence=1

custom_src="$(modinfo -F srcversion "${CUSTOM_UVM}")"
loaded_src="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)"
[[ -n "${custom_src}" && "${loaded_src}" == "${custom_src}" ]]
echo "custom_load: srcversion=${loaded_src}"

ALLOW_GPU_RUN=1 \
SHADOW_PTE_PREFILL="${SHADOW_PTE_PREFILL}" \
RUN_ROOT="${RUN_ROOT}" \
  "${SCRIPT_DIR}/run_case.sh" run

restore_stock
echo "completed_at=$(date --iso-8601=seconds)"
