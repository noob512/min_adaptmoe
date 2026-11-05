#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CUSTOM_UVM="${CUSTOM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
PYTHON="${PYTHON:-${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/result_$(date -u +%Y%m%dT%H%M%SZ).json}"
PROBE="${PROBE:-${SCRIPT_DIR}/probe.py}"
ROUTE_LOG_OUTPUT="${ROUTE_LOG_OUTPUT:-}"
SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-$((64 * 1024 * 1024))}"
PROBE_ARGS=()
if [[ -n "${REGIONS:-}" ]]; then
  PROBE_ARGS+=(--regions "${REGIONS}")
fi
if [[ -n "${LAYERS:-}" ]]; then
  PROBE_ARGS+=(--layers "${LAYERS}")
fi
STOCK_UVM="$(modinfo -F filename nvidia-uvm)"
STOCK_SRCVERSION="$(modinfo -F srcversion nvidia-uvm)"
NODE_WAS_ACTIVE="$(systemctl is-active node-exporter.service || true)"
GPU_WAS_ACTIVE="$(systemctl is-active gpu-exporter.service || true)"
RESTORED=0

gpu_clean() {
  local used pids
  if ! used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"; then
    return 1
  fi
  if ! pids="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | \
      awk -F',' '
        {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        }
        $1 ~ /^[0-9]+$/ && $2 !~ /(^|\/)nvidia-smi$/ {print $1}
      ')"; then
    return 1
  fi
  [[ "${used}" == "0" && -z "${pids}" ]]
}

loaded_srcversion() {
  local path=/sys/module/nvidia_uvm/srcversion
  [[ -r "${path}" ]] || return 0
  tr -d '[:space:]' <"${path}"
}

restore_stock() {
  local rc=0 loaded
  [[ "${RESTORED}" == 0 ]] || return 0
  set +e
  loaded="$(loaded_srcversion)"
  if ! gpu_clean; then
    echo "recovery: GPU not clean; refusing UVM unload" >&2
    rc=1
  elif [[ "${loaded}" != "${STOCK_SRCVERSION}" ]] && sudo fuser /dev/nvidia-uvm >/dev/null 2>&1; then
    echo "recovery: /dev/nvidia-uvm still has a client" >&2
    rc=1
  elif [[ "${loaded}" != "${STOCK_SRCVERSION}" ]]; then
    [[ -z "${loaded}" ]] || sudo modprobe -r nvidia_uvm || rc=1
    sudo modprobe nvidia || rc=1
    sudo modprobe nvidia_uvm || rc=1
  fi
  [[ "${NODE_WAS_ACTIVE}" == "active" ]] && sudo systemctl start node-exporter.service || true
  [[ "${GPU_WAS_ACTIVE}" == "active" ]] && sudo systemctl start gpu-exporter.service || true
  loaded="$(loaded_srcversion)"
  [[ "${loaded}" == "${STOCK_SRCVERSION}" ]] || rc=1
  [[ "$(readlink -f "$(modinfo -F filename nvidia-uvm)")" == "$(readlink -f "${STOCK_UVM}")" ]] || rc=1
  gpu_clean || rc=1
  (( rc == 0 )) && RESTORED=1
  set -e
  return "${rc}"
}

capture_route_log() {
  [[ -n "${ROUTE_LOG_OUTPUT}" ]] || return 0
  [[ -r /proc/driver/nvidia-uvm/phase_f_route_log ]] || return 0
  sudo cp -- /proc/driver/nvidia-uvm/phase_f_route_log \
    "${ROUTE_LOG_OUTPUT}"
  sudo chown "$(id -u):$(id -g)" "${ROUTE_LOG_OUTPUT}"
}

on_exit() {
  local original_rc="$1" recovery_rc=0
  trap - EXIT INT TERM
  capture_route_log || recovery_rc=$?
  restore_stock || recovery_rc=$?
  (( original_rc == 0 )) || exit "${original_rc}"
  exit "${recovery_rc}"
}

trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -f "${CUSTOM_UVM}" && -x "${PYTHON}" && -f "${PROBE}" ]]
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
  uvm_moe_slot_pool_vidmem_budget_bytes="${SLOT_BUDGET_BYTES}"
if [[ -n "${ROUTE_LOG_OUTPUT}" ]]; then
  printf '1\n' | sudo tee \
    /sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_enable \
    >/dev/null
fi

custom_src="$(modinfo -F srcversion "${CUSTOM_UVM}")"
loaded_src="$(loaded_srcversion)"
[[ -n "${custom_src}" && "${loaded_src}" == "${custom_src}" ]]

cd "${REPO_ROOT}/workloads/vllm/vllm"
PYTHONPATH="${REPO_ROOT}/workloads/vllm/vllm${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" "${PROBE}" --replays "${REPLAYS:-200}" --output "${OUTPUT}" \
  "${PROBE_ARGS[@]}"

capture_route_log
restore_stock
echo "result=${OUTPUT}"
