#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
CUSTOM_MODULE="${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko"
PROFILE="${PROFILE:-smoke}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${PROFILE}}"

case "${PROFILE}" in
  smoke)
    REGIONS="${REGIONS:-8}"
    TOKENS="${TOKENS:-32}"
    WARMUPS="${WARMUPS:-2}"
    SAMPLES="${SAMPLES:-8}"
    ;;
  full)
    REGIONS="${REGIONS:-128}"
    TOKENS="${TOKENS:-256}"
    WARMUPS="${WARMUPS:-5}"
    SAMPLES="${SAMPLES:-20}"
    ;;
  *)
    echo "unknown PROFILE=${PROFILE}; expected smoke or full" >&2
    exit 2
    ;;
esac

loaded="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
expected="$(modinfo -F srcversion "${CUSTOM_MODULE}" 2>/dev/null || true)"
if [[ -z "${loaded}" || "${loaded}" != "${expected}" ]]; then
  echo "custom UVM is not loaded: loaded=${loaded:-missing} expected=${expected:-missing}" >&2
  exit 1
fi
if [[ "$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 2>/dev/null || true)" != "1" ]]; then
  echo "uvm_moe_slot_pte_remap_enable must be 1" >&2
  exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')" ]]; then
  echo "GPU has an active compute process" >&2
  exit 1
fi

exec 9>"/tmp/vllm_slot_direct_e2e.lock"
if ! flock -n 9; then
  echo "another vllm_slot_direct_e2e run holds the experiment lock" >&2
  exit 1
fi

make -C "${SCRIPT_DIR}" CUDA_ARCH=native
mkdir -p "${RUN_ROOT}"
RESULT="${RUN_ROOT}/result.json"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH="${REPO_ROOT}/workloads/vllm/vllm${PYTHONPATH:+:${PYTHONPATH}}" \
  "${VLLM_PY}" "${SCRIPT_DIR}/slot_torch_e2e.py" \
    --library "${SCRIPT_DIR}/libvllm_slot_direct_e2e.so" \
    --output "${RESULT}" \
    --regions "${REGIONS}" \
    --tokens "${TOKENS}" \
    --warmup-forwards "${WARMUPS}" \
    --measured-forwards "${SAMPLES}" \
    > "${RUN_ROOT}/run.log" 2>&1

"${VLLM_PY}" "${SCRIPT_DIR}/verify_result.py" "${RESULT}" | tee "${RUN_ROOT}/verify.txt"
echo "RUN_ROOT=${RUN_ROOT}"
