#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROBE_DIR="${REPO_ROOT}/workloads/vllm/vllm/uvm_test"
PROBE="${PROBE_DIR}/uvm_async_layer_prefetch_probe"
UVM_MODULE="${UVM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"

DEVICE="${DEVICE:-0}"
LAYERS="${LAYERS:-8}"
LAYER_MIB="${LAYER_MIB:-256}"
COMPUTE_MS="${COMPUTE_MS:-12}"
REPEATS="${REPEATS:-5}"
CUDA_ARCH="${CUDA_ARCH:-sm_120}"
RESULT_FILE="${RESULT_FILE:-${SCRIPT_DIR}/result_$(date +%Y%m%d_%H%M%S).txt}"

loaded_srcversion="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
module_srcversion="$(modinfo -F srcversion "${UVM_MODULE}")"
if [[ -z "${loaded_srcversion}" || "${loaded_srcversion}" != "${module_srcversion}" ]]; then
  echo "nvidia_uvm mismatch: loaded=${loaded_srcversion:-missing} module=${module_srcversion:-missing}" >&2
  exit 1
fi

used_mib="$(nvidia-smi --id="${DEVICE}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${used_mib}" =~ ^[0-9]+$ ]] || (( used_mib > 128 )); then
  echo "GPU ${DEVICE} is not clean: ${used_mib:-unknown} MiB used" >&2
  exit 1
fi

make -C "${PROBE_DIR}" CUDA_ARCH="${CUDA_ARCH}" uvm_async_layer_prefetch_probe
mkdir -p "$(dirname "${RESULT_FILE}")"
"${PROBE}" \
  --device "${DEVICE}" \
  --layers "${LAYERS}" \
  --layer-mib "${LAYER_MIB}" \
  --compute-ms "${COMPUTE_MS}" \
  --repeats "${REPEATS}" | tee "${RESULT_FILE}"

echo "RESULT_FILE=${RESULT_FILE}"
