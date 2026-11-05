#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-smoke}"
DEVICE="${DEVICE:-0}"
CUDA_ARCH="${CUDA_ARCH:-native}"
RESULT_FILE="${RESULT_FILE:-/tmp/uvm_transfer_microbench_$(date +%Y%m%d_%H%M%S).txt}"

case "${PROFILE}" in
  smoke)
    COLD_MIB="${COLD_MIB:-64}"
    TENSORS="${TENSORS:-16}"
    SLOTS="${SLOTS:-2}"
    WARMUPS="${WARMUPS:-1}"
    REPEATS="${REPEATS:-3}"
    ;;
  full)
    COLD_MIB="${COLD_MIB:-512}"
    TENSORS="${TENSORS:-128}"
    SLOTS="${SLOTS:-2}"
    WARMUPS="${WARMUPS:-2}"
    REPEATS="${REPEATS:-7}"
    ;;
  *)
    echo "PROFILE must be smoke or full: ${PROFILE}" >&2
    exit 2
    ;;
esac

EQUIVALENT_RATIO="${EQUIVALENT_RATIO:-0.90}"
ONLY_LAYOUT="${ONLY_LAYOUT:-}"
ONLY_BACKEND="${ONLY_BACKEND:-}"
ONLY_MODE="${ONLY_MODE:-}"

command -v nvcc >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi --id="${DEVICE}" --query-gpu=name,driver_version,memory.used,memory.free \
  --format=csv,noheader

make -C "${SCRIPT_DIR}" CUDA_ARCH="${CUDA_ARCH}"
mkdir -p "$(dirname "${RESULT_FILE}")"
benchmark_args=(
  --device "${DEVICE}" \
  --cold-mib "${COLD_MIB}" \
  --tensors "${TENSORS}" \
  --slots "${SLOTS}" \
  --warmups "${WARMUPS}" \
  --repeats "${REPEATS}" \
  --equivalent-ratio "${EQUIVALENT_RATIO}"
)
[[ -n "${ONLY_LAYOUT}" ]] && benchmark_args+=(--only-layout "${ONLY_LAYOUT}")
[[ -n "${ONLY_BACKEND}" ]] && benchmark_args+=(--only-backend "${ONLY_BACKEND}")
[[ -n "${ONLY_MODE}" ]] && benchmark_args+=(--only-mode "${ONLY_MODE}")

"${SCRIPT_DIR}/uvm_transfer_microbench" "${benchmark_args[@]}" | tee "${RESULT_FILE}"

grep -q '^benchmark_result functional_success=1 ' "${RESULT_FILE}"
if [[ -z "${ONLY_LAYOUT}${ONLY_BACKEND}${ONLY_MODE}" ]]; then
  grep -q '^RESULT layout=multi_alloc backend=pinned_copy mode=tensor_batch ' "${RESULT_FILE}"
  grep -q '^RESULT layout=multi_alloc backend=managed_prefetch mode=tensor_batch ' "${RESULT_FILE}"
  grep -q '^RESULT layout=arena_contiguous backend=managed_prefetch mode=coalesced_batch ' "${RESULT_FILE}"
  grep -q '^RESULT layout=arena_interleaved backend=uva_direct mode=direct ' "${RESULT_FILE}"
  grep -q '^DIAGNOSIS ' "${RESULT_FILE}"
else
  grep -q '^RESULT ' "${RESULT_FILE}"
  grep -q '^benchmark_result functional_success=1 filtered=1 ' "${RESULT_FILE}"
fi

echo "RESULT_FILE=${RESULT_FILE}"
