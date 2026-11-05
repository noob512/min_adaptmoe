#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-full}"
DEVICE="${DEVICE:-0}"
CUDA_ARCH="${CUDA_ARCH:-native}"

case "${PROFILE}" in
  full)
    ALLOCATIONS="${ALLOCATIONS:-512}"
    ALLOCATION_MIB="${ALLOCATION_MIB:-4}"
    GUARD_MIB="${GUARD_MIB:-4}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    WARMUP_PER_MODE="${WARMUP_PER_MODE:-2}"
    MEASURED_PER_MODE="${MEASURED_PER_MODE:-8}"
    ;;
  smoke)
    ALLOCATIONS="${ALLOCATIONS:-16}"
    ALLOCATION_MIB="${ALLOCATION_MIB:-1}"
    GUARD_MIB="${GUARD_MIB:-1}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    WARMUP_PER_MODE="${WARMUP_PER_MODE:-2}"
    MEASURED_PER_MODE="${MEASURED_PER_MODE:-2}"
    ;;
  *)
    echo "PROFILE must be full or smoke: ${PROFILE}" >&2
    exit 2
    ;;
esac

RESULT_FILE="${RESULT_FILE:-${SCRIPT_DIR}/RESULTS_${PROFILE^^}_20260815.txt}"

command -v nvcc >/dev/null
command -v python3 >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi --id="${DEVICE}" \
  --query-gpu=name,driver_version,memory.used,memory.free,temperature.gpu,clocks.current.sm \
  --format=csv,noheader

make -C "${SCRIPT_DIR}" CUDA_ARCH="${CUDA_ARCH}"
mkdir -p "$(dirname "${RESULT_FILE}")"
PYTHONUNBUFFERED=1 python3 "${SCRIPT_DIR}/benchmark.py" \
  --library "${SCRIPT_DIR}/libuvm_batch_shim.so" \
  --device "${DEVICE}" \
  --allocations "${ALLOCATIONS}" \
  --allocation-mib "${ALLOCATION_MIB}" \
  --guard-mib "${GUARD_MIB}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup-per-mode "${WARMUP_PER_MODE}" \
  --measured-per-mode "${MEASURED_PER_MODE}" | tee "${RESULT_FILE}"

grep -q '^LAYOUT .*adjacent_test_ranges=0 ' "${RESULT_FILE}"
grep -q "^RESULT mode=direct .*outer_calls_per_sample=${ALLOCATIONS} .*runtime_prefetch_calls_per_sample=${ALLOCATIONS} " "${RESULT_FILE}"
grep -q "^RESULT mode=batch .*outer_calls_per_sample=4 .*runtime_prefetch_calls_per_sample=${ALLOCATIONS} " "${RESULT_FILE}"
grep -q '^COMPARISON .*runtime_call_reduction=1.000 ' "${RESULT_FILE}"
grep -q "^benchmark_result functional_success=1 allocations=${ALLOCATIONS} .*direct_outer_calls=${ALLOCATIONS} batch_outer_calls=4 runtime_calls_both=${ALLOCATIONS} accepted_both=${ALLOCATIONS} adjacent_test_ranges=0 " "${RESULT_FILE}"

echo "RESULT_FILE=${RESULT_FILE}"
