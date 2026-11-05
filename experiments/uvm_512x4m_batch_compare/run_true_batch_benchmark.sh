#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-full}"
DEVICE="${DEVICE:-0}"
CUDA_ARCH="${CUDA_ARCH:-native}"
STATS="${STATS:-/proc/driver/nvidia-uvm/latency_stats}"

case "${PROFILE}" in
  full)
    ALLOCATIONS="${ALLOCATIONS:-512}"
    ALLOCATION_MIB="${ALLOCATION_MIB:-4}"
    GUARD_MIB="${GUARD_MIB:-4}"
    BATCH_SIZE="${BATCH_SIZE:-128}"
    WARMUP_PER_MODE="${WARMUP_PER_MODE:-2}"
    MEASURED_PER_MODE="${MEASURED_PER_MODE:-6}"
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

RESULT_FILE="${RESULT_FILE:-${SCRIPT_DIR}/RESULTS_TRUE_BATCH_${PROFILE^^}_20260815.txt}"

command -v nvcc >/dev/null
command -v python3 >/dev/null
command -v nvidia-smi >/dev/null
if [[ ! -r "${STATS}" ]]; then
  echo "latency stats are unavailable: ${STATS}" >&2
  exit 1
fi
nvidia-smi --id="${DEVICE}" \
  --query-gpu=name,driver_version,memory.used,memory.free,temperature.gpu,clocks.current.sm \
  --format=csv,noheader

make -C "${SCRIPT_DIR}" CUDA_ARCH="${CUDA_ARCH}"
mkdir -p "$(dirname "${RESULT_FILE}")"
PYTHONUNBUFFERED=1 python3 "${SCRIPT_DIR}/true_batch_benchmark.py" \
  --library "${SCRIPT_DIR}/libuvm_batch_shim.so" \
  --device "${DEVICE}" \
  --allocations "${ALLOCATIONS}" \
  --allocation-mib "${ALLOCATION_MIB}" \
  --guard-mib "${GUARD_MIB}" \
  --batch-size "${BATCH_SIZE}" \
  --warmup-per-mode "${WARMUP_PER_MODE}" \
  --measured-per-mode "${MEASURED_PER_MODE}" \
  --stats "${STATS}" | tee "${RESULT_FILE}"

VERIFY_ARGS=("${RESULT_FILE}")
if [[ "${PROFILE}" == "full" ]]; then
  VERIFY_ARGS+=(--require-full)
fi
python3 "${SCRIPT_DIR}/verify_true_batch_results.py" "${VERIFY_ARGS[@]}"

grep -q '^DRIVER_ABI ioctl_command=90 max_ranges=128 .*synchronous=1 completion_on_return=1$' "${RESULT_FILE}"
grep -q '^RESULT mode=native_async .*profile_contract_ok=1$' "${RESULT_FILE}"
grep -q '^RESULT mode=native_4x128_sync .*profile_contract_ok=1$' "${RESULT_FILE}"
grep -q '^RESULT mode=driver_4x128_sync .*profile_contract_ok=1$' "${RESULT_FILE}"
grep -q '^benchmark_result functional_success=1 .*driver_ioctl_command=90 driver_max_ranges=128 .*profile_contract_ok=1 ' "${RESULT_FILE}"

echo "RESULT_FILE=${RESULT_FILE}"
