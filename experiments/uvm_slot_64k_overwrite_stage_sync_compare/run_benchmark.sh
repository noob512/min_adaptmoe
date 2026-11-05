#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-full}"
DEVICE="${DEVICE:-0}"
CUDA_ARCH="${CUDA_ARCH:-native}"
STATS="${STATS:-/proc/driver/nvidia-uvm/latency_stats}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SCRIPT_DIR}/artifacts}"

if [[ "${PROFILE}" != "smoke" && "${PROFILE}" != "full" ]]; then
  echo "PROFILE must be smoke or full: ${PROFILE}" >&2
  exit 2
fi

command -v nvcc >/dev/null
command -v python3 >/dev/null
make -C "${SCRIPT_DIR}" CUDA_ARCH="${CUDA_ARCH}"

args=(
  --profile "${PROFILE}"
  --device "${DEVICE}"
  --stats "${STATS}"
  --artifact-root "${ARTIFACT_ROOT}"
)
[[ -n "${SCHEDULE_BLOCKS:-}" ]] && args+=(--schedule-blocks "${SCHEDULE_BLOCKS}")
[[ -n "${WARMUPS:-}" ]] && args+=(--warmups "${WARMUPS}")
[[ -n "${TIMEOUT_SECONDS:-}" ]] && args+=(--timeout-seconds "${TIMEOUT_SECONDS}")

run_log="$(mktemp /tmp/uvm_slot_64k_overwrite_stage_sync_run.XXXXXX)"
cleanup() {
  rm -f "${run_log}"
}
trap cleanup EXIT

PYTHONUNBUFFERED=1 python3 "${SCRIPT_DIR}/run_benchmark.py" "${args[@]}" | tee "${run_log}"
artifact_dir="$(sed -n 's/^ARTIFACT_DIR=//p' "${run_log}" | tail -n 1)"
if [[ -z "${artifact_dir}" || ! -f "${artifact_dir}/summary.json" ]]; then
  echo "runner did not produce a complete artifact directory" >&2
  exit 1
fi

verify_args=("${artifact_dir}")
if [[ "${PROFILE}" == "full" ]]; then
  verify_args+=(--require-full)
fi
python3 "${SCRIPT_DIR}/verify_results.py" "${verify_args[@]}"
echo "VERIFIED_ARTIFACT_DIR=${artifact_dir}"
