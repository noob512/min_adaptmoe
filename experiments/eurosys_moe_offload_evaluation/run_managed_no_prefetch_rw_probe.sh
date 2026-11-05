#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE="${SCRIPT_DIR}/managed_no_prefetch_rw_probe.cu"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/managed_no_prefetch_rw_$(date -u +%Y%m%dT%H%M%SZ)}"
BINARY="${RUN_ROOT}/managed_no_prefetch_rw_probe"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 2
}
[[ -f "${SOURCE}" ]] || { echo "missing source: ${SOURCE}" >&2; exit 2; }
[[ -f "${BASE_HARNESS}" ]] || { echo "missing harness: ${BASE_HARNESS}" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc unavailable" >&2; exit 2; }

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ALLOW_GPU_RUN=1
export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1207959552}"
# shellcheck source=/dev/null
source "${BASE_HARNESS}"

runtime_preflight
RUN_ROOT_ACTIVE="${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"

/usr/local/cuda-12.8/bin/nvcc -O2 -std=c++17 -lineinfo \
  "${SOURCE}" -o "${BINARY}" \
  >"${RUN_ROOT}/build.stdout" 2>"${RUN_ROOT}/build.stderr"

KERNEL_EPOCH="$(date +%s)"
load_custom_module
echo 1 | sudo tee \
  /sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_enable >/dev/null
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"

printf '%q ' timeout --signal=TERM --kill-after=5s 30s \
  "${BINARY}" --device "${GPU_INDEX}" --mib "${PROBE_MIB:-64}" \
  >"${RUN_ROOT}/command.txt"
printf '\n' >>"${RUN_ROOT}/command.txt"

set +e
timeout --signal=TERM --kill-after=5s 30s \
  "${BINARY}" --device "${GPU_INDEX}" --mib "${PROBE_MIB:-64}" \
  >"${RUN_ROOT}/probe.stdout" 2>"${RUN_ROOT}/probe.stderr"
PROBE_RC=$?
set -e
printf '%s\n' "${PROBE_RC}" >"${RUN_ROOT}/probe.returncode.txt"

# Evidence capture and recovery are not additional experiment attempts.
sudo cat /proc/driver/nvidia-uvm/phase_f_route_log \
  >"${RUN_ROOT}/phase_f_route_log.txt" 2>"${RUN_ROOT}/phase_f_route_log.err" \
  || true
sudo journalctl -k --since "@${KERNEL_EPOCH}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err" || true
gpu_state_json "${RUN_ROOT}/gpu.before_recovery.json" || true

PROBE_STATUS="failed"
(( PROBE_RC == 0 )) && PROBE_STATUS="passed"
"${VLLM_PY}" - "${RUN_ROOT}/result.json" "${PROBE_STATUS}" \
  "${PROBE_RC}" "${PROBE_MIB:-64}" <<'PY'
import json, sys, time
path, status, rc, mib = sys.argv[1:]
payload = {
    "schema": "managed_no_prefetch_rw_probe_v1",
    "timestamp": time.time(),
    "status": status,
    "probe_returncode": int(rc),
    "allocation_bytes": int(mib) * 1024 * 1024,
    "manual_migration_fault_mode": 2,
    "prefetch_used": False,
    "single_attempt": True,
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

(( PROBE_RC == 0 ))
