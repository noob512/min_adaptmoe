#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
DIRECT_RUNNER="${SCRIPT_DIR}/direct_batch_phase_runner.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
ENV_FILE="${SCRIPT_DIR}/configs/e2_ours_2g_env.json"
MODEL="/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16"
EVAL_FIXTURE="${FIXTURE_ROOT}/evaluation_short_dialogue.json"
WARMUP_FIXTURE="${FIXTURE_ROOT}/calibration_short_dialogue.json"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e2-direct-batch-pure-phase-fence-off-$(date -u +%Y%m%dT%H%M%SZ)}"
E2_CONCURRENCIES="${E2_CONCURRENCIES:-1,4,8,16}"

[[ ! -e "${RUN_ROOT}" ]] || { echo "error: RUN_ROOT exists: ${RUN_ROOT}" >&2; exit 1; }
for required in "${DIRECT_RUNNER}" "${BASE_HARNESS}" "${ENV_FILE}" \
  "${MODEL}/config.json" "${EVAL_FIXTURE}" "${WARMUP_FIXTURE}"; do
  [[ -f "${required}" ]] || { echo "error: missing file: ${required}" >&2; exit 1; }
done
command -v jq >/dev/null
command -v flock >/dev/null

# Serialize with formal E2 as well as the lower-level phase-switch harness.
exec 8>"/tmp/eurosys_moe_offload_e2_gpu0.lock"
flock -n 8 || { echo "error: another E2 runner holds the GPU lock" >&2; exit 1; }

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ALLOW_GPU_RUN=1
export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1207959552}"
# shellcheck source=/dev/null
source "${BASE_HARNESS}"

mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"

# Load the frozen cold11 configuration without evaluating shell source text.
while IFS=$'\t' read -r key value; do
  [[ "${key}" =~ ^[A-Z_][A-Z0-9_]*$ ]] \
    || { echo "error: invalid environment key: ${key}" >&2; exit 1; }
  export "${key}=${value}"
done < <(jq -r 'to_entries[] | [.key, .value] | @tsv' "${ENV_FILE}")

export VLLM_CPU_OFFLOAD_MANIFEST_FILE="${RUN_ROOT}/offload_manifest.jsonl"
export VLLM_UVM_LOG_FILE="${RUN_ROOT}/uvm.log"
export VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE="${RUN_ROOT}/phase_transition.json"
if [[ "${E2_CUDA_ADDRESS_DIAGNOSTIC:-0}" == 1 ]]; then
  export E2_CUDA_ADDRESS_DIAGNOSTIC_FILE="${RUN_ROOT}/cuda_address_snapshots.json"
else
  unset E2_CUDA_ADDRESS_DIAGNOSTIC_FILE
fi

# Fence-off is a hard precondition for this requested qualification.  Remove
# both the runtime variable and the formal launcher's spelling inherited from
# any parent shell.
unset VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE
unset FORMAL_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE

jq --arg manifest "${VLLM_CPU_OFFLOAD_MANIFEST_FILE}" \
   --arg uvm_log "${VLLM_UVM_LOG_FILE}" \
   --arg phase_result "${VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE}" \
   --arg parameter_mapping_rebuild "${VLLM_UVM_REBUILD_PARAMETER_GPU_READ_MAPPINGS:-default_false}" \
   --arg cuda_address_diagnostic "${E2_CUDA_ADDRESS_DIAGNOSTIC:-0}" \
   --arg cuda_address_diagnostic_mode "${E2_CUDA_ADDRESS_DIAGNOSTIC_MODE:-full}" \
  '. + {
    VLLM_CPU_OFFLOAD_MANIFEST_FILE: $manifest,
    VLLM_UVM_LOG_FILE: $uvm_log,
    VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE: $phase_result,
    VLLM_UVM_REBUILD_PARAMETER_GPU_READ_MAPPINGS: $parameter_mapping_rebuild,
    E2_CUDA_ADDRESS_DIAGNOSTIC: $cuda_address_diagnostic,
    E2_CUDA_ADDRESS_DIAGNOSTIC_MODE: $cuda_address_diagnostic_mode
  } | del(.VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE)' \
  "${ENV_FILE}" >"${RUN_ROOT}/environment.json"

git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/git_status.txt"
sha256sum "${DIRECT_RUNNER}" "${BASH_SOURCE[0]}" "${ENV_FILE}" \
  "${EVAL_FIXTURE}" "${WARMUP_FIXTURE}" "${MODEL}/config.json" \
  "${CUSTOM_UVM}" "${STOCK_UVM}" >"${RUN_ROOT}/input_sha256.txt"
gpu_state_json "${RUN_ROOT}/gpu.before.json"
cp -- "${STATS_PATH}" "${RUN_ROOT}/latency_stats.before.txt"

runner_args=(
  --model "${MODEL}"
  --fixture "${EVAL_FIXTURE}"
  --warmup-fixture "${WARMUP_FIXTURE}"
  --output "${RUN_ROOT}/result.json"
  --concurrencies "${E2_CONCURRENCIES}"
  --warmup-requests 3
  --warmup-output-tokens 8
  --output-tokens 32
  --seed 20260823
  --kv-cache-memory-bytes 2147483648
)
{
  printf 'cd %q && env -u VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE ' "${VLLM_ROOT}"
  printf 'timeout --kill-after=30s 1800s taskset -c 0-13 %q %q ' "${VLLM_PY}" "${DIRECT_RUNNER}"
  printf '%q ' "${runner_args[@]}"
  printf '\n'
} >"${RUN_ROOT}/command.txt"

sampler_fields='timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.current.sm,clocks.current.memory,pstate'
nvidia-smi --query-gpu="${sampler_fields}" --format=csv,noheader,nounits \
  --loop-ms=200 >"${RUN_ROOT}/nvml.csv" 2>"${RUN_ROOT}/nvml.err" &
sampler_pid=$!
kernel_epoch="$(date +%s)"

set +e
(
  cd "${VLLM_ROOT}"
  env -u VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE \
    /usr/bin/time -v -o "${RUN_ROOT}/resource_time.txt" \
    timeout --kill-after=30s 1800s taskset -c 0-13 \
    "${VLLM_PY}" "${DIRECT_RUNNER}" "${runner_args[@]}"
) >"${RUN_ROOT}/run.log" 2>&1
runner_rc=$?
kill "${sampler_pid}" 2>/dev/null
wait "${sampler_pid}" 2>/dev/null
sampler_rc=$?
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err"
journal_rc=$?
set -e

printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/returncode.txt"
printf '%s\n' "${sampler_rc}" >"${RUN_ROOT}/nvml_returncode.txt"
printf '%s\n' "${journal_rc}" >"${RUN_ROOT}/kernel_log_returncode.txt"
cp -- "${STATS_PATH}" "${RUN_ROOT}/latency_stats.after.txt"

wait_gpu_clean || { gpu_state_json "${RUN_ROOT}/gpu.after.json"; exit 96; }
gpu_state_json "${RUN_ROOT}/gpu.after.json"
restore_stock
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters

"${VLLM_PY}" - "${RUN_ROOT}" "${runner_rc}" "${journal_rc}" "${E2_CONCURRENCIES}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
runner_rc = int(sys.argv[2])
journal_rc = int(sys.argv[3])
expected_concurrencies = [int(value) for value in sys.argv[4].split(",")]
result_path = root / "result.json"
result = json.loads(result_path.read_text()) if result_path.is_file() else None
log = (root / "run.log").read_text(encoding="utf-8", errors="replace")
kernel_log = (root / "kernel.since.log").read_text(encoding="utf-8", errors="replace")
fatal_patterns = {
    "cuda_error": r"CUDA error",
    "illegal_access": r"illegal memory access",
    "xid": r"\bXid\b",
    "pte_fault": r"PTE.*fault|PDE.*fault|FAULT_PDE|FAULT_PTE",
    "traceback": r"Traceback \(most recent call last\)",
}
fatal = {
    name: bool(re.search(pattern, log + "\n" + kernel_log, re.IGNORECASE))
    for name, pattern in fatal_patterns.items()
}
cells = result.get("cells", []) if isinstance(result, dict) else []
checks = {
    "runner_returncode_zero": runner_rc == 0,
    "kernel_log_returncode_zero": journal_rc == 0,
    "result_status_ok": isinstance(result, dict) and result.get("status") == "ok",
    "fence_absent": isinstance(result, dict) and not result["configuration"][
        "runtime_post_resume_cold_read_gate_present"
    ],
    "no_client": isinstance(result, dict)
    and result["configuration"].get("client_used") is False,
    "no_http_server": isinstance(result, dict)
    and result["configuration"].get("http_server_started") is False,
    "requested_cells": [cell.get("concurrency") for cell in cells]
    == expected_concurrencies,
    "all_requests_succeeded": bool(cells)
    and all(cell.get("successful_requests") == cell.get("request_count") for cell in cells),
    "all_cells_single_first_step_batch": bool(cells)
    and all(cell.get("single_first_step_batch") is True for cell in cells),
    "no_fatal_log_pattern": not any(fatal.values()),
}
verification = {
    "schema": "eurosys_e2_direct_batch_pure_phase_verification_v1",
    "status": "ok" if all(checks.values()) else "error",
    "checks": checks,
    "fatal_patterns": fatal,
}
(root / "verification.json").write_text(
    json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
raise SystemExit(0 if verification["status"] == "ok" else 1)
PY
verify_rc=$?

echo "RUN_ROOT=${RUN_ROOT}"
echo "runner_returncode=${runner_rc}"
echo "verification=${RUN_ROOT}/verification.json"
(( runner_rc == 0 && journal_rc == 0 && verify_rc == 0 ))
