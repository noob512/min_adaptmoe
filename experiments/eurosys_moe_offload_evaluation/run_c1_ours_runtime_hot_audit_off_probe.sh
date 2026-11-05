#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ROOT="${1:-}"
MODEL="/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
BLOCK_RUNNER="${SCRIPT_DIR}/formal_openai_block.py"
CLIENT="${SCRIPT_DIR}/streaming_client.py"
WORKLOAD_CONFIG="${SCRIPT_DIR}/configs/c1_short_mechanism_pilot.json"
ENV_CONFIG="${SCRIPT_DIR}/configs/c1_ours_runtime_hot_audit_off_env.json"
REFERENCE_ROOT="${SCRIPT_DIR}/artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01"
COLD_PLAN="${REFERENCE_ROOT}/provenance/cold1_plan/cold_plan_c1.json"
EXACT_NAMES="${REFERENCE_ROOT}/provenance/cold1_plan/exact_parameter_names.txt"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

die() { echo "error: $*" >&2; exit 1; }

[[ -n "${RUN_ROOT}" ]] || die "usage: $0 RUN_ROOT"
[[ ! -e "${RUN_ROOT}" ]] || die "refusing to overwrite ${RUN_ROOT}"
for path in "${MODEL}/config.json" "${VLLM_PY}" "${BLOCK_RUNNER}" \
  "${CLIENT}" "${WORKLOAD_CONFIG}" "${ENV_CONFIG}" "${COLD_PLAN}" \
  "${EXACT_NAMES}" "${BASE_HARNESS}"; do
  [[ -e "${path}" ]] || die "missing required path: ${path}"
done

mkdir -p "${RUN_ROOT}/provenance"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
sha256sum "${MODEL}/config.json" "${BLOCK_RUNNER}" "${CLIENT}" \
  "${WORKLOAD_CONFIG}" "${ENV_CONFIG}" "${COLD_PLAN}" "${EXACT_NAMES}" \
  "${REPO_ROOT}/workloads/vllm/vllm/vllm/model_executor/models/utils.py" \
  >"${RUN_ROOT}/provenance/input_sha256.txt"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/git_status.txt"
git -C "${REPO_ROOT}" diff --binary -- \
  workloads/vllm/vllm/vllm/model_executor/models/utils.py \
  workloads/vllm/vllm/tests/test_cpu_offload_exact_static.py \
  >"${RUN_ROOT}/provenance/runtime_hot_audit_off.patch"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ALLOW_GPU_RUN=1
export SLOT_BUDGET_BYTES=1207959552
export MANUAL_MIGRATION_FAULT_MODE=2
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"

CASE_ROOT="${RUN_ROOT}/ours_cold1_short_dialogue_c1"
set +e
"${VLLM_PY}" "${BLOCK_RUNNER}" \
  --case-root "${CASE_ROOT}" \
  --protocol-id eurosys-pa-v2-20260824 \
  --system ours \
  --model "${MODEL}" \
  --base-url http://127.0.0.1:30304 \
  --workload-config "${WORKLOAD_CONFIG}" \
  --server-env "${ENV_CONFIG}" \
  --server-env-override "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${EXACT_NAMES}" \
  --server-env-override "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE=${COLD_PLAN}" \
  --server-env-override "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${CASE_ROOT}/offload_manifest.jsonl" \
  --server-env-override "VLLM_UVM_LOG_FILE=${CASE_ROOT}/uvm.log" \
  --server-env-override "VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE=${CASE_ROOT}/phase_transition.json" \
  --python "${VLLM_PY}" \
  --streaming-client "${CLIENT}" \
  --ready-timeout 1200 \
  --request-timeout 1800 \
  --gpu-clean-timeout 240 \
  --stop-on-first-workload-failure \
  -- \
  taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 30304 --disable-frontend-multiprocessing \
  --model "${MODEL}" --trust-remote-code --dtype bfloat16 \
  --max-model-len 192 --gpu-memory-utilization 0.97 \
  --kv-cache-memory-bytes 6291456 --cpu-offload-gb 30 \
  --max-num-batched-tokens 128 --max-num-seqs 1 \
  --no-enable-prefix-caching --generation-config vllm \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_SEPARATE","cudagraph_capture_sizes":[1,128],"max_cudagraph_capture_size":128}' \
  >"${RUN_ROOT}/block_runner.log" 2>&1
BLOCK_RC=$?
set -e
printf '%s\n' "${BLOCK_RC}" >"${RUN_ROOT}/returncode.txt"

restore_stock
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters

if (( BLOCK_RC != 0 )); then
  tail -n 200 "${RUN_ROOT}/block_runner.log" >&2 || true
  exit "${BLOCK_RC}"
fi

echo "RUN_ROOT=${RUN_ROOT}"
echo "verification=${CASE_ROOT}/verification.json"
