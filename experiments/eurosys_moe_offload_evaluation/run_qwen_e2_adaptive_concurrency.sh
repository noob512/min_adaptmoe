#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
BLOCK_RUNNER="${SCRIPT_DIR}/formal_openai_block.py"
CLIENT="${SCRIPT_DIR}/streaming_client.py"
WORKLOADS="${SCRIPT_DIR}/configs/e2_matrix_cells.json"
ENV_FILE="${SCRIPT_DIR}/configs/e2_qwen_adaptive_concurrency_env.json"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8"
PLAN_ROOT="${FIXTURE_ROOT}/e2_adaptive_plans"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
MODEL="${FORMAL_MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${1:-}"

die() {
  echo "error: $*" >&2
  exit 1
}

[[ -n "${RUN_ROOT}" ]] || die "usage: $0 NEW_RUN_ROOT"
[[ ! -e "${RUN_ROOT}" ]] || die "run root already exists: ${RUN_ROOT}"
[[ -x "${VLLM_PY}" && -f "${MODEL}/config.json" ]] \
  || die "Qwen model or vLLM environment is unavailable"
mkdir -p "${RUN_ROOT}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ALLOW_GPU_RUN=1
export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1207959552}"
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
RUN_ROOT_ACTIVE="${RUN_ROOT}"

run_process_block() {
  local label="$1" width="$2" workload_filter="$3" capture_sizes="$4"
  local case_root="${RUN_ROOT}/${label}"
  local plan_dir="${PLAN_ROOT}/cold${width}"
  local plan="${plan_dir}/cold_decode_plan_2g.json"
  local names="${plan_dir}/exact_parameter_names.txt"
  local compilation_config
  [[ -f "${plan}" && -f "${names}" ]] || die "missing cold${width} plan"
  compilation_config="$(${VLLM_PY} - "${capture_sizes}" <<'PY'
import json, sys
sizes = sorted({int(value) for value in sys.argv[1].split(",")})
print(json.dumps({
    "mode": "NONE",
    "cudagraph_mode": "FULL_SEPARATE",
    "cudagraph_capture_sizes": sizes,
    "max_cudagraph_capture_size": max(sizes),
}, separators=(",", ":")))
PY
)"

  "${VLLM_PY}" "${BLOCK_RUNNER}" \
    --case-root "${case_root}" \
    --protocol-id eurosys-pa-v2-20260824 \
    --system ours \
    --model "${MODEL}" \
    --base-url http://127.0.0.1:30201 \
    --workload-config "${WORKLOADS}" \
    --workload-filter "${workload_filter}" \
    --fixture-root "${FIXTURE_ROOT}" \
    --server-env "${ENV_FILE}" \
    --server-env-override "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE=${plan}" \
    --server-env-override "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${names}" \
    --server-env-override "VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE=${case_root}/phase_transition.json" \
    --server-env-override "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${case_root}/offload_manifest.jsonl" \
    --server-env-override "VLLM_UVM_LOG_FILE=${case_root}/uvm.log" \
    --python "${VLLM_PY}" \
    --streaming-client "${CLIENT}" \
    --ready-timeout 1200 \
    --request-timeout 7200 \
    --gpu-clean-timeout 240 \
    --stop-on-first-workload-failure \
    -- \
    taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 --port 30201 --disable-frontend-multiprocessing \
      --model "${MODEL}" --trust-remote-code --dtype auto \
      --max-model-len 8320 --gpu-memory-utilization 0.97 \
      --kv-cache-memory-bytes 2147483648 --cpu-offload-gb 30 \
      --max-num-batched-tokens 16384 --max-num-seqs 16 \
      --no-enable-prefix-caching --generation-config vllm \
      --compilation-config "${compilation_config}"
}

runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"

# The mapping is frozen before evaluation. C1 never exceeds an 8,192-token
# Prefill step and uses cold19. C4/C8/C16 can reach the 16,384-token scheduler
# ceiling and use cold25. Within either process block the placement is static.
run_process_block \
  c1_cold19 19 \
  short_dialogue_c1,ordinary_chat_c1,rag_summary_c1,long_generation_c1 \
  1,4,8,16,128,512,1024,2048,4096,8192

run_process_block \
  c4_c8_c16_cold25 25 \
  short_dialogue_c4,short_dialogue_c8,short_dialogue_c16,ordinary_chat_c4,ordinary_chat_c8,ordinary_chat_c16,rag_summary_c4,rag_summary_c8,rag_summary_c16,long_generation_c4,long_generation_c8,long_generation_c16 \
  1,4,8,16,128,512,1024,2048,4096,8192,16384

restore_stock || die "failed to restore stock UVM"
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters
echo "QWEN_E2_ADAPTIVE_ROOT=${RUN_ROOT}"
