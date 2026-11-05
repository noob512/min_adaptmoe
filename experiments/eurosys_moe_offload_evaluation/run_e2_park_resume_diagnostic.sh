#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
DIAGNOSTIC="${SCRIPT_DIR}/diagnose_e2_park_resume.py"
SERVER_ENV_JSON="${E2_DIAGNOSTIC_SERVER_ENV_JSON:-${SCRIPT_DIR}/configs/e2_ours_2g_env.json}"
MODEL="/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16"
EVALUATION_FIXTURE="${E2_DIAGNOSTIC_EVALUATION_FIXTURE:-${FIXTURE_ROOT}/evaluation_ordinary_chat.json}"
WARMUP_FIXTURE="${E2_DIAGNOSTIC_WARMUP_FIXTURE:-${FIXTURE_ROOT}/calibration_ordinary_chat.json}"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e2_park_resume_4k_$(date -u +%Y%m%dT%H%M%SZ)}"
CASE_ROOT="${RUN_ROOT}/case"
PORT="${PORT:-30201}"
DIAGNOSTIC_CUDAGRAPH_MODE="${E2_DIAGNOSTIC_CUDAGRAPH_MODE:-FULL_SEPARATE}"
DIAGNOSTIC_WARMUP_REQUESTS="${E2_DIAGNOSTIC_WARMUP_REQUESTS:-2}"
DIAGNOSTIC_WAVE_REQUESTS="${E2_DIAGNOSTIC_WAVE_REQUESTS:-8}"
DIAGNOSTIC_SEQUENCES_PER_REQUEST="${E2_DIAGNOSTIC_SEQUENCES_PER_REQUEST:-2}"
DIAGNOSTIC_WARMUP_OUTPUT_TOKENS="${E2_DIAGNOSTIC_WARMUP_OUTPUT_TOKENS:-8}"
DIAGNOSTIC_OUTPUT_TOKENS="${E2_DIAGNOSTIC_OUTPUT_TOKENS:-128}"
DIAGNOSTIC_MAX_INFERENCE_STARTS_PER_60S="${E2_DIAGNOSTIC_MAX_INFERENCE_STARTS_PER_60S:-10}"
DIAGNOSTIC_KV_CACHE_BYTES="${E2_DIAGNOSTIC_KV_CACHE_BYTES:-2147483648}"
DIAGNOSTIC_AUDIT_MIN_TOKENS="${E2_DIAGNOSTIC_AUDIT_MIN_TOKENS:-8192}"
DIAGNOSTIC_FUSED_MOE_CHUNK_SIZE="${E2_DIAGNOSTIC_FUSED_MOE_CHUNK_SIZE:-16384}"
DIAGNOSTIC_PREFILL_RESERVE_WARMUP_TOKENS="${E2_DIAGNOSTIC_PREFILL_RESERVE_WARMUP_TOKENS:-0}"
DIAGNOSTIC_NATIVE_HOLE_SENTINEL_BYTES="${E2_DIAGNOSTIC_NATIVE_HOLE_SENTINEL_BYTES:-0}"
SERVER_PID=""
SAMPLER_PID=""
SERVER_RETURN_CODE=""

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 2
}
for required in \
  "${VLLM_PY}" "${BASE_HARNESS}" "${DIAGNOSTIC}" "${SERVER_ENV_JSON}" \
  "${MODEL}/config.json" "${EVALUATION_FIXTURE}" "${WARMUP_FIXTURE}"; do
  [[ -e "${required}" ]] || {
    echo "missing required input: ${required}" >&2
    exit 2
  }
done

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ALLOW_GPU_RUN=1
export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1207959552}"
# shellcheck source=/dev/null
source "${BASE_HARNESS}"

stop_process_group() {
  local pid="$1" waited=0
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    while kill -0 "${pid}" 2>/dev/null && (( waited < 120 )); do
      sleep 0.25
      waited=$((waited + 1))
    done
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 2
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  set +e
  wait "${pid}" 2>/dev/null
  SERVER_RETURN_CODE="$?"
  set -e
}

cleanup_diagnostic() {
  local original_rc="$1" recovery_rc=0
  trap - EXIT INT TERM
  if [[ -n "${SAMPLER_PID}" ]]; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  stop_process_group "${SERVER_PID}"
  if [[ -d "${CASE_ROOT}" ]]; then
    printf '%s\n' "${SERVER_RETURN_CODE:-unknown}" \
      >"${CASE_ROOT}/server_returncode.txt"
  fi
  wait_gpu_clean || recovery_rc=1
  restore_stock || recovery_rc=1
  if [[ -d "${RUN_ROOT}" && "${MODULE_RESTORED}" == 1 ]]; then
    module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}" \
      || recovery_rc=1
  fi
  restart_exporters
  (( original_rc == 0 )) || exit "${original_rc}"
  exit "${recovery_rc}"
}

trap 'cleanup_diagnostic $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

runtime_preflight
mkdir -p "${CASE_ROOT}"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"

mapfile -t SERVER_ENV < <(
  jq -r 'to_entries[] | "\(.key)=\(.value)"' "${SERVER_ENV_JSON}"
)
SERVER_ENV+=(
  "CUDA_LAUNCH_BLOCKING=1"
  "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${CASE_ROOT}/offload_manifest.jsonl"
  "VLLM_UVM_LOG_FILE=${CASE_ROOT}/uvm.log"
  "VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE=${CASE_ROOT}/phase_transition.json"
  "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=1"
  "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE_GRANULARITY_BYTES=2097152"
  "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE_RUNTIME_GRANULARITY_BYTES=4096"
  "VLLM_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT=1"
  "VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS=${DIAGNOSTIC_AUDIT_MIN_TOKENS}"
  "VLLM_UVM_PREFILL_LAYER_MEMORY_TRACE=${E2_DIAGNOSTIC_LAYER_MEMORY_TRACE:-0}"
  "VLLM_UVM_PREFILL_LAYER_MEMORY_TRACE_MIN_TOKENS=${DIAGNOSTIC_AUDIT_MIN_TOKENS}"
  "VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=${DIAGNOSTIC_PREFILL_RESERVE_WARMUP_TOKENS}"
  "VLLM_UVM_NATIVE_HOLE_SENTINEL_BYTES=${DIAGNOSTIC_NATIVE_HOLE_SENTINEL_BYTES}"
  "VLLM_UVM_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=${E2_DIAGNOSTIC_NATIVE_HOLE_SENTINEL_EMPTY_CACHE:-0}"
  "VLLM_FUSED_MOE_CHUNK_SIZE=${DIAGNOSTIC_FUSED_MOE_CHUNK_SIZE}"
  "VLLM_UVM_KV_CACHE_MANAGED=0"
  "VLLM_UVM_KV_RUNTIME_ENABLE=0"
  "VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0"
  "VLLM_UVM_KV_AUTO_SCALE_ENABLE=0"
  "VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0"
  "VLLM_UVM_POLICY_RUNTIME_ENABLE=0"
)
case "${DIAGNOSTIC_CUDAGRAPH_MODE}" in
  FULL_SEPARATE)
    COMPILATION_CONFIG='{"mode":"NONE","cudagraph_mode":"FULL_SEPARATE","cudagraph_capture_sizes":[1,4,8,16,32],"max_cudagraph_capture_size":32}'
    ;;
  NONE)
    COMPILATION_CONFIG='{"mode":"NONE","cudagraph_mode":"NONE"}'
    SERVER_ENV+=(
      "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_remote_decode"
    )
    ;;
  *)
    echo "unsupported E2_DIAGNOSTIC_CUDAGRAPH_MODE: ${DIAGNOSTIC_CUDAGRAPH_MODE}" >&2
    exit 2
    ;;
esac
SERVER_ENV+=("E2_DIAGNOSTIC_CUDAGRAPH_MODE=${DIAGNOSTIC_CUDAGRAPH_MODE}")
printf '%s\n' "${SERVER_ENV[@]}" >"${CASE_ROOT}/environment.txt"

SERVER_COMMAND=(
  taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server
  --host 127.0.0.1 --port "${PORT}" --disable-frontend-multiprocessing
  --model "${MODEL}" --trust-remote-code --dtype bfloat16
  --max-model-len 8320 --gpu-memory-utilization 0.97
  --kv-cache-memory-bytes "${DIAGNOSTIC_KV_CACHE_BYTES}" --cpu-offload-gb 30
  --max-num-batched-tokens 16384 --max-num-seqs 16
  --generation-config vllm
  --compilation-config "${COMPILATION_CONFIG}"
)
printf '%q ' "${SERVER_COMMAND[@]}" >"${CASE_ROOT}/server_command.txt"
printf '\n' >>"${CASE_ROOT}/server_command.txt"

KERNEL_EPOCH="$(date +%s)"
setsid env "${SERVER_ENV[@]}" "${SERVER_COMMAND[@]}" \
  >"${CASE_ROOT}/server.log" 2>&1 &
SERVER_PID="$!"
nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits --loop-ms=200 \
  >"${CASE_ROOT}/nvml.csv" 2>"${CASE_ROOT}/nvml.err" &
SAMPLER_PID="$!"

READY=0
for ((attempt = 0; attempt < 1200; ++attempt)); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    break
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" \
      >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done
if (( READY == 0 )); then
  echo "server did not become healthy" >&2
  exit 3
fi

set +e
"${VLLM_PY}" "${DIAGNOSTIC}" \
  --model "${MODEL}" \
  --fixture "${EVALUATION_FIXTURE}" \
  --warmup-fixture "${WARMUP_FIXTURE}" \
  --output "${CASE_ROOT}/diagnostic.json" \
  --base-url "http://127.0.0.1:${PORT}" \
  --warmup-requests "${DIAGNOSTIC_WARMUP_REQUESTS}" \
  --wave-requests "${DIAGNOSTIC_WAVE_REQUESTS}" \
  --sequences-per-request "${DIAGNOSTIC_SEQUENCES_PER_REQUEST}" \
  --warmup-output-tokens "${DIAGNOSTIC_WARMUP_OUTPUT_TOKENS}" \
  --output-tokens "${DIAGNOSTIC_OUTPUT_TOKENS}" \
  --minimum-start-interval 6.1 --timeout 7200 \
  --max-inference-starts-per-60s \
    "${DIAGNOSTIC_MAX_INFERENCE_STARTS_PER_60S}" \
  >"${CASE_ROOT}/diagnostic.stdout" 2>&1
CLIENT_RC="$?"
set -e
printf '%s\n' "${CLIENT_RC}" >"${CASE_ROOT}/diagnostic_returncode.txt"

stop_process_group "${SERVER_PID}"
SERVER_PID=""
sudo journalctl -k --since "@${KERNEL_EPOCH}" --no-pager -o short-unix \
  >"${CASE_ROOT}/kernel.since.log" 2>"${CASE_ROOT}/kernel.since.err" || true

if [[ -f "${CASE_ROOT}/diagnostic.json" ]]; then
  jq '{status, rate_limit, configuration,
       successful_requests: ([.records[] | select(.success)] | length),
       total_requests: (.records | length)}' "${CASE_ROOT}/diagnostic.json"
fi
exit "${CLIENT_RC}"
