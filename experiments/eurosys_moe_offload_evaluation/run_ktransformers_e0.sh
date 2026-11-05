#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
CLIENT="${SCRIPT_DIR}/streaming_client.py"
VERIFIER="${SCRIPT_DIR}/verify_ktransformers_e0.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
PROTOCOL_ID="${PROTOCOL_ID:-eurosys-pa-v2-20260824}"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/${PROTOCOL_ID}/deepseek_v2_lite_bf16"
WARMUP_FIXTURE="${WARMUP_FIXTURE:-${FIXTURE_ROOT}/calibration_short_dialogue.json}"
EVALUATION_FIXTURE="${EVALUATION_FIXTURE:-${FIXTURE_ROOT}/validation_short_dialogue.json}"
PLACEMENT_MANIFEST="${PLACEMENT_MANIFEST:-${SCRIPT_DIR}/inputs/generated/eurosys-pa-v1-20260823/deepseek_v2_lite_bf16/calibration_assets_attempt3/kt_placements/kt_frequency_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/artifacts/ktransformers_e0_$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-30020}"
BASE_URL="http://127.0.0.1:${PORT}"
GPU_CLEAN_MIB="${GPU_CLEAN_MIB:-128}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"

SERVER_PID=""
MONITOR_PID=""
CURRENT_STAGE="preflight"

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

wait_for_clean_gpu() {
  local used_mib="" pids="" attempt
  for attempt in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
      && (( used_mib <= GPU_CLEAN_MIB )) \
      && [[ -z "${pids//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 2
  done
  die "GPU did not become clean (used=${used_mib:-unknown}, pids=${pids:-none})"
}

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

stop_server() {
  local attempt
  stop_monitor
  if [[ -z "${SERVER_PID}" ]]; then
    return 0
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    for attempt in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    sleep 2
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

write_failed() {
  local rc="$1"
  [[ -d "${OUTPUT_ROOT}" ]] || return 0
  [[ ! -e "${OUTPUT_ROOT}/COMPLETED.json" ]] || return 0
  [[ ! -e "${OUTPUT_ROOT}/FAILED.json" ]] || return 0
  jq -n \
    --arg stage "${CURRENT_STAGE}" \
    --argjson returncode "${rc}" \
    --arg created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{schema_version: 1, status: "failed", stage: $stage,
      returncode: $returncode, created_at: $created_at}' \
    >"${OUTPUT_ROOT}/FAILED.json" || true
}

cleanup() {
  local rc=$?
  stop_server
  if (( rc != 0 )); then
    write_failed "${rc}"
  fi
}
trap cleanup EXIT

wait_for_server() {
  local elapsed=0
  while (( elapsed < SERVER_START_TIMEOUT )); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 240 "${CURRENT_CASE}/server.log" >&2 || true
      die "KTransformers server exited during startup"
    fi
    # Four health requests/minute leaves room under the user's global
    # ten-request/minute ceiling for warmup, measurement, and metrics.
    sleep 15
    elapsed=$((elapsed + 15))
  done
  tail -n 240 "${CURRENT_CASE}/server.log" >&2 || true
  die "KTransformers server did not become ready within ${SERVER_START_TIMEOUT}s"
}

[[ "${ALLOW_GPU_RUN:-0}" == "1" ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ ! -e "${OUTPUT_ROOT}" ]] || die "refusing to overwrite ${OUTPUT_ROOT}"
require_file "${KT_PY}"
require_file "${CLIENT}"
require_file "${VERIFIER}"
require_file "${MODEL}/config.json"
require_file "${WARMUP_FIXTURE}"
require_file "${EVALUATION_FIXTURE}"
require_file "${PLACEMENT_MANIFEST}"
command -v curl >/dev/null || die "curl is unavailable"
command -v jq >/dev/null || die "jq is unavailable"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
command -v taskset >/dev/null || die "taskset is unavailable"

kt_commit="$(git -C "${KT_ROOT}" rev-parse HEAD)"
sglang_commit="$(git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD)"
[[ "${kt_commit}" == "eb9b70c4115cff151ace2cae5b0fc9db3690e31e" ]] \
  || die "KTransformers commit drift: ${kt_commit}"
[[ "${sglang_commit}" == "bc7f0058fafe6c05dc6ad52d6b1c854af2a673b2" ]] \
  || die "SGLang commit drift: ${sglang_commit}"
jq -e '.metadata.fixture_role == "calibration"
  and .metadata.workload == "short_dialogue"
  and .metadata.prompt_tokens == 128' "${WARMUP_FIXTURE}" >/dev/null
jq -e '.metadata.fixture_role == "validation"
  and .metadata.workload == "short_dialogue"
  and .metadata.prompt_tokens == 128
  and .metadata.output_tokens == 32' "${EVALUATION_FIXTURE}" >/dev/null

frequency_path="$(jq -er '.tiers["4g"].frequency_path' "${PLACEMENT_MANIFEST}")"
gpu_experts="$(jq -er '.tiers["4g"].gpu_experts_per_layer' "${PLACEMENT_MANIFEST}")"
require_file "${frequency_path}"
[[ "${gpu_experts}" == "54" ]] || die "4g placement expected 54 GPU experts/layer"
curl -fsS "${BASE_URL}/health" >/dev/null 2>&1 \
  && die "port ${PORT} is already serving"

mkdir -p "${OUTPUT_ROOT}/metadata"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
git -C "${KT_ROOT}" status --porcelain >"${OUTPUT_ROOT}/metadata/ktransformers.status"
git -C "${KT_ROOT}" diff --binary HEAD >"${OUTPUT_ROOT}/metadata/ktransformers.patch"
printf '%s\n' "${kt_commit}" >"${OUTPUT_ROOT}/metadata/ktransformers.commit"
printf '%s\n' "${sglang_commit}" >"${OUTPUT_ROOT}/metadata/sglang.commit"
sha256sum "${MODEL}/config.json" "${WARMUP_FIXTURE}" "${EVALUATION_FIXTURE}" \
  "${PLACEMENT_MANIFEST}" "${frequency_path}" >"${OUTPUT_ROOT}/metadata/inputs.sha256"
nvidia-smi -q >"${OUTPUT_ROOT}/metadata/nvidia_smi_q.txt"
lscpu >"${OUTPUT_ROOT}/metadata/lscpu.txt"
"${KT_PY}" -m pip freeze >"${OUTPUT_ROOT}/metadata/pip_freeze.txt"

for mode in eager graph; do
  CURRENT_STAGE="${mode}_server"
  CURRENT_CASE="${OUTPUT_ROOT}/${mode}"
  mkdir "${CURRENT_CASE}"
  server_command=(
    taskset -c 0-13
    "${KT_PY}" -m sglang.launch_server
    --model-path "${MODEL}"
    --host 127.0.0.1 --port "${PORT}"
    --tp-size 1
    --context-length 192
    --max-running-requests 1
    --mem-fraction-static 0.97
    --max-total-tokens 1024
    --chunked-prefill-size 256
    --max-prefill-tokens 256
    --schedule-policy fcfs
    --stream-interval 1
    --random-seed 20260823
    --trust-remote-code
    --kt-weight-path "${MODEL}"
    --kt-method BF16
    --kt-cpuinfer 14
    --kt-threadpool-count 1
    --kt-numa-nodes 0
    --kt-num-gpu-experts "${gpu_experts}"
    --kt-expert-placement-strategy frequency
    --init-expert-location "${frequency_path}"
    --kt-max-deferred-experts-per-token 0
    --kt-gpu-prefill-token-threshold 0
    --attention-backend flashinfer
    --cuda-graph-max-bs 1
    --cuda-graph-bs 1
    --enable-mixed-chunk
    --disable-shared-experts-fusion
    --disable-radix-cache
    --enable-metrics
  )
  if [[ "${mode}" == "eager" ]]; then
    server_command+=(--disable-cuda-graph)
  fi
  printf '%q ' "${server_command[@]}" >"${CURRENT_CASE}/server_command.txt"
  printf '\n' >>"${CURRENT_CASE}/server_command.txt"
  wait_for_clean_gpu
  setsid env \
    PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
    SGLANG_APPLY_CONFIG_BACKUP=none \
    "${server_command[@]}" >"${CURRENT_CASE}/server.log" 2>&1 &
  SERVER_PID="$!"
  printf '%s\n' "${SERVER_PID}" >"${CURRENT_CASE}/server.pid"
  wait_for_server
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits >"${CURRENT_CASE}/ready_compute_processes.csv"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits >"${CURRENT_CASE}/ready_gpu.csv"
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
    --format=csv,noheader,nounits -lms 200 \
    >"${CURRENT_CASE}/nvml.csv" 2>"${CURRENT_CASE}/nvml.err" &
  MONITOR_PID="$!"

  CURRENT_STAGE="${mode}_client"
  PATH="${KT_ROOT}/.venv/bin:${PATH}" "${KT_PY}" "${CLIENT}" \
    --system ktransformers \
    --model "${MODEL}" \
    --fixture "${EVALUATION_FIXTURE}" \
    --warmup-fixture "${WARMUP_FIXTURE}" \
    --output "${CURRENT_CASE}/client.json" \
    --base-url "${BASE_URL}" \
    --concurrencies 1 \
    --warmup-requests 3 \
    --warmup-output-tokens 8 \
    --measurement-requests 10 \
    --minimum-request-start-interval 6 \
    --request-timeout 3600 \
    --seed 20260823 \
    >"${CURRENT_CASE}/client.log" 2>&1
  curl -fsS "${BASE_URL}/metrics" >"${CURRENT_CASE}/metrics.after.prom"
  stop_server
  printf '0\n' >"${CURRENT_CASE}/server_returncode.txt"
  wait_for_clean_gpu
  nvidia-smi --query-gpu=timestamp,memory.used \
    --format=csv,noheader,nounits >"${CURRENT_CASE}/gpu_after.csv"
done

CURRENT_STAGE="verification"
"${KT_PY}" "${VERIFIER}" "${OUTPUT_ROOT}" \
  --placement-manifest "${PLACEMENT_MANIFEST}" \
  --protocol-id "${PROTOCOL_ID}" \
  --output "${OUTPUT_ROOT}/verification.json" \
  >"${OUTPUT_ROOT}/verification.stdout.json"
jq -e '.status == "ok"' "${OUTPUT_ROOT}/verification.json" >/dev/null
jq -n \
  --arg completed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg verification_sha256 "$(sha256sum "${OUTPUT_ROOT}/verification.json" | awk '{print $1}')" \
  '{schema_version: 1, status: "complete", completed_at: $completed_at,
    verification_sha256: $verification_sha256}' >"${OUTPUT_ROOT}/COMPLETED.json"
CURRENT_STAGE="complete"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
