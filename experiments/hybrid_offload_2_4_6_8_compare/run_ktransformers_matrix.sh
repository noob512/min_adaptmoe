#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
PYTHON="${KT_ROOT}/.venv/bin/python"
CLIENT="${REPO_ROOT}/experiments/ktransformers_cpu_expert_compare/benchmark_sglang_stream.py"
MODEL="${MODEL:?Set MODEL to the local DeepSeek-V2-Lite-Chat snapshot directory}"
EVAL_FIXTURE="${EVAL_FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/deepseek_v2_lite_bf16/fixtures/evaluation_fixture.json}"
WARMUP_FIXTURE="${WARMUP_FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/deepseek_v2_lite_bf16/fixtures/warmup_fixture.json}"
PLACEMENT_MANIFEST="${PLACEMENT_MANIFEST:?Generate KT placements and set PLACEMENT_MANIFEST to the resulting kt_frequency_manifest.json}"
TIERS="${TIERS:-2g,4g,6g,8g}"
REPEATS="${REPEATS:-3}"
PORT="${PORT:-30010}"
BASE_URL="http://127.0.0.1:${PORT}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/ktransformers_$(date -u +%Y%m%dT%H%M%SZ)}"
SERVER_PID=""

die() { echo "error: $*" >&2; exit 1; }

gpu_clean() {
  local used pids
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ "${used}" == 0 && -z "${pids//[[:space:]]/}" ]]
}

wait_gpu_clean() {
  local attempt
  for ((attempt = 0; attempt < 300; ++attempt)); do
    gpu_clean && return 0
    sleep 1
  done
  return 1
}

stop_server() {
  local attempt
  [[ -n "${SERVER_PID}" ]] || return 0
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    for ((attempt = 0; attempt < 30; ++attempt)); do
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

trap 'stop_server' EXIT INT TERM

wait_server() {
  local case_dir="$1" attempt
  for ((attempt = 0; attempt < 900; ++attempt)); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 200 "${case_dir}/server.log" >&2 || true
      return 1
    fi
    sleep 1
  done
  return 1
}

run_client() {
  local fixture="$1" output="$2" output_tokens="$3"
  PATH="${KT_ROOT}/.venv/bin:${PATH}" "${PYTHON}" "${CLIENT}" \
    --fixture "${fixture}" --output "${output}" --base-url "${BASE_URL}" \
    --concurrencies 1 --output-tokens "${output_tokens}" \
    --request-timeout 1800 --flush-cache-before-phase \
    --require-prompt-token-ids
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be positive"
[[ -x "${PYTHON}" ]] || die "missing KT Python: ${PYTHON}"
[[ -f "${CLIENT}" ]] || die "missing client: ${CLIENT}"
[[ -f "${MODEL}/config.json" ]] || die "missing model: ${MODEL}"
[[ -f "${EVAL_FIXTURE}" && -f "${WARMUP_FIXTURE}" ]] || die "missing fixtures"
[[ -f "${PLACEMENT_MANIFEST}" ]] || die "missing placement manifest"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
curl -fsS "${BASE_URL}/health" >/dev/null 2>&1 && die "port ${PORT} is already serving"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

printf 'tier\trepeat\tresult\n' >"${RUN_ROOT}/execution.tsv"
IFS=',' read -r -a tiers <<<"${TIERS}"
for tier in "${tiers[@]}"; do
  frequency="$(jq -er --arg tier "${tier}" '.tiers[$tier].frequency_path' "${PLACEMENT_MANIFEST}")"
  gpu_experts="$(jq -er --arg tier "${tier}" '.tiers[$tier].gpu_experts_per_layer' "${PLACEMENT_MANIFEST}")"
  [[ -f "${frequency}" ]] || die "missing frequency plan: ${frequency}"
  case_dir="${RUN_ROOT}/${tier}"
  mkdir "${case_dir}"
  wait_gpu_clean || die "GPU is not clean before ${tier}"

  server_command=(
    taskset -c 0-13 "${PYTHON}" -m sglang.launch_server
    --model-path "${MODEL}" --host 127.0.0.1 --port "${PORT}"
    --tp-size 1 --context-length 544 --max-running-requests 1
    --mem-fraction-static 0.97
    --max-total-tokens 1024 --chunked-prefill-size 1024
    --max-prefill-tokens 1024 --schedule-policy fcfs --stream-interval 1
    --random-seed 0 --trust-remote-code
    --kt-weight-path "${MODEL}" --kt-method BF16 --kt-cpuinfer 14
    --kt-threadpool-count 1 --kt-numa-nodes 0
    --kt-num-gpu-experts "${gpu_experts}"
    --kt-expert-placement-strategy frequency --init-expert-location "${frequency}"
    --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0
    --attention-backend flashinfer --cuda-graph-max-bs 1 --cuda-graph-bs 1
    --enable-mixed-chunk --disable-shared-experts-fusion --disable-radix-cache
  )
  printf '%q ' "${server_command[@]}" >"${case_dir}/server_command.txt"
  printf '\n' >>"${case_dir}/server_command.txt"
  setsid env PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    SGLANG_APPLY_CONFIG_BACKUP=none \
    "${server_command[@]}" >"${case_dir}/server.log" 2>&1 &
  SERVER_PID="$!"
  echo "${SERVER_PID}" >"${case_dir}/server.pid"
  wait_server "${case_dir}" || die "KT server failed to start for ${tier}"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits \
    >"${case_dir}/gpu.ready.csv"

  run_client "${WARMUP_FIXTURE}" "${case_dir}/warmup.json" 4 \
    >"${case_dir}/warmup.log" 2>&1
  for ((repeat = 1; repeat <= REPEATS; ++repeat)); do
    output="${case_dir}/eval_r${repeat}.json"
    run_client "${EVAL_FIXTURE}" "${output}" 12 \
      >"${case_dir}/eval_r${repeat}.log" 2>&1
    jq -e '.status == "ok" and .phases[0].failed_requests == 0 and
           .phases[0].completed_requests == 1 and
           .phases[0].completed_output_tokens == 12' "${output}" >/dev/null \
      || die "KT client result failed: ${tier}/r${repeat}"
    printf '%s\t%s\t%s\n' "${tier}" "${repeat}" "${output}" \
      >>"${RUN_ROOT}/execution.tsv"
  done
  stop_server
  wait_gpu_clean || die "GPU did not clean up after ${tier}"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    >"${case_dir}/gpu.after.csv"
  grep -E 'KT GPU experts: layer .* \(MoE\) has' "${case_dir}/server.log" \
    >"${case_dir}/placement_counts.log" || true
  grep -Ei 'cuda graph|cudagraph' "${case_dir}/server.log" \
    >"${case_dir}/cuda_graph_evidence.log" || true
done

echo "RUN_ROOT=${RUN_ROOT}"
