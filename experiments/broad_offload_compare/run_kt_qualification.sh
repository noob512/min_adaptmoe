#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
BENCHMARK="${REPO_ROOT}/experiments/ktransformers_cpu_expert_compare/benchmark_sglang_stream.py"
PREPARE_FIXTURES="${REPO_ROOT}/experiments/custom_cold_qwen_sharegpt_concurrency_snapshot/prepare_sharegpt_fixtures.py"
REFERENCE_SNAPSHOT="${REFERENCE_SNAPSHOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/e9_sharegpt_selection_reference_20260828.json}"
PREPARE_STATS="${SCRIPT_DIR}/prepare_kt_frequency_stats.py"
VERIFY_CASE="${SCRIPT_DIR}/verify_kt_qualification_case.py"
SUMMARIZER="${SCRIPT_DIR}/summarize_kt_qualification.py"

EXPECTED_KT_COMMIT="5a9c7d4cfaaed0136930e4b629ce5d55cec71a54"
EXPECTED_SGLANG_COMMIT="f3ef6723f61ecbfa3cd28b76eea08af6da49a7d0"
EXPECTED_SGLANG_PATCH_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
EXPECTED_MODEL_SNAPSHOT="d206ba732169f29bb77fbf80fc2c4b81d4d30782"
EXPECTED_DATASET_SHA256="35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/${EXPECTED_MODEL_SNAPSHOT}}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/broad_offload_kt_qualification_$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
CPUSET="${CPUSET:-0-13}"
GPU_EXPERTS_RATIO="${GPU_EXPERTS_RATIO:-0.77083333333333337}"
KV_TOKENS="${KV_TOKENS:-13104}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-512}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
QUAL_CHUNKED_PREFILL_SIZE="${QUAL_CHUNKED_PREFILL_SIZE:-4096}"
QUAL_MAX_PREFILL_TOKENS="${QUAL_MAX_PREFILL_TOKENS:-4096}"
QUAL_CONCURRENCIES="${QUAL_CONCURRENCIES:-1,8}"
QUAL_OUTPUT_TOKENS="${QUAL_OUTPUT_TOKENS:-32}"
WARM_CONCURRENCY="${WARM_CONCURRENCY:-8}"
WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS:-8}"
CALIBRATION_CONCURRENCY="${CALIBRATION_CONCURRENCY:-32}"
CALIBRATION_OUTPUT_TOKENS="${CALIBRATION_OUTPUT_TOKENS:-8}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
GPU_CLEAN_MIB="${GPU_CLEAN_MIB:-128}"
QUAL_DISABLE_CUDA_GRAPH="${QUAL_DISABLE_CUDA_GRAPH:-0}"
QUAL_PLACEMENT_STRATEGY="${QUAL_PLACEMENT_STRATEGY:-frequency}"
QUAL_DISABLE_OVERLAP_SCHEDULE="${QUAL_DISABLE_OVERLAP_SCHEDULE:-0}"
QUAL_ENABLE_MIXED_CHUNK="${QUAL_ENABLE_MIXED_CHUNK:-1}"

FIXTURE_DIR="${RUN_ROOT}/fixtures"
EVALUATION_FIXTURE="${EVALUATION_FIXTURE_OVERRIDE:-${FIXTURE_DIR}/evaluation_fixture.json}"
CALIBRATION_FIXTURE="${CALIBRATION_FIXTURE_OVERRIDE:-${FIXTURE_DIR}/calibration_fixture.json}"
CALIBRATION_DIR="${RUN_ROOT}/calibration"
RAW_RECORDER_DIR="${CALIBRATION_DIR}/recorder_raw"
FREQUENCY_STATS="${FREQUENCY_STATS_OVERRIDE:-${CALIBRATION_DIR}/frequency_stats.pt}"
FREQUENCY_STATS_MANIFEST="${FREQUENCY_STATS_MANIFEST_OVERRIDE:-${CALIBRATION_DIR}/frequency_stats_manifest.json}"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

SERVER_PID=""
MONITOR_PID=""
SERVER_CASE_DIR=""
SERVER_COMMAND=()

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: $1"
}

wait_for_clean_gpu() {
  local used_mib=""
  local compute_processes=""
  local _
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    compute_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
      && (( used_mib <= GPU_CLEAN_MIB )) \
      && [[ -z "${compute_processes//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  nvidia-smi >&2 || true
  die "GPU did not become clean (used=${used_mib:-unknown} MiB, pids=${compute_processes:-none})"
}

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

stop_server() {
  local _
  stop_monitor
  if [[ -z "${SERVER_PID}" ]]; then
    return 0
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
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

cleanup() {
  stop_server
}
trap cleanup EXIT

preflight() {
  require_dir "${KT_ROOT}"
  require_file "${KT_PY}"
  require_file "${VLLM_PY}"
  require_file "${BENCHMARK}"
  require_file "${PREPARE_FIXTURES}"
  require_file "${REFERENCE_SNAPSHOT}"
  require_file "${PREPARE_STATS}"
  require_file "${VERIFY_CASE}"
  require_file "${SUMMARIZER}"
  require_dir "${MODEL}"
  require_file "${MODEL}/config.json"
  require_file "${DATASET}"
  command -v curl >/dev/null || die "curl is unavailable"
  command -v jq >/dev/null || die "jq is unavailable"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v taskset >/dev/null || die "taskset is unavailable"

  local kt_commit sglang_commit sglang_patch_sha model_snapshot dataset_sha physical_cores numa_nodes
  kt_commit="$(git -C "${KT_ROOT}" rev-parse HEAD)"
  sglang_commit="$(git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD)"
  sglang_patch_sha="$(git -C "${KT_ROOT}/third_party/sglang" diff --binary | sha256sum | awk '{print $1}')"
  model_snapshot="$(basename "${MODEL}")"
  dataset_sha="$(sha256sum "${DATASET}" | awk '{print $1}')"
  physical_cores="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 ":" $2}' | sort -u | wc -l)"
  numa_nodes="$(lscpu | awk -F: '/^NUMA node\(s\)/ {gsub(/[[:space:]]/, "", $2); print $2}')"
  [[ "${kt_commit}" == "${EXPECTED_KT_COMMIT}" ]] || die "KT commit drift: ${kt_commit}"
  [[ "${sglang_commit}" == "${EXPECTED_SGLANG_COMMIT}" ]] || die "SGLang commit drift: ${sglang_commit}"
  [[ "${sglang_patch_sha}" == "${EXPECTED_SGLANG_PATCH_SHA256}" ]] \
    || die "SGLang compatibility patch drift: ${sglang_patch_sha}"
  [[ "${model_snapshot}" == "${EXPECTED_MODEL_SNAPSHOT}" ]] || die "model snapshot drift: ${model_snapshot}"
  [[ "${dataset_sha}" == "${EXPECTED_DATASET_SHA256}" ]] || die "dataset hash drift: ${dataset_sha}"
  [[ "${physical_cores}" == "14" ]] || die "expected 14 physical cores, got ${physical_cores}"
  [[ "${numa_nodes}" == "1" ]] || die "expected one NUMA node, got ${numa_nodes}"

  PATH="${KT_ROOT}/.venv/bin:${PATH}" "${KT_PY}" - <<'PY'
import torch
import kt_kernel
from kt_kernel import kt_kernel_ext

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (12, 0)
assert getattr(kt_kernel, "__version__", None) == "0.6.4"
assert getattr(kt_kernel, "__cpu_variant__", None) == "avx512_bf16"
assert hasattr(kt_kernel_ext.CPUInfer, "submit_with_cuda_stream")
print("KT qualification preflight: version=0.6.4 variant=avx512_bf16 sm=120")
PY

  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    die "${BASE_URL} is already serving; choose another PORT or stop it"
  fi
  echo "KT qualification preflight: OK"
}

prepare_fixtures() {
  preflight
  mkdir -p "${RUN_ROOT}" "${FIXTURE_DIR}"
  if [[ ! -s "${EVALUATION_FIXTURE}" || ! -s "${CALIBRATION_FIXTURE}" ]]; then
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "${VLLM_PY}" "${PREPARE_FIXTURES}" \
        --dataset "${DATASET}" \
        --model "${MODEL}" \
        --reference-snapshot "${REFERENCE_SNAPSHOT}" \
        --output-dir "${FIXTURE_DIR}" \
        --request-count 32 \
        --prompt-min-tokens 300 \
        --prompt-max-tokens 320 \
        --output-tokens 64 \
        > "${FIXTURE_DIR}/prepare_fixtures.stdout.json"
  fi
  require_file "${EVALUATION_FIXTURE}"
  require_file "${CALIBRATION_FIXTURE}"
  require_file "${FIXTURE_DIR}/fixture_metadata.json"
  jq -e '.disjoint_source_indices == true and .evaluation_matches_native_snapshot == true' \
    "${FIXTURE_DIR}/fixture_metadata.json" >/dev/null \
    || die "fixture disjointness/reference validation failed"
  sha256sum "${EVALUATION_FIXTURE}" "${CALIBRATION_FIXTURE}" \
    > "${FIXTURE_DIR}/fixture_sha256.txt"
  echo "fixtures=${FIXTURE_DIR}"
}

record_run_metadata() {
  mkdir -p "${RUN_ROOT}/metadata"
  git -C "${KT_ROOT}" rev-parse HEAD > "${RUN_ROOT}/metadata/ktransformers.commit"
  git -C "${KT_ROOT}" submodule status > "${RUN_ROOT}/metadata/ktransformers.submodules"
  git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD \
    > "${RUN_ROOT}/metadata/sglang.commit"
  git -C "${KT_ROOT}/third_party/sglang" diff --binary \
    > "${RUN_ROOT}/metadata/sglang.compatibility.patch"
  sha256sum "${RUN_ROOT}/metadata/sglang.compatibility.patch" \
    > "${RUN_ROOT}/metadata/sglang.compatibility.patch.sha256"
  git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_ROOT}/metadata/uvm_repo.commit"
  lscpu > "${RUN_ROOT}/metadata/lscpu.txt"
  nvidia-smi -q > "${RUN_ROOT}/metadata/nvidia_smi_q.txt"
  "${KT_PY}" -m pip freeze > "${RUN_ROOT}/metadata/kt_pip_freeze.txt"
}

wait_for_server() {
  local elapsed
  for elapsed in $(seq 1 "${SERVER_START_TIMEOUT}"); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      echo "server ready after ${elapsed}s"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 240 "${SERVER_CASE_DIR}/server.log" >&2 || true
      die "KT server exited during startup"
    fi
    sleep 1
  done
  tail -n 240 "${SERVER_CASE_DIR}/server.log" >&2 || true
  die "KT server did not become ready within ${SERVER_START_TIMEOUT}s"
}

start_server() {
  local case_dir="$1"
  local recorder_dir="${2:-}"
  SERVER_CASE_DIR="${case_dir}"
  mkdir -p "${case_dir}"
  [[ ! -e "${case_dir}/server.log" ]] || die "server log already exists: ${case_dir}/server.log"
  printf '%q ' "${SERVER_COMMAND[@]}" > "${case_dir}/server_command.txt"
  printf '\n' >> "${case_dir}/server_command.txt"
  {
    echo "PATH=${KT_ROOT}/.venv/bin:${PATH}"
    echo "HF_HUB_OFFLINE=1"
    echo "TRANSFORMERS_OFFLINE=1"
    echo "TOKENIZERS_PARALLELISM=false"
    echo "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${recorder_dir:-unset}"
  } > "${case_dir}/server_environment.txt"

  wait_for_clean_gpu
  if [[ -n "${recorder_dir}" ]]; then
    mkdir -p "${recorder_dir}"
    setsid env \
      PATH="${KT_ROOT}/.venv/bin:${PATH}" \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${recorder_dir}" \
      "${SERVER_COMMAND[@]}" > "${case_dir}/server.log" 2>&1 &
  else
    setsid env \
      PATH="${KT_ROOT}/.venv/bin:${PATH}" \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      "${SERVER_COMMAND[@]}" > "${case_dir}/server.log" 2>&1 &
  fi
  SERVER_PID="$!"
  echo "${SERVER_PID}" > "${case_dir}/server.pid"
  wait_for_server
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits > "${case_dir}/ready_compute_processes.csv"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits > "${case_dir}/ready_gpu.csv"
  taskset -pc "${SERVER_PID}" > "${case_dir}/server_affinity.txt"
}

run_client() {
  local fixture="$1"
  local output="$2"
  local concurrencies="$3"
  local output_tokens="$4"
  local -a prompt_args=()
  if [[ "${REQUIRE_PROMPT_TOKEN_IDS:-0}" == "1" ]]; then
    prompt_args+=(--require-prompt-token-ids)
  fi
  PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    "${KT_PY}" "${BENCHMARK}" \
      --fixture "${fixture}" \
      --output "${output}" \
      --base-url "${BASE_URL}" \
      --concurrencies "${concurrencies}" \
      --output-tokens "${output_tokens}" \
      --request-timeout "${REQUEST_TIMEOUT}" \
      --flush-cache-before-phase \
      "${prompt_args[@]}"
}

build_calibration_server_command() {
  SERVER_COMMAND=(
    taskset -c "${CPUSET}"
    "${KT_PY}" -m sglang.launch_server
    --model-path "${MODEL}"
    --host 127.0.0.1 --port "${PORT}"
    --tp-size 1
    --context-length "${MAX_CONTEXT_LENGTH}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --max-total-tokens "${KV_TOKENS}"
    --chunked-prefill-size "${QUAL_CHUNKED_PREFILL_SIZE}"
    --max-prefill-tokens "${QUAL_MAX_PREFILL_TOKENS}"
    --schedule-policy fcfs --stream-interval 1 --random-seed 0
    --kt-weight-path "${MODEL}"
    --kt-method FP8
    --kt-cpuinfer 14 --kt-threadpool-count 1 --kt-numa-nodes 0
    --kt-gpu-experts-ratio "${GPU_EXPERTS_RATIO}"
    --kt-expert-placement-strategy uniform
    --kt-max-deferred-experts-per-token 0
    --kt-gpu-prefill-token-threshold 0
    --attention-backend flashinfer --fp8-gemm-backend triton
    --expert-distribution-recorder-mode stat
    --expert-distribution-recorder-buffer-size -1
    --disable-shared-experts-fusion
    --disable-radix-cache
    --disable-cuda-graph
    --disable-overlap-schedule
  )
}

record_calibration() {
  prepare_fixtures
  record_run_metadata
  [[ ! -e "${CALIBRATION_DIR}/server.log" ]] \
    || die "calibration already exists under ${CALIBRATION_DIR}"
  mkdir -p "${CALIBRATION_DIR}" "${RAW_RECORDER_DIR}"
  build_calibration_server_command
  start_server "${CALIBRATION_DIR}" "${RAW_RECORDER_DIR}"

  curl -fsS -X POST "${BASE_URL}/start_expert_distribution_record" \
    > "${CALIBRATION_DIR}/recorder_start.txt"
  run_client \
    "${CALIBRATION_FIXTURE}" \
    "${CALIBRATION_DIR}/calibration_traffic.json" \
    "${CALIBRATION_CONCURRENCY}" \
    "${CALIBRATION_OUTPUT_TOKENS}" \
    > "${CALIBRATION_DIR}/calibration_traffic.log" 2>&1
  curl -fsS -X POST "${BASE_URL}/stop_expert_distribution_record" \
    > "${CALIBRATION_DIR}/recorder_stop.txt"
  curl -fsS -X POST "${BASE_URL}/dump_expert_distribution_record" \
    > "${CALIBRATION_DIR}/recorder_dump.txt"
  stop_server
  wait_for_clean_gpu

  local -a raw_stats=()
  mapfile -t raw_stats < <(
    find "${RAW_RECORDER_DIR}" -maxdepth 1 -type f \
      -name 'expert_distribution_recorder_*.pt' -print | sort
  )
  [[ "${#raw_stats[@]}" -eq 1 ]] \
    || die "expected one raw stat recorder file, got ${#raw_stats[@]}"

  PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    "${KT_PY}" "${PREPARE_STATS}" \
      --input "${raw_stats[0]}" \
      --output "${FREQUENCY_STATS}" \
      --manifest "${FREQUENCY_STATS_MANIFEST}" \
      --calibration-fixture "${CALIBRATION_FIXTURE}" \
      --evaluation-fixture "${EVALUATION_FIXTURE}" \
      --gpu-experts-ratio "${GPU_EXPERTS_RATIO}" \
      > "${CALIBRATION_DIR}/prepare_frequency_stats.stdout.json"
  jq -e \
    '.fixtures_disjoint_by_source_index == true and
     .binary_authoritative_mask_used_as_counts == false and
     .routing.total_routed_slots > 0' \
    "${FREQUENCY_STATS_MANIFEST}" >/dev/null \
    || die "frequency stats manifest failed integrity checks"
  echo "frequency_stats=${FREQUENCY_STATS}"
  echo "frequency_stats_manifest=${FREQUENCY_STATS_MANIFEST}"
}

build_qualification_server_command() {
  local mode="$1"
  local threads="$2"
  local -a profile_args=()
  local -a graph_args=()
  local -a placement_args=()
  local -a overlap_args=()
  local -a mixed_args=()
  case "${QUAL_DISABLE_CUDA_GRAPH}" in
    0)
      ;;
    1)
      graph_args=(--disable-cuda-graph)
      ;;
    *)
      die "QUAL_DISABLE_CUDA_GRAPH must be 0 or 1, got ${QUAL_DISABLE_CUDA_GRAPH}"
      ;;
  esac
  case "${QUAL_PLACEMENT_STRATEGY}" in
    frequency)
      placement_args=(
        --kt-expert-placement-strategy frequency
        --init-expert-location "${FREQUENCY_STATS}"
      )
      ;;
    uniform)
      placement_args=(--kt-expert-placement-strategy uniform)
      ;;
    *)
      die "QUAL_PLACEMENT_STRATEGY must be frequency or uniform"
      ;;
  esac
  case "${QUAL_DISABLE_OVERLAP_SCHEDULE}" in
    0) ;;
    1) overlap_args=(--disable-overlap-schedule) ;;
    *) die "QUAL_DISABLE_OVERLAP_SCHEDULE must be 0 or 1" ;;
  esac
  case "${QUAL_ENABLE_MIXED_CHUNK}" in
    0) ;;
    1) mixed_args=(--enable-mixed-chunk) ;;
    *) die "QUAL_ENABLE_MIXED_CHUNK must be 0 or 1" ;;
  esac
  if [[ "${mode}" == "exact" ]]; then
    profile_args=(
      --kt-max-deferred-experts-per-token 0
      --kt-gpu-prefill-token-threshold 0
    )
  elif [[ "${mode}" == "vendor-static" ]]; then
    profile_args=(
      --kt-max-deferred-experts-per-token 2
      --kt-gpu-prefill-token-threshold 1024
    )
  elif [[ "${mode}" == "vendor-best" ]]; then
    profile_args=(
      --kt-max-deferred-experts-per-token 2
      --kt-gpu-prefill-token-threshold 1024
      --kt-enable-dynamic-expert-update
    )
  else
    die "unknown qualification mode: ${mode}"
  fi

  SERVER_COMMAND=(
    taskset -c "${CPUSET}"
    "${KT_PY}" -m sglang.launch_server
    --model-path "${MODEL}"
    --host 127.0.0.1 --port "${PORT}"
    --tp-size 1
    --context-length "${MAX_CONTEXT_LENGTH}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --max-total-tokens "${KV_TOKENS}"
    --chunked-prefill-size "${QUAL_CHUNKED_PREFILL_SIZE}"
    --max-prefill-tokens "${QUAL_MAX_PREFILL_TOKENS}"
    --schedule-policy fcfs --stream-interval 1 --random-seed 0
    --trust-remote-code
    --kt-weight-path "${MODEL}"
    --kt-method FP8
    --kt-cpuinfer "${threads}" --kt-threadpool-count 1 --kt-numa-nodes 0
    --kt-gpu-experts-ratio "${GPU_EXPERTS_RATIO}"
    "${placement_args[@]}"
    "${profile_args[@]}"
    --attention-backend flashinfer --fp8-gemm-backend triton
    "${mixed_args[@]}"
    "${overlap_args[@]}"
    "${graph_args[@]}"
    --disable-shared-experts-fusion
    --disable-radix-cache
  )
}

start_monitor() {
  local case_dir="$1"
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader,nounits -lms 200 \
    > "${case_dir}/phase_nvml.csv" 2> "${case_dir}/phase_nvml.err" &
  MONITOR_PID="$!"
}

write_case_metadata() {
  local case_dir="$1"
  local mode="$2"
  local threads="$3"
  local semantics
  local cuda_graph_enabled=true
  local overlap_enabled=true
  local mixed_chunk_enabled=true
  if [[ "${QUAL_DISABLE_CUDA_GRAPH}" == "1" ]]; then
    cuda_graph_enabled=false
  fi
  if [[ "${QUAL_DISABLE_OVERLAP_SCHEDULE}" == "1" ]]; then
    overlap_enabled=false
  fi
  if [[ "${QUAL_ENABLE_MIXED_CHUNK}" == "0" ]]; then
    mixed_chunk_enabled=false
  fi
  if [[ "${mode}" == "exact" ]]; then
    semantics="real-frequency fixed placement; exact top-k contributions; cuda_graph=${cuda_graph_enabled}; overlap/mixed enabled"
  elif [[ "${mode}" == "vendor-static" ]]; then
    semantics="vendor performance candidate; deferral=2; layerwise threshold=1024; dynamic placement disabled; quality check required"
  else
    semantics="vendor performance candidate; deferral=2; layerwise threshold=1024; dynamic placement enabled; quality check required"
  fi
  jq -n \
    --arg mode "${mode}" \
    --argjson threads "${threads}" \
    --arg semantics "${semantics}" \
    --arg model "${MODEL}" \
    --arg stats "${FREQUENCY_STATS}" \
    --arg stats_sha256 "$(sha256sum "${FREQUENCY_STATS}" | awk '{print $1}')" \
    --arg evaluation_fixture "${EVALUATION_FIXTURE}" \
    --arg evaluation_sha256 "$(sha256sum "${EVALUATION_FIXTURE}" | awk '{print $1}')" \
    --arg kt_commit "$(git -C "${KT_ROOT}" rev-parse HEAD)" \
    --arg sglang_commit "$(git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD)" \
    --arg gpu_ratio "${GPU_EXPERTS_RATIO}" \
    --argjson kv_tokens "${KV_TOKENS}" \
    --argjson cuda_graph_enabled "${cuda_graph_enabled}" \
    --arg placement_strategy "${QUAL_PLACEMENT_STRATEGY}" \
    --argjson overlap_enabled "${overlap_enabled}" \
    --argjson mixed_chunk_enabled "${mixed_chunk_enabled}" \
    '{
      schema_version: 1,
      mode: $mode,
      cpu_threads: $threads,
      semantics: $semantics,
      model: $model,
      frequency_stats: {path: $stats, sha256: $stats_sha256},
      evaluation_fixture: {path: $evaluation_fixture, sha256: $evaluation_sha256},
      ktransformers_commit: $kt_commit,
      sglang_commit: $sglang_commit,
      gpu_experts_ratio_cli: $gpu_ratio,
      kv_tokens: $kv_tokens,
      cuda_graph_enabled: $cuda_graph_enabled,
      placement_strategy: $placement_strategy,
      overlap_enabled: $overlap_enabled,
      mixed_chunk_enabled: $mixed_chunk_enabled
    }' > "${case_dir}/case_metadata.json"
}

run_case() {
  local mode="$1"
  local threads="$2"
  [[ "${threads}" =~ ^(7|12|14|28)$ ]] \
    || die "threads must be 7, 12, 14, or 28, got ${threads}"
  preflight
  require_file "${EVALUATION_FIXTURE}"
  require_file "${CALIBRATION_FIXTURE}"
  require_file "${FREQUENCY_STATS}"
  require_file "${FREQUENCY_STATS_MANIFEST}"

  local case_dir="${RUN_ROOT}/cases/${CASE_LABEL:-${mode}_t${threads}}"
  local kernel_epoch journal_rc
  [[ ! -e "${case_dir}" ]] || die "qualification case already exists: ${case_dir}"
  mkdir -p "${case_dir}"
  kernel_epoch="$(date +%s)"
  build_qualification_server_command "${mode}" "${threads}"
  start_server "${case_dir}"

  run_client \
    "${CALIBRATION_FIXTURE}" \
    "${case_dir}/warmup.json" \
    "${WARM_CONCURRENCY}" \
    "${WARM_OUTPUT_TOKENS}" \
    > "${case_dir}/warmup.log" 2>&1
  start_monitor "${case_dir}"
  run_client \
    "${EVALUATION_FIXTURE}" \
    "${case_dir}/benchmark.json" \
    "${QUAL_CONCURRENCIES}" \
    "${QUAL_OUTPUT_TOKENS}" \
    > "${case_dir}/benchmark.log" 2>&1
  stop_monitor
  stop_server
  wait_for_clean_gpu
  set +e
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    > "${case_dir}/kernel.since.log" 2> "${case_dir}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${journal_rc}" > "${case_dir}/kernel_log_returncode.txt"
  (( journal_rc == 0 )) || die "kernel journal collection failed for ${case_dir}"
  if rg -qi \
      'NVRM: Xid|FAULT_PDE|manual[_ -]migration[_ -]fault|(^|[^[:alnum:]_])Oops|(^|[^[:alnum:]_])BUG:|kernel panic|general protection fault|Call Trace' \
      "${case_dir}/kernel.since.log"; then
    die "kernel log contains a fatal GPU/UVM signature for ${case_dir}"
  fi
  nvidia-smi --query-gpu=timestamp,memory.used \
    --format=csv,noheader,nounits > "${case_dir}/gpu_after.csv"
  write_case_metadata "${case_dir}" "${mode}" "${threads}"

  PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    "${KT_PY}" "${VERIFY_CASE}" \
      --mode "${mode}" \
      --threads "${threads}" \
      --cuda-graph "$([[ "${QUAL_DISABLE_CUDA_GRAPH}" == "1" ]] && echo disabled || echo enabled)" \
      --placement "${QUAL_PLACEMENT_STRATEGY}" \
      --overlap "$([[ "${QUAL_DISABLE_OVERLAP_SCHEDULE}" == "1" ]] && echo disabled || echo enabled)" \
      --mixed-chunk "$([[ "${QUAL_ENABLE_MIXED_CHUNK}" == "0" ]] && echo disabled || echo enabled)" \
      --command-file "${case_dir}/server_command.txt" \
      --server-log "${case_dir}/server.log" \
      --stats-manifest "${FREQUENCY_STATS_MANIFEST}" \
      --output "${case_dir}/runtime_verification.json" \
      > "${case_dir}/runtime_verification.stdout.json"
  echo "completed=${case_dir}"
}

summarize() {
  require_file "${FREQUENCY_STATS_MANIFEST}"
  PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    "${KT_PY}" "${SUMMARIZER}" "${RUN_ROOT}" \
      --output-json "${RUN_ROOT}/kt_qualification_snapshot.json" \
      --output-md "${RUN_ROOT}/KT_QUALIFICATION_RESULTS.md" \
      > "${RUN_ROOT}/summarize_kt_qualification.stdout.json"
  echo "snapshot=${RUN_ROOT}/kt_qualification_snapshot.json"
  echo "report=${RUN_ROOT}/KT_QUALIFICATION_RESULTS.md"
}

run_sweep() {
  [[ ! -e "${RUN_ROOT}/calibration" && ! -e "${RUN_ROOT}/cases" ]] \
    || die "RUN_ROOT already contains calibration/cases: ${RUN_ROOT}"
  prepare_fixtures
  record_calibration
  # ABBA-like ordering avoids always measuring both exact cases first.
  run_case exact 12
  run_case vendor-best 12
  run_case vendor-best 14
  run_case exact 14
  summarize
}

show_help() {
  cat <<'EOF'
Usage:
  run_kt_qualification.sh preflight
  run_kt_qualification.sh prepare-fixtures
  run_kt_qualification.sh record-calibration
  run_kt_qualification.sh run-case exact 12
  run_kt_qualification.sh run-case vendor-best 14
  run_kt_qualification.sh run-case vendor-static 12
  run_kt_qualification.sh sweep
  run_kt_qualification.sh summarize

Set one explicit RUN_ROOT when invoking separate commands. `sweep` creates the
disjoint fixtures, records real routing counts, freezes frequency placement,
runs exact/vendor-best with 12/14 CPU threads, and writes a summary.

Profiles:
  exact        Real frequency placement, deferral/dynamic/layerwise disabled;
               CUDA graph, overlap scheduling, and mixed chunk are enabled by
               default. Set QUAL_DISABLE_CUDA_GRAPH=1 for a graph-only
               ablation; overlap scheduling and mixed chunk remain enabled.
  vendor-best  Same initial real frequency placement, plus deferral=2,
               layerwise threshold=1024, and dynamic expert update. This mode
               is approximate/adaptive and requires a separate quality check.
  vendor-static Same approximate deferral/layerwise settings, with dynamic
                placement disabled, for the documentation-recommended on/off
                dynamic-update sweep.

No command is executed merely by printing this help. The runner never modifies
the existing pilot assets or source trees.
EOF
}

case "${COMMAND}" in
  preflight)
    preflight
    ;;
  prepare-fixtures)
    prepare_fixtures
    ;;
  record-calibration)
    record_calibration
    ;;
  run-case)
    [[ $# -eq 2 ]] || die "run-case requires MODE and THREADS"
    run_case "$1" "$2"
    ;;
  sweep)
    [[ $# -eq 0 ]] || die "sweep takes no positional arguments"
    run_sweep
    ;;
  summarize)
    summarize
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
