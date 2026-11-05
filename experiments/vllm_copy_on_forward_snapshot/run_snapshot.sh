#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
PHASED_RUNNER="${VLLM_ROOT}/uvm_test/vllm_generate_phased.py"
PREPARE_FIXTURES="${REPO_ROOT}/experiments/custom_cold_qwen_sharegpt_concurrency_snapshot/prepare_sharegpt_fixtures.py"
SUMMARIZER="${SCRIPT_DIR}/summarize.py"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
REFERENCE_SNAPSHOT="${REFERENCE_SNAPSHOT:-${REPO_ROOT}/experiments/native_uva_qwen_sharegpt_concurrency_snapshot/snapshot_20260812.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_copy_on_forward_$(date +%Y%m%d_%H%M%S)}"
SNAPSHOT_OUTPUT="${SNAPSHOT_OUTPUT:-${RUN_ROOT}/snapshot.json}"

OFFLOAD_GIB="${OFFLOAD_GIB:-2.0}"
REQUEST_COUNT="${REQUEST_COUNT:-4}"
FIXTURE_REQUEST_COUNT="${FIXTURE_REQUEST_COUNT:-32}"
EXPECTED_PROMPT_TOKENS="${EXPECTED_PROMPT_TOKENS:-1217}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-1288490189}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
RUN_CASES="${RUN_CASES:-baseline,uva,copy}"

FIXTURE_DIR="${RUN_ROOT}/fixtures"
EVALUATION_FIXTURE="${FIXTURE_DIR}/evaluation_fixture.json"
mkdir -p "${RUN_ROOT}" "${FIXTURE_DIR}"

should_run_case() {
  local name="$1"
  [[ ",${RUN_CASES}," == *",${name},"* ]]
}

wait_for_clean_gpu() {
  local used_mib=""
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= 128 )); then
      return 0
    fi
    sleep 2
  done
  echo "GPU did not become clean: ${used_mib:-unknown} MiB" >&2
  return 1
}

prepare_fixtures() {
  if [[ -s "${EVALUATION_FIXTURE}" ]]; then
    return 0
  fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${VLLM_PY}" \
    "${PREPARE_FIXTURES}" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --reference-snapshot "${REFERENCE_SNAPSHOT}" \
    --output-dir "${FIXTURE_DIR}" \
    --request-count "${FIXTURE_REQUEST_COUNT}" \
    --prompt-min-tokens 300 \
    --prompt-max-tokens 320 \
    --output-tokens 1 \
    > "${RUN_ROOT}/prepare_fixtures.log"
}

run_case() {
  local name="$1"
  local backend="$2"
  local offload_gib="$3"
  local case_root="${RUN_ROOT}/${name}"
  local monitor_pid=""
  local -a env_args=(
    VLLM_USE_UVM=0
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
    VLLM_UVM_KV_CACHE_MANAGED=0
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_KV_RUNTIME_ENABLE=0
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND="${backend}"
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_root}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_CPU_OFFLOAD_COPY_TRACE=0
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
  )
  if [[ "${backend}" == "copy" ]]; then
    env_args+=(
      VLLM_CPU_OFFLOAD_COPY_TRACE=1
      VLLM_CPU_OFFLOAD_COPY_SYNC=1
    )
  fi

  if [[ -e "${case_root}" ]]; then
    echo "case directory already exists: ${case_root}" >&2
    return 2
  fi
  mkdir -p "${case_root}"
  printf '%s\n' "${env_args[@]}" > "${case_root}/environment.txt"
  printf '%s\n' "${offload_gib}" > "${case_root}/cpu_offload_gib.txt"

  wait_for_clean_gpu
  nvidia-smi dmon -s t -d 1 -o DT > "${case_root}/dmon_pcie.csv" 2>&1 &
  monitor_pid=$!
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
      "${PHASED_RUNNER}" \
      --model "${MODEL}" \
      --fixture "${EVALUATION_FIXTURE}" \
      --output "${case_root}/result.json" \
      --progress "${case_root}/progress.jsonl" \
      --phase-names warm,eval1,eval2,eval3,eval4,eval5 \
      --phase-request-counts "${REQUEST_COUNT},${REQUEST_COUNT},${REQUEST_COUNT},${REQUEST_COUNT},${REQUEST_COUNT},${REQUEST_COUNT}" \
      --phase-output-tokens 1,1,1,1,1,1 \
      --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
      --cpu-offload-gb "${offload_gib}" \
      --max-model-len 512 \
      --max-num-seqs "${REQUEST_COUNT}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  ) > "${case_root}/run.log" 2>&1
  local rc=$?
  set -e
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  printf '%s\n' "${rc}" > "${case_root}/returncode.txt"
  if [[ "${rc}" -ne 0 ]]; then
    tail -n 240 "${case_root}/run.log" >&2 || true
    return "${rc}"
  fi
  wait_for_clean_gpu
}

prepare_fixtures
if should_run_case baseline; then
  run_case baseline uva 0.0
fi
if should_run_case uva; then
  run_case uva uva "${OFFLOAD_GIB}"
fi
if should_run_case copy; then
  run_case copy copy "${OFFLOAD_GIB}"
fi

if [[ -s "${RUN_ROOT}/baseline/result.json" \
   && -s "${RUN_ROOT}/uva/result.json" \
   && -s "${RUN_ROOT}/copy/result.json" ]]; then
  "${VLLM_PY}" "${SUMMARIZER}" "${RUN_ROOT}" \
    --output "${SNAPSHOT_OUTPUT}" \
    | tee "${RUN_ROOT}/snapshot.stdout.json"
fi
echo "RUN_ROOT=${RUN_ROOT}"
echo "SNAPSHOT_OUTPUT=${SNAPSHOT_OUTPUT}"
