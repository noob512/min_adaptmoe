#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
DECODE_RUNNER="${VLLM_ROOT}/uvm_test/vllm_decode_step_benchmark.py"
PREPARE_FIXTURE="${SCRIPT_DIR}/prepare_fixture.py"
SUMMARIZER="${SCRIPT_DIR}/summarize.py"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_copy_decode_crossover_$(date +%Y%m%d_%H%M%S)}"
SNAPSHOT_OUTPUT="${SNAPSHOT_OUTPUT:-${RUN_ROOT}/snapshot.json}"

REPEATS="${REPEATS:-3}"
FIRST_REPEAT="${FIRST_REPEAT:-1}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64}"
OFFLOAD_GIB="${OFFLOAD_GIB:-2.0}"
WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-5}"
MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-16}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-268435456}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

FIXTURE="${RUN_ROOT}/fixture.json"
FIXTURE_METADATA="${RUN_ROOT}/fixture_metadata.json"
mkdir -p "${RUN_ROOT}"

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

prepare_fixture() {
  if [[ -s "${FIXTURE}" && -s "${FIXTURE_METADATA}" ]]; then
    return 0
  fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${VLLM_PY}" \
    "${PREPARE_FIXTURE}" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --output "${FIXTURE}" \
    --metadata "${FIXTURE_METADATA}" \
    --request-count "${MAX_NUM_SEQS}" \
    --prompt-min-tokens 1 \
    --prompt-max-tokens 8 \
    > "${RUN_ROOT}/prepare_fixture.log"
}

run_case() {
  local repeat="$1"
  local label="$2"
  local backend="$3"
  local offload_gib="$4"
  local scan_order="$5"
  local ordered_batches="${BATCH_SIZES}"
  local case_root="${RUN_ROOT}/r${repeat}_${label}"
  local monitor_pid=""

  if [[ "${scan_order}" == "desc" ]]; then
    ordered_batches="$("${VLLM_PY}" - "${BATCH_SIZES}" <<'PY'
import sys
print(",".join(reversed([part for part in sys.argv[1].split(",") if part])))
PY
)"
  fi
  if [[ -e "${case_root}" ]]; then
    echo "case directory already exists: ${case_root}" >&2
    return 2
  fi
  mkdir -p "${case_root}"
  printf '%s\n' "${scan_order}" > "${case_root}/scan_order.txt"
  printf '%s\n' "${ordered_batches}" > "${case_root}/batch_order.txt"
  printf '%s\n' "${backend}" > "${case_root}/backend.txt"
  printf '%s\n' "${offload_gib}" > "${case_root}/cpu_offload_gib.txt"

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
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND="${backend}"
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_root}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_CPU_OFFLOAD_COPY_TRACE=0
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_LOG_BATCHSIZE_INTERVAL=-1
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
  )
  printf '%s\n' "${env_args[@]}" > "${case_root}/environment.txt"

  wait_for_clean_gpu
  nvidia-smi dmon -s t -d 1 -o DT > "${case_root}/dmon_pcie.csv" 2>&1 &
  monitor_pid=$!
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
      "${DECODE_RUNNER}" \
      --model "${MODEL}" \
      --fixture "${FIXTURE}" \
      --output "${case_root}/result.json" \
      --progress "${case_root}/progress.jsonl" \
      --batch-sizes "${ordered_batches}" \
      --warmup-decode-steps "${WARMUP_DECODE_STEPS}" \
      --measured-decode-steps "${MEASURED_DECODE_STEPS}" \
      --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
      --cpu-offload-gb "${offload_gib}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
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

run_repeat() {
  local repeat="$1"
  local scan_order=asc
  if (( repeat % 2 == 0 )); then
    scan_order=desc
  fi
  case $(((repeat - 1) % 3)) in
    0)
      run_case "${repeat}" baseline uva 0.0 "${scan_order}"
      run_case "${repeat}" uva uva "${OFFLOAD_GIB}" "${scan_order}"
      run_case "${repeat}" copy copy "${OFFLOAD_GIB}" "${scan_order}"
      ;;
    1)
      run_case "${repeat}" uva uva "${OFFLOAD_GIB}" "${scan_order}"
      run_case "${repeat}" copy copy "${OFFLOAD_GIB}" "${scan_order}"
      run_case "${repeat}" baseline uva 0.0 "${scan_order}"
      ;;
    2)
      run_case "${repeat}" copy copy "${OFFLOAD_GIB}" "${scan_order}"
      run_case "${repeat}" baseline uva 0.0 "${scan_order}"
      run_case "${repeat}" uva uva "${OFFLOAD_GIB}" "${scan_order}"
      ;;
  esac
}

prepare_fixture
for repeat in $(seq "${FIRST_REPEAT}" "${REPEATS}"); do
  run_repeat "${repeat}"
done

"${VLLM_PY}" "${SUMMARIZER}" "${RUN_ROOT}" \
  --output "${SNAPSHOT_OUTPUT}" \
  | tee "${RUN_ROOT}/snapshot.stdout.json"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SNAPSHOT_OUTPUT=${SNAPSHOT_OUTPUT}"
