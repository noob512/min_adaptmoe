#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen_native_uva_forward_tokens_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
FIRST_REPEAT="${FIRST_REPEAT:-1}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS:-8}"
PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS:-1}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-8}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-268435456}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
ENGINE_MULTIPROCESSING="${ENGINE_MULTIPROCESSING:-0}"

IFS=',' read -r -a ASC_BATCHES <<< "${BATCH_SIZES}"
if [[ "${#ASC_BATCHES[@]}" -eq 0 ]]; then
  echo "BATCH_SIZES must not be empty" >&2
  exit 2
fi

MAX_BATCH=0
for batch in "${ASC_BATCHES[@]}"; do
  if [[ ! "${batch}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid batch size: ${batch}" >&2
    exit 2
  fi
  if (( batch > MAX_BATCH )); then
    MAX_BATCH="${batch}"
  fi
done

if (( PROMPT_MAX_TOKENS + OUTPUT_TOKENS > MAX_MODEL_LEN )); then
  echo "prompt + output exceeds MAX_MODEL_LEN" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"

join_by_comma() {
  local IFS=','
  echo "$*"
}

wait_for_clean_gpu() {
  local used_mib
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && [[ "${used_mib}" -le 128 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "GPU did not become clean: ${used_mib} MiB" >&2
  nvidia-smi >&2 || true
  return 1
}

build_phases() {
  local order="$1"
  local -n names_ref="$2"
  local -n counts_ref="$3"
  local -n outputs_ref="$4"
  local ordered=()

  if [[ "${order}" == "asc" ]]; then
    ordered=("${ASC_BATCHES[@]}")
  else
    for ((index=${#ASC_BATCHES[@]} - 1; index >= 0; index--)); do
      ordered+=("${ASC_BATCHES[index]}")
    done
  fi

  names_ref=(warm_b${MAX_BATCH})
  counts_ref=("${MAX_BATCH}")
  outputs_ref=("${WARM_OUTPUT_TOKENS}")
  for batch in "${ordered[@]}"; do
    names_ref+=("decode_b${batch}")
    counts_ref+=("${batch}")
    outputs_ref+=("${OUTPUT_TOKENS}")
  done
}

run_one() {
  local repeat="$1"
  local label="$2"
  local offload_gib="$3"
  local order="$4"
  local case_root="${RUN_ROOT}/r${repeat}_${label}"
  local monitor_pid=""
  local phase_names=()
  local phase_counts=()
  local phase_outputs=()

  build_phases "${order}" phase_names phase_counts phase_outputs
  mkdir -p "${case_root}"
  printf '%s\n' "${order}" > "${case_root}/scan_order.txt"
  printf '%s\n' "${ENGINE_MULTIPROCESSING}" > "${case_root}/engine_multiprocessing.txt"
  wait_for_clean_gpu
  nvidia-smi dmon -s t -d 1 -o DT > "${case_root}/dmon_pcie.csv" 2>&1 &
  monitor_pid=$!

  echo "RUN_BEGIN repeat=${repeat} label=${label} order=${order} offload_gib=${offload_gib}"
  set +e
  RUN_ROOT="${case_root}" \
  MODEL="${MODEL}" \
  REQUEST_COUNT="${MAX_BATCH}" \
  PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS}" \
  PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS}" \
  OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
  FIXTURE_OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
  LARGE_KV_BYTES="${KV_CACHE_BYTES}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_SEQS="${MAX_BATCH}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  GPU_MEMORY_UTILIZATION=0.97 \
  VLLM_ENABLE_V1_MULTIPROCESSING="${ENGINE_MULTIPROCESSING}" \
  NATIVE_CPU_OFFLOAD_GB="${offload_gib}" \
  NATIVE_PHASE_NAMES="$(join_by_comma "${phase_names[@]}")" \
  NATIVE_PHASE_COUNTS="$(join_by_comma "${phase_counts[@]}")" \
  NATIVE_PHASE_OUTPUTS="$(join_by_comma "${phase_outputs[@]}")" \
  TIMEOUT_SEC="${TIMEOUT_SEC}" \
  RUN_CASES=native_uva \
  bash "${RUNNER}" > "${case_root}/orchestrator.log" 2>&1
  local runner_rc=$?
  set -e

  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  monitor_pid=""

  local case_rc=missing
  if [[ -f "${case_root}/native_uva/returncode.txt" ]]; then
    case_rc="$(tr -d '[:space:]' < "${case_root}/native_uva/returncode.txt")"
  fi
  echo "RUN_END repeat=${repeat} label=${label} runner_rc=${runner_rc} case_rc=${case_rc}"
  if [[ "${runner_rc}" -ne 0 || "${case_rc}" != "0" ]]; then
    tail -n 120 "${case_root}/orchestrator.log" >&2 || true
    return 1
  fi
  wait_for_clean_gpu
}

for repeat in $(seq "${FIRST_REPEAT}" "${REPEATS}"); do
  if (( repeat % 2 == 1 )); then
    scan_order=asc
    run_one "${repeat}" baseline 0.0 "${scan_order}"
    run_one "${repeat}" offload20 20.0 "${scan_order}"
  else
    scan_order=desc
    run_one "${repeat}" offload20 20.0 "${scan_order}"
    run_one "${repeat}" baseline 0.0 "${scan_order}"
  fi
done

"${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python" \
  "${SCRIPT_DIR}/summarize_forward_token_scan.py" "${RUN_ROOT}" |
  tee "${RUN_ROOT}/comparison_summary.stdout.json"

echo "RUN_ROOT=${RUN_ROOT}"
