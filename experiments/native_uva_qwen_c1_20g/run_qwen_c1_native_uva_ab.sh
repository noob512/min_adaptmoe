#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen_native_uva_c1_20g_$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
REQUEST_COUNT="${REQUEST_COUNT:-32}"
WARM_REQUESTS="${WARM_REQUESTS:-4}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS:-320}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-360}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-268435456}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

mkdir -p "${RUN_ROOT}"

wait_for_clean_gpu() {
  local used_mib
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && [[ "${used_mib}" -le 1024 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "GPU did not become clean: ${used_mib} MiB" >&2
  return 1
}

run_one() {
  local repeat="$1"
  local label="$2"
  local offload_gib="$3"
  local case_root="${RUN_ROOT}/r${repeat}_${label}"
  local monitor_pid=""

  mkdir -p "${case_root}"
  wait_for_clean_gpu
  nvidia-smi dmon -s t -d 1 -o DT > "${case_root}/dmon_pcie.csv" 2>&1 &
  monitor_pid=$!

  echo "RUN_BEGIN repeat=${repeat} label=${label} offload_gib=${offload_gib}"
  set +e
  RUN_ROOT="${case_root}" \
  MODEL="${MODEL}" \
  REQUEST_COUNT="${REQUEST_COUNT}" \
  PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS}" \
  PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS}" \
  OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
  LARGE_KV_BYTES="${KV_CACHE_BYTES}" \
  MAX_MODEL_LEN=512 \
  MAX_NUM_SEQS=1 \
  MAX_NUM_BATCHED_TOKENS=2048 \
  GPU_MEMORY_UTILIZATION=0.97 \
  NATIVE_CPU_OFFLOAD_GB="${offload_gib}" \
  NATIVE_PHASE_NAMES=warm,measure \
  NATIVE_PHASE_COUNTS="${WARM_REQUESTS},${REQUEST_COUNT}" \
  NATIVE_PHASE_OUTPUTS="${OUTPUT_TOKENS},${OUTPUT_TOKENS}" \
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
    tail -n 100 "${case_root}/orchestrator.log" >&2 || true
    return 1
  fi
  wait_for_clean_gpu
}

for repeat in $(seq 1 "${REPEATS}"); do
  if (( repeat % 2 == 1 )); then
    run_one "${repeat}" baseline 0.0
    run_one "${repeat}" offload20 20.0
  else
    run_one "${repeat}" offload20 20.0
    run_one "${repeat}" baseline 0.0
  fi
done

"${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python" \
  "${SCRIPT_DIR}/summarize_qwen_c1_native_uva_ab.py" "${RUN_ROOT}" |
  tee "${RUN_ROOT}/comparison_summary.stdout.json"

echo "RUN_ROOT=${RUN_ROOT}"
