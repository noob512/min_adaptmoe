#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
PHASED_RUNNER="${VLLM_ROOT}/uvm_test/vllm_generate_phased.py"
PREPARE_FIXTURES="${SCRIPT_DIR}/prepare_sharegpt_fixtures.py"
BUILD_PLANS="${SCRIPT_DIR}/build_target_cold_plans.py"
SUMMARIZER="${SCRIPT_DIR}/summarize_custom_cold_snapshot.py"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
NATIVE_SNAPSHOT="${NATIVE_SNAPSHOT:-${REPO_ROOT}/experiments/native_uva_qwen_sharegpt_concurrency_snapshot/snapshot_20260812.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen_custom_cold_sharegpt_concurrency_snapshot_$(date +%Y%m%d_%H%M%S)}"
UVM_MODULE="${UVM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"

TARGETS_GIB="${TARGETS_GIB:-0,2,6,10,14,18}"
CONCURRENCIES="${CONCURRENCIES:-1,2,3,4,8,16,24,32}"
REQUEST_COUNT="${REQUEST_COUNT:-32}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS:-8}"
PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS:-300}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-320}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-1288490189}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
RUN_CASES="${RUN_CASES:-pilot,baseline_asc,g2,g6,g10,baseline_desc,g14,g18}"
SUMMARIZE="${SUMMARIZE:-1}"

FIXTURE_DIR="${RUN_ROOT}/fixtures"
PLAN_DIR="${RUN_ROOT}/plans"
EVALUATION_FIXTURE="${FIXTURE_DIR}/evaluation_fixture.json"
CALIBRATION_FIXTURE="${FIXTURE_DIR}/calibration_fixture.json"

mkdir -p "${RUN_ROOT}" "${FIXTURE_DIR}" "${PLAN_DIR}"

should_run_case() {
  local name="$1"
  [[ ",${RUN_CASES}," == *",${name},"* ]]
}

wait_for_clean_gpu() {
  local used_mib
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= 128 )); then
      return 0
    fi
    sleep 2
  done
  echo "GPU did not become clean: ${used_mib} MiB" >&2
  nvidia-smi >&2 || true
  return 1
}

verify_driver() {
  local expected_srcversion loaded_srcversion
  expected_srcversion="$(modinfo -F srcversion "${UVM_MODULE}")"
  loaded_srcversion="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  if [[ -z "${expected_srcversion}" || "${loaded_srcversion}" != "${expected_srcversion}" ]]; then
    echo "nvidia_uvm mismatch: loaded=${loaded_srcversion:-missing} expected=${expected_srcversion:-missing}" >&2
    return 1
  fi
  "${VLLM_PY}" - "${RUN_ROOT}/driver_metadata.json" "${UVM_MODULE}" \
    "${loaded_srcversion}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
module = Path(sys.argv[2])
loaded = sys.argv[3]
metadata = {
    "module": str(module),
    "loaded_srcversion": loaded,
    "module_srcversion": subprocess.check_output(
        ["modinfo", "-F", "srcversion", str(module)], text=True
    ).strip(),
    "module_version": subprocess.check_output(
        ["modinfo", "-F", "version", str(module)], text=True
    ).strip(),
    "uname_release": subprocess.check_output(["uname", "-r"], text=True).strip(),
}
output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY
}

prepare_fixtures() {
  if [[ -s "${EVALUATION_FIXTURE}" && -s "${CALIBRATION_FIXTURE}" ]]; then
    return 0
  fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${VLLM_PY}" \
    "${PREPARE_FIXTURES}" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --reference-snapshot "${NATIVE_SNAPSHOT}" \
    --output-dir "${FIXTURE_DIR}" \
    --request-count "${REQUEST_COUNT}" \
    --prompt-min-tokens "${PROMPT_MIN_TOKENS}" \
    --prompt-max-tokens "${PROMPT_MAX_TOKENS}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    > "${RUN_ROOT}/prepare_fixtures.log"
}

build_eval_phases() {
  local order="$1"
  local -n names_ref="$2"
  local -n counts_ref="$3"
  local -n outputs_ref="$4"
  local -a ascending=()
  local -a ordered=()
  IFS=',' read -r -a ascending <<< "${CONCURRENCIES}"
  if [[ "${order}" == "asc" ]]; then
    ordered=("${ascending[@]}")
  else
    for ((index=${#ascending[@]} - 1; index >= 0; index--)); do
      ordered+=("${ascending[index]}")
    done
  fi
  names_ref=("warm_c${REQUEST_COUNT}")
  counts_ref=("${REQUEST_COUNT}")
  outputs_ref=("${WARM_OUTPUT_TOKENS}")
  for concurrency in "${ordered[@]}"; do
    names_ref+=("sharegpt_c${concurrency}")
    counts_ref+=("${concurrency}")
    outputs_ref+=("${OUTPUT_TOKENS}")
  done
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

run_case() {
  local name="$1"
  local mode="$2"
  local fixture="$3"
  local plan="$4"
  local order="$5"
  local case_root="${RUN_ROOT}/${name}"
  local monitor_pid=""
  local -a phase_names=()
  local -a phase_counts=()
  local -a phase_outputs=()
  local -a env_args=(
    VLLM_USE_UVM=0
    VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0
    VLLM_UVM_KV_CACHE_MANAGED=0
    VLLM_UVM_KV_RUNTIME_ENABLE=0
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
  )

  if [[ "${mode}" == "pilot" ]]; then
    phase_names=(calibration_pilot)
    phase_counts=("${REQUEST_COUNT}")
    phase_outputs=("${OUTPUT_TOKENS}")
    env_args+=(
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=0
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1
      VLLM_UVM_MOE_ROUTING_TRACE_FILE="${case_root}/routing_trace.jsonl"
    )
  else
    build_eval_phases "${order}" phase_names phase_counts phase_outputs
    env_args+=(
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
      VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}"
      VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
      VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
      VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
      VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
    )
  fi

  mkdir -p "${case_root}"
  printf '%s\n' "${env_args[@]}" > "${case_root}/environment.txt"
  printf '%s\n' "${order}" > "${case_root}/scan_order.txt"
  if [[ -n "${plan}" ]]; then
    sha256sum "${plan}" > "${case_root}/plan.sha256"
  fi

  wait_for_clean_gpu
  nvidia-smi dmon -s t -d 1 -o DT > "${case_root}/dmon_pcie.csv" 2>&1 &
  monitor_pid=$!
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
      "${PHASED_RUNNER}" \
      --model "${MODEL}" \
      --fixture "${fixture}" \
      --output "${case_root}/result.json" \
      --progress "${case_root}/progress.jsonl" \
      --phase-names "$(join_by_comma "${phase_names[@]}")" \
      --phase-request-counts "$(join_by_comma "${phase_counts[@]}")" \
      --phase-output-tokens "$(join_by_comma "${phase_outputs[@]}")" \
      --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
      --cpu-offload-gb 0 \
      --max-model-len "${MAX_MODEL_LEN}" \
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
    tail -n 200 "${case_root}/run.log" >&2 || true
    return "${rc}"
  fi
  wait_for_clean_gpu
}

verify_driver
prepare_fixtures

if should_run_case pilot; then
  run_case pilot pilot "${CALIBRATION_FIXTURE}" "" calibration
fi

if [[ ! -s "${RUN_ROOT}/pilot/routing_trace.jsonl" ]]; then
  echo "missing pilot routing trace: ${RUN_ROOT}/pilot/routing_trace.jsonl" >&2
  exit 1
fi
"${VLLM_PY}" "${BUILD_PLANS}" \
  --routing-trace "${RUN_ROOT}/pilot/routing_trace.jsonl" \
  --pilot-progress "${RUN_ROOT}/pilot/progress.jsonl" \
  --output-dir "${PLAN_DIR}" \
  --targets-gib "${TARGETS_GIB}" \
  > "${RUN_ROOT}/build_plans.log"

if should_run_case baseline_asc; then
  run_case baseline_asc custom "${EVALUATION_FIXTURE}" "${PLAN_DIR}/plan_g0.json" asc
fi
if should_run_case g2; then
  run_case g2 custom "${EVALUATION_FIXTURE}" "${PLAN_DIR}/plan_g2.json" asc
fi
if should_run_case g6; then
  run_case g6 custom "${EVALUATION_FIXTURE}" "${PLAN_DIR}/plan_g6.json" asc
fi
if should_run_case g10; then
  run_case g10 custom "${EVALUATION_FIXTURE}" "${PLAN_DIR}/plan_g10.json" asc
fi
if should_run_case baseline_desc; then
  run_case baseline_desc custom "${EVALUATION_FIXTURE}" "${PLAN_DIR}/plan_g0.json" desc
fi
if should_run_case g14; then
  run_case g14 custom "${EVALUATION_FIXTURE}" "${PLAN_DIR}/plan_g14.json" desc
fi
if should_run_case g18; then
  run_case g18 custom "${EVALUATION_FIXTURE}" "${PLAN_DIR}/plan_g18.json" desc
fi

if [[ "${SUMMARIZE}" == "1" ]]; then
  "${VLLM_PY}" "${SUMMARIZER}" \
    "${RUN_ROOT}" --native-snapshot "${NATIVE_SNAPSHOT}" \
    | tee "${RUN_ROOT}/snapshot_summary.stdout.json"
fi
echo "RUN_ROOT=${RUN_ROOT}"
