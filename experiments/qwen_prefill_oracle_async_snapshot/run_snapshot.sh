#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
PHASED_RUNNER="${VLLM_ROOT}/uvm_test/vllm_generate_phased.py"
PREPARE_FIXTURES="${REPO_ROOT}/experiments/custom_cold_qwen_sharegpt_concurrency_snapshot/prepare_sharegpt_fixtures.py"
BUILD_PLAN="${SCRIPT_DIR}/build_oracle_plan.py"
SUMMARIZER="${SCRIPT_DIR}/summarize.py"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
REFERENCE_SNAPSHOT="${REFERENCE_SNAPSHOT:-${REPO_ROOT}/experiments/native_uva_qwen_sharegpt_concurrency_snapshot/snapshot_20260812.json}"
UVM_MODULE="${UVM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen_prefill_oracle_async_$(date +%Y%m%d_%H%M%S)}"
SNAPSHOT_OUTPUT="${SNAPSHOT_OUTPUT:-${RUN_ROOT}/snapshot.json}"

REQUEST_COUNT="${REQUEST_COUNT:-4}"
FIXTURE_REQUEST_COUNT="${FIXTURE_REQUEST_COUNT:-32}"
EXPERTS_PER_LAYER="${EXPERTS_PER_LAYER:-9}"
EXPECTED_PROMPT_TOKENS="${EXPECTED_PROMPT_TOKENS:-1217}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-1288490189}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
RUN_CASES="${RUN_CASES:-pilot,baseline,remote,sync,async}"

FIXTURE_DIR="${RUN_ROOT}/fixtures"
PLAN_DIR="${RUN_ROOT}/plans"
EVALUATION_FIXTURE="${FIXTURE_DIR}/evaluation_fixture.json"
ORACLE_PLAN="${PLAN_DIR}/oracle_plan.json"
EMPTY_PLAN="${PLAN_DIR}/empty_plan.json"

mkdir -p "${RUN_ROOT}" "${FIXTURE_DIR}" "${PLAN_DIR}"

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

verify_driver() {
  local loaded_srcversion module_srcversion
  loaded_srcversion="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  module_srcversion="$(modinfo -F srcversion "${UVM_MODULE}")"
  if [[ -z "${loaded_srcversion}" || "${loaded_srcversion}" != "${module_srcversion}" ]]; then
    echo "nvidia_uvm mismatch: loaded=${loaded_srcversion:-missing} module=${module_srcversion:-missing}" >&2
    return 1
  fi
  "${VLLM_PY}" - "${RUN_ROOT}/driver_metadata.json" "${UVM_MODULE}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
module = Path(sys.argv[2])
metadata = {
    "module": str(module),
    "loaded_srcversion": Path("/sys/module/nvidia_uvm/srcversion").read_text().strip(),
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
  local kind="$2"
  local plan="$3"
  local case_root="${RUN_ROOT}/${name}"
  local monitor_pid=""
  local phase_name="prefill_eval"
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

  if [[ "${kind}" == "pilot" ]]; then
    phase_name="prefill_oracle_pilot"
    env_args+=(
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=0
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
      VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE=off
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1
      VLLM_UVM_MOE_ROUTING_TRACE_FILE="${case_root}/routing_trace.jsonl"
    )
  else
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
    if [[ "${kind}" == "sync" || "${kind}" == "async" ]]; then
      env_args+=(
        VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE="${kind}"
        VLLM_UVM_PREFILL_ORACLE_MIN_TOKENS=2
        VLLM_UVM_PREFILL_ORACLE_FAIL_FAST=1
        VLLM_UVM_PREFILL_ORACLE_TRACE_FILE="${case_root}/oracle_trace.jsonl"
      )
    else
      env_args+=(VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE=off)
    fi
  fi

  mkdir -p "${case_root}"
  printf '%s\n' "${env_args[@]}" > "${case_root}/environment.txt"
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
      --fixture "${EVALUATION_FIXTURE}" \
      --output "${case_root}/result.json" \
      --progress "${case_root}/progress.jsonl" \
      --phase-names "${phase_name}" \
      --phase-request-counts "${REQUEST_COUNT}" \
      --phase-output-tokens 1 \
      --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
      --cpu-offload-gb 0 \
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

verify_driver
prepare_fixtures

if should_run_case pilot; then
  run_case pilot pilot ""
fi
if [[ ! -s "${RUN_ROOT}/pilot/routing_trace.jsonl" ]]; then
  echo "missing pilot routing trace" >&2
  exit 1
fi
"${VLLM_PY}" "${BUILD_PLAN}" \
  --routing-trace "${RUN_ROOT}/pilot/routing_trace.jsonl" \
  --pilot-progress "${RUN_ROOT}/pilot/progress.jsonl" \
  --output-dir "${PLAN_DIR}" \
  --experts-per-layer "${EXPERTS_PER_LAYER}" \
  --expected-num-tokens "${EXPECTED_PROMPT_TOKENS}" \
  > "${RUN_ROOT}/build_plan.log"

if should_run_case baseline; then
  run_case baseline baseline "${EMPTY_PLAN}"
fi
if should_run_case remote; then
  run_case remote remote "${ORACLE_PLAN}"
fi
if should_run_case sync; then
  run_case sync sync "${ORACLE_PLAN}"
fi
if should_run_case async; then
  run_case async async "${ORACLE_PLAN}"
fi

if [[ -s "${RUN_ROOT}/baseline/result.json" \
   && -s "${RUN_ROOT}/remote/result.json" \
   && -s "${RUN_ROOT}/sync/result.json" \
   && -s "${RUN_ROOT}/async/result.json" ]]; then
  "${VLLM_PY}" "${SUMMARIZER}" "${RUN_ROOT}" \
    --output "${SNAPSHOT_OUTPUT}" \
    | tee "${RUN_ROOT}/snapshot.stdout.json"
fi
echo "RUN_ROOT=${RUN_ROOT}"
echo "SNAPSHOT_OUTPUT=${SNAPSHOT_OUTPUT}"
