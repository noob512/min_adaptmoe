#!/usr/bin/env bash
set -euo pipefail

PATH_PARITY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATH_PARITY_REPO_ROOT="$(cd "${PATH_PARITY_SCRIPT_DIR}/../.." && pwd)"
PATH_PARITY_BASE_HARNESS="${PATH_PARITY_REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${PATH_PARITY_BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

SCRIPT_DIR="${PATH_PARITY_SCRIPT_DIR}"
REPO_ROOT="${PATH_PARITY_REPO_ROOT}"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py"
VERIFY="${SCRIPT_DIR}/verify_experiment.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${SCRIPT_DIR}/inputs/qwen3_b8_33token_fixture.json}"
NAMES="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/exact_parameter_names.txt"
STATIC_PLAN="${REPO_ROOT}/experiments/vllm_uvm_physical_kv_pool_resize/inputs/cold_decode_plan_6g.json"
SUPERSET_PLAN="${REPO_ROOT}/experiments/vllm_uvm_physical_kv_pool_resize/inputs/cold_decode_plan_7g.json"
INCREMENTAL_PLAN="${REPO_ROOT}/experiments/vllm_uvm_three_queue_e2e/inputs/incremental_wave_plan.json"
KV_MAX_BYTES="${KV_MAX_BYTES:-1610612736}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
CPUSET="${CPUSET:-0-13}"

die_path_parity() {
  echo "KV/expert path parity: $*" >&2
  exit 1
}

require_path_parity_file() {
  [[ -f "$1" ]] || die_path_parity "missing file: $1"
}

static_preflight_path_parity() {
  local path
  for path in "${RUNNER}" "${VERIFY}" "${MODEL}/config.json" "${FIXTURE}" \
    "${NAMES}" "${STATIC_PLAN}" "${SUPERSET_PLAN}" "${INCREMENTAL_PLAN}"; do
    require_path_parity_file "${path}"
  done
  [[ "$(wc -l <"${NAMES}")" -eq 96 ]] \
    || die_path_parity "expected 96 exact parameter names"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY}"
  "${VLLM_PY}" - "${FIXTURE}" "${STATIC_PLAN}" "${SUPERSET_PLAN}" \
    "${INCREMENTAL_PLAN}" <<'PY'
import json
import sys

fixture, static_path, superset_path, incremental_path = sys.argv[1:]
requests = json.load(open(fixture, encoding="utf-8"))["requests"]
assert len(requests) == 8
assert all(len(item["prompt_token_ids"]) == 33 for item in requests)
assert len({tuple(item["prompt_token_ids"]) for item in requests}) == 1
static = json.load(open(static_path, encoding="utf-8"))["offload_plan"]
superset = json.load(open(superset_path, encoding="utf-8"))["offload_plan"]
wave = json.load(open(incremental_path, encoding="utf-8"))["offload_plan"]
static_keys = {(int(item["layer_id"]), int(item["expert_id"])) for item in static}
superset_keys = {(int(item["layer_id"]), int(item["expert_id"])) for item in superset}
wave_keys = {(int(item["layer_id"]), int(item["expert_id"])) for item in wave}
assert len(wave) == len(wave_keys) == 48
assert {layer for layer, _ in wave_keys} == set(range(48))
assert wave_keys <= (superset_keys - static_keys)
PY
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" -m pytest -q \
      tests/v1/core/test_uvm_dynamic_pool_queues.py \
      tests/v1/engine/test_uvm_adaptive_kv.py \
      tests/v1/worker/test_uvm_post_warmup_rebalance.py \
      -k 'backup or autoscale or incremental_expert'
  )
  echo "KV/expert path parity static preflight: OK"
}

run_path_parity_case() {
  local run_root="$1" label="$2" initial_target="$3" batch_size="$4"
  local qualification="$5" autoscale="$6" direct_high="$7"
  local high_requests="$8" low_requests="$9" roundtrip="${10}"
  local case_root="${run_root}/${label}"
  local manifest="${case_root}/cpu_offload_manifest.jsonl"
  local -a optional_args=()
  mkdir -p "${case_root}"
  : >"${manifest}"
  if [[ "${direct_high}" == 1 ]]; then
    optional_args+=(--kv-path-direct-high-preset)
  fi
  if [[ "${roundtrip}" == 1 ]]; then
    optional_args+=(--kv-autoscale-roundtrip)
  fi
  gpu_state_json "${case_root}/gpu.before.json"

  set +e
  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      VLLM_ATTENTION_BACKEND=TRITON_ATTN \
      VLLM_CPU_OFFLOAD_BACKEND=copy \
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${STATIC_PLAN}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
      VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode \
      VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592 \
      VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 \
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
      VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288 \
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${INCREMENTAL_PLAN}" \
      VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=0 \
      VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1 \
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST= \
      VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch \
      VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
      VLLM_CPU_OFFLOAD_COPY_SYNC=0 \
      VLLM_CPU_OFFLOAD_MANIFEST_FILE="${manifest}" \
      VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
      VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1 \
      VLLM_UVM_LOG_FILE="${case_root}/uvm.log" \
      VLLM_UVM_KV_CACHE_MANAGED=1 \
      VLLM_UVM_KV_CACHE_LAZY_RESERVE=1 \
      VLLM_UVM_KV_RUNTIME_ENABLE=1 \
      VLLM_UVM_KV_RUNTIME_MODE=enforce \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1 \
      VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${initial_target}" \
      VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS="${initial_target}" \
      VLLM_UVM_KV_BACKUP_RESERVE_BLOCKS=8 \
      VLLM_UVM_KV_BACKUP_GROW_BLOCKS=16 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_AUTO_SCALE_ENABLE="${autoscale}" \
      VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK=3 \
      VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR=2 \
      VLLM_UVM_KV_AUTO_SCALE_SHRINK_DIVISOR=2 \
      VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK=0.80 \
      VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS=20 \
      VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS=40 \
      VLLM_UVM_KV_AUTO_SCALE_SAFETY_BLOCKS=0 \
      VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS=0 \
      VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS=1 \
      VLLM_UVM_KV_AUTO_SCALE_EXPERT_RESTORE_ENABLE=1 \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1 \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1 \
      VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_ADAPTIVE_FAIL_FAST=1 \
      VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1 \
      VLLM_UVM_THREE_QUEUE_QUALIFICATION="${qualification}" \
      VLLM_USE_UVM=0 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      OMP_NUM_THREADS=14 \
      MKL_NUM_THREADS=14 \
      CUDA_VISIBLE_DEVICES=0 \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" \
      taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${RUNNER}" \
        --model "${MODEL}" \
        --fixture "${FIXTURE}" \
        --exact-names "${NAMES}" \
        --expert-plan "${STATIC_PLAN}" \
        --offload-manifest "${manifest}" \
        --output "${case_root}/result.json" \
        --arm D \
        --batch-size "${batch_size}" \
        --cudagraph-capture-sizes 1,8,16,33,264 \
        --warmup-decode-steps 1 \
        --measured-decode-steps 1 \
        --kv-autoscale-low-prompt-tokens 16 \
        --kv-autoscale-low-output-tokens 5 \
        --kv-path-high-performance-requests "${high_requests}" \
        --kv-path-low-performance-requests "${low_requests}" \
        --kv-path-performance-output-tokens 23 \
        --kv-path-performance-warmup-decode-steps 5 \
        --kv-cache-memory-bytes "${KV_MAX_BYTES}" \
        --cpu-offload-gb 30.0 \
        --max-model-len 512 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 512 \
        --gpu-memory-utilization 0.97 \
        --seed 0 \
        --expected-parameter-count 96 \
        --expected-full-parameter-bytes 28991029248 \
        --expected-selected-bytes 6568280064 \
        "${optional_args[@]}"
  ) >"${case_root}/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" >"${case_root}/returncode.txt"
  wait_gpu_clean || die_path_parity "GPU did not clean after ${label}"
  gpu_state_json "${case_root}/gpu.after.json"
  if (( rc != 0 )); then
    tail -n 300 "${case_root}/run.log" >&2 || true
  fi
  return "${rc}"
}

run_qualification_path_parity() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_path_parity "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight_path_parity
  runtime_preflight
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_qualification}"
  [[ ! -e "${run_root}" ]] || die_path_parity "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"

  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  wait_gpu_clean || die_path_parity "GPU is not clean before qualification"
  local kernel_epoch journal_rc dynamic_rc direct_high_rc direct_low_rc
  kernel_epoch="$(date +%s)"

  if run_path_parity_case "${run_root}" dynamic 24 8 1 1 0 3 3 1; then
    dynamic_rc=0
  else
    dynamic_rc=$?
  fi
  if (( dynamic_rc == 0 )) && \
    run_path_parity_case "${run_root}" direct_high 40 8 0 0 1 3 0 0; then
    direct_high_rc=0
  else
    direct_high_rc=$?
  fi
  if (( dynamic_rc == 0 && direct_high_rc == 0 )) && \
    run_path_parity_case "${run_root}" direct_low 20 1 0 0 0 0 3 0; then
    direct_low_rc=0
  else
    direct_low_rc=$?
  fi

  set +e
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${journal_rc}" >"${run_root}/kernel_log_returncode.txt"

  wait_gpu_clean || die_path_parity "GPU did not clean after qualification"
  restore_stock || die_path_parity "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  (( journal_rc == 0 )) || die_path_parity "kernel journal collection failed"
  (( dynamic_rc == 0 )) || die_path_parity "dynamic runner failed"
  (( direct_high_rc == 0 )) || die_path_parity "direct-high runner failed"
  (( direct_low_rc == 0 )) || die_path_parity "direct-low runner failed"
  "${VLLM_PY}" "${VERIFY}" "${run_root}" \
    --output "${run_root}/verification.json"
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
}

case "${1:-help}" in
  static-preflight)
    static_preflight_path_parity
    ;;
  qualify|qualification|run)
    run_qualification_path_parity
    ;;
  verify)
    [[ $# -eq 2 ]] || die_path_parity "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2" --output "$2/verification.json"
    ;;
  *)
    echo "usage: $0 {static-preflight|qualify|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
