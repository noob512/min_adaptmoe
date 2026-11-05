#!/usr/bin/env bash
set -euo pipefail

PHYSICAL_KV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYSICAL_KV_REPO_ROOT="$(cd "${PHYSICAL_KV_SCRIPT_DIR}/../.." && pwd)"
PHYSICAL_KV_BASE_HARNESS="${PHYSICAL_KV_REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${PHYSICAL_KV_BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

SCRIPT_DIR="${PHYSICAL_KV_SCRIPT_DIR}"
REPO_ROOT="${PHYSICAL_KV_REPO_ROOT}"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/vllm_uvm_dual_topology_graph/benchmark_dual_topology_steps.py"
VERIFY="${SCRIPT_DIR}/verify_experiment.py"
INPUT_ROOT="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs"
NAMES="${INPUT_ROOT}/exact_parameter_names.txt"
PLAN_6G="${REPO_ROOT}/experiments/ktransformers_graph_decode_repro_compare/inputs/cold_decode_plan_6g.json"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
KV_MAX_BYTES="${KV_MAX_BYTES:-1610612736}"
KV_LOW_BLOCKS="${KV_LOW_BLOCKS:-105}"
KV_HIGH_BLOCKS="${KV_HIGH_BLOCKS:-1023}"
KV_PRESSURE_THRESHOLD_BLOCKS="${KV_PRESSURE_THRESHOLD_BLOCKS:-106}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
CPUSET="${CPUSET:-0-13}"

die_physical_kv() {
  echo "physical-kv experiment: $*" >&2
  exit 1
}

require_physical_kv_file() {
  [[ -f "$1" ]] || die_physical_kv "missing file: $1"
}

static_preflight_physical_kv() {
  require_physical_kv_file "${RUNNER}"
  require_physical_kv_file "${VERIFY}"
  require_physical_kv_file "${NAMES}"
  require_physical_kv_file "${PLAN_6G}"
  [[ "$(wc -l <"${NAMES}")" -eq 96 ]] \
    || die_physical_kv "expected 96 exact parameter names"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile \
    "${RUNNER}" "${VERIFY}" \
    "${VLLM_ROOT}/vllm/v1/core/uvm_kv_runtime_policy.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py"
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" -m pytest -q \
      tests/v1/core/test_uvm_dynamic_pool_policy.py \
      tests/v1/core/test_uvm_dynamic_pool_queues.py \
      tests/v1/engine/test_uvm_adaptive_kv.py \
      tests/v1/worker/test_uvm_post_warmup_rebalance.py \
      -k 'dynamic_pool or kv_resize or physical_span_snapshot'
  )
  echo "physical-kv static preflight: OK"
}

write_sequence() {
  local output="$1" mode="$2"
  "${VLLM_PY}" - "${output}" "${mode}" <<'PY'
import json
import sys

output, mode = sys.argv[1:]

def record(label, batch_size, pressure):
    return {
        "label": label,
        "batch_size": batch_size,
        "max_tokens": 23,
        "expected_pressure_class": pressure,
        "decode_warmup_steps": 5,
    }

if mode == "dynamic":
    records = [
        record(f"dynamic_high_{index}", 8, "high")
        for index in range(1, 4)
    ] + [
        record(f"dynamic_low_{index}", 1, "low")
        for index in range(1, 4)
    ]
elif mode == "static_low":
    records = [
        record(f"static_low_{index}", 1, "low")
        for index in range(1, 4)
    ]
elif mode == "static_high":
    records = [
        record(f"static_high_{index}", 8, "high")
        for index in range(1, 4)
    ]
else:
    raise SystemExit(f"unknown sequence mode: {mode}")

with open(output, "w", encoding="utf-8") as handle:
    json.dump(records, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

run_arm() {
  local run_root="$1" label="$2" mode="$3" initial_blocks="$4" initial_batch="$5"
  local arm_root="${run_root}/${label}"
  local sequence="${arm_root}/physical_kv_sequence.json"
  local manifest="${arm_root}/cpu_offload_manifest.jsonl"
  local output="${arm_root}/result.json"
  local reuse_probe_bytes=0
  if [[ "${label}" == dynamic ]]; then
    reuse_probe_bytes=536870912
  fi
  mkdir -p "${arm_root}"
  write_sequence "${sequence}" "${mode}"
  : >"${manifest}"

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
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${PLAN_6G}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
      VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode \
      VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592 \
      VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 \
      VLLM_CPU_OFFLOAD_COPY_TRACE=1 \
      VLLM_CPU_OFFLOAD_COPY_SYNC=0 \
      VLLM_CPU_OFFLOAD_MANIFEST_FILE="${manifest}" \
      VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
      VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1 \
      VLLM_UVM_LOG_FILE="${arm_root}/uvm.log" \
      VLLM_UVM_KV_CACHE_MANAGED=1 \
      VLLM_UVM_KV_CACHE_LAZY_RESERVE=1 \
      VLLM_UVM_KV_RUNTIME_ENABLE=1 \
      VLLM_UVM_KV_RUNTIME_MODE=enforce \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1 \
      VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${initial_blocks}" \
      VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS="${initial_blocks}" \
      VLLM_UVM_KV_NEXT_REQUEST_LOW_TARGET_BLOCKS="${KV_LOW_BLOCKS}" \
      VLLM_UVM_KV_NEXT_REQUEST_HIGH_TARGET_BLOCKS="${KV_HIGH_BLOCKS}" \
      VLLM_UVM_KV_NEXT_REQUEST_PRESSURE_THRESHOLD_BLOCKS="${KV_PRESSURE_THRESHOLD_BLOCKS}" \
      VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1 \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1 \
      VLLM_UVM_KV_ADAPTIVE_FAIL_FAST=1 \
      VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
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
        --expert-plan "${PLAN_6G}" \
        --dynamic-kv-sequence "${sequence}" \
        --physical-kv-reuse-probe-bytes "${reuse_probe_bytes}" \
        --offload-manifest "${manifest}" \
        --output "${output}" \
        --arm D \
        --batch-size "${initial_batch}" \
        --cudagraph-capture-sizes 1,8,314,2454 \
        --warmup-decode-steps 5 \
        --measured-decode-steps 16 \
        --kv-cache-memory-bytes "${KV_MAX_BYTES}" \
        --cpu-offload-gb 30.0 \
        --max-model-len 512 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.97 \
        --seed 0 \
        --audit-trace \
        --expected-parameter-count 96 \
        --expected-full-parameter-bytes 28991029248 \
        --expected-selected-bytes 6568280064
  ) >"${arm_root}/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" >"${arm_root}/returncode.txt"
  if (( rc != 0 )); then
    tail -n 240 "${arm_root}/run.log" >&2 || true
    return "${rc}"
  fi
  wait_gpu_clean || die_physical_kv "GPU did not clean after ${label}"
}

run_qualification_physical_kv() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_physical_kv "GPU execution requires ALLOW_GPU_RUN=1"
  runtime_preflight
  require_physical_kv_file "${MODEL}/config.json"
  require_physical_kv_file "${FIXTURE}"
  require_physical_kv_file "${NAMES}"
  require_physical_kv_file "${PLAN_6G}"
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_qualification}"
  [[ ! -e "${run_root}" ]] || die_physical_kv "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"

  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  wait_gpu_clean || die_physical_kv "GPU is not clean before qualification"
  gpu_state_json "${run_root}/gpu.before.json"
  local kernel_epoch journal_rc
  kernel_epoch="$(date +%s)"

  run_arm "${run_root}" dynamic dynamic "${KV_LOW_BLOCKS}" 1
  run_arm "${run_root}" static_low static_low "${KV_LOW_BLOCKS}" 1
  # A full high KV target cannot coexist with the sealed prefill alias
  # topology on this 32 GiB device. Establish the common low startup state,
  # perform the mandatory base request/cutover, then apply high at the first
  # legal scheduler-idle boundary. This is the direct serving-state baseline.
  run_arm "${run_root}" static_high static_high "${KV_LOW_BLOCKS}" 1

  set +e
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${journal_rc}" >"${run_root}/kernel_log_returncode.txt"
  (( journal_rc == 0 )) || die_physical_kv "kernel journal collection failed"

  wait_gpu_clean || die_physical_kv "GPU did not clean after qualification"
  gpu_state_json "${run_root}/gpu.after.json"
  restore_stock || die_physical_kv "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  "${VLLM_PY}" "${VERIFY}" "${run_root}" \
    --output "${run_root}/verification.json"
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
}

case "${1:-help}" in
  static-preflight) static_preflight_physical_kv ;;
  qualify|qualification|run) run_qualification_physical_kv ;;
  verify)
    [[ $# -eq 2 ]] || die_physical_kv "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2" --output "$2/verification.json"
    ;;
  *)
    echo "usage: $0 {static-preflight|qualify|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
