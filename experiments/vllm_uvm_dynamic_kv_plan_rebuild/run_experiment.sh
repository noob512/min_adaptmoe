#!/usr/bin/env bash
set -euo pipefail

DYNAMIC_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMIC_REPO_ROOT="$(cd "${DYNAMIC_SCRIPT_DIR}/../.." && pwd)"
DYNAMIC_BASE_HARNESS="${DYNAMIC_REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${DYNAMIC_BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

SCRIPT_DIR="${DYNAMIC_SCRIPT_DIR}"
REPO_ROOT="${DYNAMIC_REPO_ROOT}"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/vllm_uvm_dual_topology_graph/benchmark_dual_topology_steps.py"
VERIFY="${SCRIPT_DIR}/verify_experiment.py"
TEST_FILE="${VLLM_ROOT}/tests/test_cpu_offload_exact_static.py"
INPUT_ROOT="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs"
NAMES="${NAMES:-${INPUT_ROOT}/exact_parameter_names.txt}"
PLAN_2G="${PLAN_2G:-${INPUT_ROOT}/cold_decode_plan_2g.json}"
PLAN_4G="${PLAN_4G:-${INPUT_ROOT}/cold_decode_plan_4g.json}"
PLAN_2G_SELECTED_BYTES="${PLAN_2G_SELECTED_BYTES:-2264924160}"
PLAN_4G_SELECTED_BYTES="${PLAN_4G_SELECTED_BYTES:-4303355904}"
PLAN_2G_LABEL="${PLAN_2G_LABEL:-2g_low_pressure}"
PLAN_4G_LABEL="${PLAN_4G_LABEL:-4g_high_pressure}"
EXPECTED_PARAMETER_COUNT="${EXPECTED_PARAMETER_COUNT:-96}"
EXPECTED_FULL_PARAMETER_BYTES="${EXPECTED_FULL_PARAMETER_BYTES:-28991029248}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-30.0}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-167772160}"
SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-2415919104}"
STATIC_ALIAS_PTE_MODE="${STATIC_ALIAS_PTE_MODE:-auto}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
CPUSET="${CPUSET:-0-13}"

die_dynamic() {
  echo "dynamic-plan experiment: $*" >&2
  exit 1
}

require_dynamic_file() {
  [[ -f "$1" ]] || die_dynamic "missing file: $1"
}

static_preflight_dynamic() {
  require_dynamic_file "${RUNNER}"
  require_dynamic_file "${VERIFY}"
  require_dynamic_file "${NAMES}"
  require_dynamic_file "${PLAN_2G}"
  require_dynamic_file "${PLAN_4G}"
  [[ "$(wc -l <"${NAMES}")" -eq "${EXPECTED_PARAMETER_COUNT}" ]] \
    || die_dynamic "expected ${EXPECTED_PARAMETER_COUNT} exact parameter names"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile \
    "${RUNNER}" "${VERIFY}" \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py"
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" -m pytest -q "${TEST_FILE}" \
      -k 'dynamic_plan_rebuild or dynamic_expert_plan_rebuild or shutdown_reference_release'
  )
  echo "dynamic-plan static preflight: OK"
}

write_sequence() {
  local output="$1"
  "${VLLM_PY}" - "${output}" "${PLAN_4G}" "${PLAN_2G}" \
    "${PLAN_4G_SELECTED_BYTES}" "${PLAN_2G_SELECTED_BYTES}" \
    "${PLAN_4G_LABEL}" "${PLAN_2G_LABEL}" <<'PY'
import json
import sys

output, plan_4g, plan_2g, bytes_4g, bytes_2g, label_4g, label_2g = sys.argv[1:]
records = [
    {
        "label": label_4g,
        "plan": plan_4g,
        "expected_selected_bytes": int(bytes_4g),
        "batch_size": 4,
        "max_tokens": 2,
        "pressure_threshold_tokens": 1_000,
        "expected_pressure_class": "high",
    },
    {
        "label": label_2g,
        "plan": plan_2g,
        "expected_selected_bytes": int(bytes_2g),
        "batch_size": 1,
        "max_tokens": 2,
        "pressure_threshold_tokens": 1_000,
        "expected_pressure_class": "low",
    },
]
with open(output, "w", encoding="utf-8") as handle:
    json.dump(records, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

run_qualification() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_dynamic "GPU execution requires ALLOW_GPU_RUN=1"
  runtime_preflight
  require_dynamic_file "${MODEL}/config.json"
  require_dynamic_file "${FIXTURE}"
  require_dynamic_file "${NAMES}"
  require_dynamic_file "${PLAN_2G}"
  require_dynamic_file "${PLAN_4G}"
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_qualification}"
  [[ ! -e "${run_root}" ]] || die_dynamic "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  local sequence="${run_root}/dynamic_sequence.json"
  local manifest="${run_root}/cpu_offload_manifest.jsonl"
  local output="${run_root}/result.json"
  write_sequence "${sequence}"
  : >"${manifest}"

  local parameter_va_mode="stable_rebuild_recapture"
  if [[ "${VLLM_UVM_E5_NONSTABLE_VA_RECAPTURE:-0}" == 1 ]]; then
    parameter_va_mode="nonstable_reallocate_recapture"
  fi
  "${VLLM_PY}" - "${run_root}/run_config.json" "${sequence}" \
    "${PLAN_2G}" "${PLAN_2G_SELECTED_BYTES}" "${EXPECTED_PARAMETER_COUNT}" \
    "${EXPECTED_FULL_PARAMETER_BYTES}" "${KV_CACHE_BYTES}" \
    "${SLOT_BUDGET_BYTES}" "${CPU_OFFLOAD_GB}" "${parameter_va_mode}" <<'PY'
import json, os, sys
(output, sequence_path, initial_plan, initial_bytes, parameter_count,
 full_bytes, kv_bytes, slot_bytes, cpu_offload_gb, parameter_va_mode) = sys.argv[1:]
records = json.load(open(sequence_path, encoding="utf-8"))
payload = {
    "schema": "vllm_uvm_dynamic_plan_rebuild_run_v2",
    "initial_plan": os.path.realpath(initial_plan),
    "initial_selected_bytes": int(initial_bytes),
    "expected_parameter_count": int(parameter_count),
    "expected_full_parameter_bytes": int(full_bytes),
    "physical_kv_allocation_bytes": int(kv_bytes),
    "slot_budget_bytes": int(slot_bytes),
    "cpu_offload_gb": float(cpu_offload_gb),
    "parameter_va_mode": parameter_va_mode,
    "dynamic_records": records,
}
open(output, "x", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  wait_gpu_clean || die_dynamic "GPU is not clean before qualification"
  gpu_state_json "${run_root}/gpu.before.json"
  local kernel_epoch rc journal_rc
  kernel_epoch="$(date +%s)"
  printf '%s\n' \
    "initial=${PLAN_2G_LABEL}/b1/low" \
    "generation1=${PLAN_4G_LABEL}/b4/high" \
    "generation2=${PLAN_2G_LABEL}/b1/low" \
    >"${run_root}/policy.txt"
  printf '%s\n' \
    "NAMES=${NAMES}" \
    "PLAN_2G=${PLAN_2G}" \
    "PLAN_4G=${PLAN_4G}" \
    "PLAN_2G_SELECTED_BYTES=${PLAN_2G_SELECTED_BYTES}" \
    "PLAN_4G_SELECTED_BYTES=${PLAN_4G_SELECTED_BYTES}" \
    "EXPECTED_PARAMETER_COUNT=${EXPECTED_PARAMETER_COUNT}" \
    "EXPECTED_FULL_PARAMETER_BYTES=${EXPECTED_FULL_PARAMETER_BYTES}" \
    "KV_CACHE_BYTES=${KV_CACHE_BYTES}" \
    "SLOT_BUDGET_BYTES=${SLOT_BUDGET_BYTES}" \
    "CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB}" \
    "STATIC_ALIAS_PTE_MODE=${STATIC_ALIAS_PTE_MODE}" \
    "PARAMETER_VA_MODE=${parameter_va_mode}" \
    >"${run_root}/environment.txt"
  sha256sum "${BASH_SOURCE[0]}" "${RUNNER}" "${VERIFY}" "${NAMES}" \
    "${PLAN_2G}" "${PLAN_4G}" "${FIXTURE}" "${MODEL}/config.json" \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_static_alias.py" \
    "${CUSTOM_UVM}" >"${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"

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
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${PLAN_2G}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE="${STATIC_ALIAS_PTE_MODE}" \
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
      VLLM_UVM_LOG_FILE="${run_root}/uvm.log" \
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
        --expert-plan "${PLAN_2G}" \
        --dynamic-plan-sequence "${sequence}" \
        --offload-manifest "${manifest}" \
        --output "${output}" \
        --arm D \
        --batch-size 1 \
        --cudagraph-capture-sizes 1,4,314,1217 \
        --warmup-decode-steps 2 \
        --measured-decode-steps 4 \
        --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
        --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
        --max-model-len 512 \
        --max-num-seqs 4 \
        --max-num-batched-tokens 2048 \
        --gpu-memory-utilization 0.97 \
        --seed 0 \
        --audit-trace \
        --expected-parameter-count "${EXPECTED_PARAMETER_COUNT}" \
        --expected-full-parameter-bytes "${EXPECTED_FULL_PARAMETER_BYTES}" \
        --expected-selected-bytes "${PLAN_2G_SELECTED_BYTES}"
  ) >"${run_root}/run.log" 2>&1
  rc=$?
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${rc}" >"${run_root}/returncode.txt"
  printf '%s\n' "${journal_rc}" >"${run_root}/kernel_log_returncode.txt"
  wait_gpu_clean || die_dynamic "GPU did not clean after qualification"
  gpu_state_json "${run_root}/gpu.after.json"

  restore_stock || die_dynamic "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  (( rc == 0 )) || {
    tail -n 240 "${run_root}/run.log" >&2 || true
    die_dynamic "runner failed with rc=${rc}"
  }
  (( journal_rc == 0 )) || die_dynamic "kernel journal collection failed"
  "${VLLM_PY}" "${VERIFY}" "${run_root}" \
    --output "${run_root}/verification.json"
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
}

case "${1:-help}" in
  static-preflight) static_preflight_dynamic ;;
  qualify|qualification|run) run_qualification ;;
  verify)
    [[ $# -eq 2 ]] || die_dynamic "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2" --output "$2/verification.json"
    ;;
  *)
    echo "usage: $0 {static-preflight|qualify|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
