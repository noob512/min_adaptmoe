#!/usr/bin/env bash
set -euo pipefail

E14_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${E14_DIR}/../../.." && pwd)"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1207959552}"
export ACCESS_COUNTER_MIGRATION_ENABLE="${ACCESS_COUNTER_MIGRATION_ENABLE:-0}"
export UVM_PREFETCH_ENABLE="${UVM_PREFETCH_ENABLE:-0}"
export UVM_THRASHING_ENABLE="${UVM_THRASHING_ENABLE:-0}"
export STATIC_ALIAS_DUAL_VIEW_ENABLE=0
export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=0
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

CONFIG="${E14_CONFIG:-${E14_DIR}/config.json}"
E14_VERIFY="${E14_DIR}/verify_e14.py"
E14_BENCH="${REPO_ROOT}/experiments/vllm_p3_static_alias_graph_integration/benchmark_full_graph_steps.py"
E14_ROUTE_BUILDER="${E14_DIR}/build_request_cold_plan.py"
VLLM_MODELS_UTILS="${VLLM_ROOT}/vllm/model_executor/models/utils.py"
MODEL="${MODEL:?Set MODEL to the local Qwen3-30B-A3B-FP8 snapshot directory}"
FIXTURE="${EVAL_FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8/evaluation_short_dialogue.json}"
COLD_PLAN="${E14_COLD_PLAN:-${REPO_ROOT}/experiments/vllm_uvm_large_kv_multilevel/inputs/cold_decode_plan_6g.json}"
COLD_NAMES="${REPO_ROOT}/experiments/vllm_uvm_large_kv_multilevel/inputs/exact_parameter_names.txt"
WHOLE_NAMES="${E14_DIR}/inputs/whole_layers_0_10_exact_parameter_names.txt"
WHOLE_PLAN="${E14_DIR}/inputs/whole_layers_0_10_full_expert_plan.json"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
CPUSET="${CPUSET:-0-13}"
CPU_THREADS="${CPU_THREADS:-14}"

# The benchmark runs from VLLM_ROOT, so user-provided overrides must not remain
# relative to the caller's working directory.
CONFIG="$(realpath -e -- "${CONFIG}")"
MODEL="$(realpath -e -- "${MODEL}")"
FIXTURE="$(realpath -e -- "${FIXTURE}")"
COLD_PLAN="$(realpath -e -- "${COLD_PLAN}")"

die_e14() { echo "E14: $*" >&2; exit 1; }

static_preflight_e14() {
  local path
  for path in "${CONFIG}" "${E14_VERIFY}" "${E14_BENCH}" "${E14_ROUTE_BUILDER}" "${MODEL}/config.json" \
      "${FIXTURE}" "${COLD_PLAN}" "${COLD_NAMES}" "${WHOLE_NAMES}" \
      "${WHOLE_PLAN}"; do
    require_file "${path}"
  done
  [[ "$(wc -l <"${COLD_NAMES}")" -eq 96 ]] || die_e14 "cold exact-name count differs"
  [[ "$(wc -l <"${WHOLE_NAMES}")" -eq 22 ]] || die_e14 "whole-layer exact-name count differs"
  "${VLLM_PY}" - "${CONFIG}" "${REPO_ROOT}" "${COLD_PLAN}" <<'PY'
import json, pathlib, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
root = pathlib.Path(sys.argv[2])
cold_plan = pathlib.Path(sys.argv[3]).resolve()
assert config["schema"] == "eurosys_e14_uva_layer_vs_cold_protocol_v1"
assert config["batch_sizes"] == [1, 4, 16, 32]
assert config["prompt_tokens_per_request"] == 8
assert config["capture_sizes"] == []
assert config["execution"]["phase_mode"] == "remote_only"
assert config["execution"]["remote_only_direct_forward"] is True
assert config["execution"]["per_forward_copy_trace"] is False
assert config["execution"]["cudagraph_mode"] == "NONE"
assert config["execution"]["enforce_eager"] is True
assert config["arms"]["D"]["selected_main_bytes"] == 1_392 * 4_718_592
assert config["arms"]["W"]["selected_main_bytes"] == 11 * 603_979_776
assert abs(config["arms"]["W"]["selected_main_bytes"] / config["arms"]["D"]["selected_main_bytes"] - 1) < .02
for key in ("fixture", "cold_plan", "cold_exact_names", "whole_layer_exact_names"):
    assert (root / config[key]).is_file(), key
assert (root / config["cold_plan"]).resolve() == cold_plan
plan = json.load(open(cold_plan, encoding="utf-8"))
records = plan["offload_plan"]
coordinates = {(int(item["layer_id"]), int(item["expert_id"])) for item in records}
assert len(records) == len(coordinates) == 1_392
assert all(0 <= layer < 48 and 0 <= expert < 128 for layer, expert in coordinates)
assert all(sum(layer == expected for layer, _ in coordinates) == 29 for expected in range(48))
assert int(plan["metadata"]["selected_main_bytes"]) == config["arms"]["D"]["selected_main_bytes"]
PY
  "${VLLM_PY}" -m py_compile "${E14_VERIFY}" "${E14_BENCH}" "${E14_ROUTE_BUILDER}"
  bash -n "${BASH_SOURCE[0]}"
  echo "E14 static preflight: OK"
}

arm_name_e14() {
  case "$1" in
    D) echo distributed_cold29 ;;
    W) echo fixed_whole_layers_0_10 ;;
    *) die_e14 "unknown arm $1" ;;
  esac
}

run_case_e14() {
  local symbol="$1" ordinal="$2" batch_order="$3" case_dir="$4"
  local arm names plan expected_backing selected_bytes selected_experts
  arm="$(arm_name_e14 "${symbol}")"
  case "${symbol}" in
    D)
      names="${COLD_NAMES}"
      plan="${COLD_PLAN}"
      expected_backing=273678336
      selected_bytes=6568280064
      selected_experts=1392
      ;;
    W)
      names="${WHOLE_NAMES}"
      plan="${WHOLE_PLAN}"
      expected_backing=1207959552
      selected_bytes=6643777536
      selected_experts=1408
      ;;
  esac
  mkdir "${case_dir}"
  : >"${case_dir}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die_e14 "GPU is not clean before ${arm}"
  gpu_state_json "${case_dir}/gpu.before.json"
  cp -- "${STATS_PATH}" "${case_dir}/latency_stats.before.txt"
  local kernel_epoch rc journal_rc
  kernel_epoch="$(date +%s)"

  local -a envs=(
    VLLM_USE_UVM=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_ATTENTION_BACKEND=TRITON_ATTN
    VLLM_CPU_OFFLOAD_BACKEND=copy
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${names}"
    VLLM_CPU_OFFLOAD_STATIC_BANK=0
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=0
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${plan}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT=4
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${expected_backing}"
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
    VLLM_UVM_ORIGINAL_VA_PHASE_MODE=remote_only
    VLLM_UVM_REMOTE_ONLY_DIRECT_FORWARD=1
    VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0
    VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592
    VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read
    VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT=0
    VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
    VLLM_CPU_OFFLOAD_COPY_TRACE=0
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_UVM_LOG_FILE="${case_dir}/uvm.log"
    VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE="${ROUTING_TRACE_ENABLE}"
    VLLM_MOE_TOKEN_EXPERT_TRACE_FILE="${case_dir}/expert_trace.jsonl"
    E14_ARM="${symbol}"
    E14_SELECTED_MAIN_BYTES="${selected_bytes}"
    E14_SELECTED_EXPERT_INSTANCES="${selected_experts}"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    PYTHONUNBUFFERED=1
    OMP_NUM_THREADS="${CPU_THREADS}"
    MKL_NUM_THREADS="${CPU_THREADS}"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
  )
  printf '%s\n' "${envs[@]}" >"${case_dir}/environment.txt"

  local -a args=(
    --model "${MODEL}"
    --fixture "${FIXTURE}"
    --output "${case_dir}/result.json"
    --offload-manifest "${case_dir}/cpu_offload_manifest.jsonl"
    --profile-label "e14_${arm}_p${ordinal}"
    --mechanism-class "e14_pure_uva_${arm}"
    --execution-mode eager
    --prompt-token-limit 8
    --batch-sizes "${batch_order}"
    --cudagraph-capture-sizes "${CAPTURE_SIZES}"
    --conditioning-repeats "${CONDITIONING_REPEATS}"
    --measurement-repeats "${MEASUREMENT_REPEATS}"
    --warmup-decode-steps "${WARMUP_DECODE_STEPS}"
    --measured-decode-steps "${MEASURED_DECODE_STEPS}"
    --kv-cache-memory-bytes 536870912
    --cpu-offload-gb 6.5
    --max-model-len 64
    --max-num-seqs 32
    --max-num-batched-tokens 256
    --gpu-memory-utilization 0.97
    --seed 20260903
    --formal-e3
  )
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${envs[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}" "${VLLM_PY}" "${E14_BENCH}"
    printf '%q ' "${args[@]}"
    printf '\n'
  } >"${case_dir}/command.txt"

  set +e
  (
    cd "${VLLM_ROOT}"
    env "${envs[@]}" /usr/bin/time -v -o "${case_dir}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${E14_BENCH}" "${args[@]}"
  ) >"${case_dir}/run.log" 2>&1
  rc=$?
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${case_dir}/kernel.since.log" 2>"${case_dir}/kernel.since.err"
  journal_rc=$?
  set -e
  echo "${rc}" >"${case_dir}/returncode.txt"
  echo "${journal_rc}" >"${case_dir}/kernel_log_returncode.txt"
  cp -- "${STATS_PATH}" "${case_dir}/latency_stats.after.txt"
  (( journal_rc == 0 )) || return 97
  wait_gpu_clean || die_e14 "GPU/client state did not clean after ${arm}"
  gpu_state_json "${case_dir}/gpu.after.json"
  return "${rc}"
}

run_suite_e14() {
  local mode="$1" order
  static_preflight_e14
  runtime_preflight
  ROUTING_TRACE_ENABLE=0
  case "${mode}" in
    smoke)
      order="${ORDER:-DW}"
      BATCHES_ASC=1,16
      BATCHES_DESC=16,1
      CAPTURE_SIZES=1
      CONDITIONING_REPEATS=0
      MEASUREMENT_REPEATS=1
      WARMUP_DECODE_STEPS=1
      MEASURED_DECODE_STEPS=2
      ;;
    qualification)
      order="${ORDER:-DW}"
      BATCHES_ASC=1,4,16,32
      BATCHES_DESC=32,16,4,1
      CAPTURE_SIZES=1
      CONDITIONING_REPEATS=1
      MEASUREMENT_REPEATS=2
      WARMUP_DECODE_STEPS=3
      MEASURED_DECODE_STEPS=8
      ;;
    formal)
      order="${ORDER:-DWWDDW}"
      BATCHES_ASC=1,4,16,32
      BATCHES_DESC=32,16,4,1
      CAPTURE_SIZES=1
      CONDITIONING_REPEATS=1
      MEASUREMENT_REPEATS=3
      WARMUP_DECODE_STEPS=5
      MEASURED_DECODE_STEPS=16
      ;;
    route-profile)
      order=D
      BATCHES_ASC=32
      BATCHES_DESC=32
      CAPTURE_SIZES=1
      CONDITIONING_REPEATS=0
      MEASUREMENT_REPEATS=1
      WARMUP_DECODE_STEPS=5
      MEASURED_DECODE_STEPS=16
      ROUTING_TRACE_ENABLE=1
      ;;
    *) die_e14 "unknown mode ${mode}" ;;
  esac
  [[ "${order}" =~ ^[DW]+$ ]] || die_e14 "ORDER contains an unknown arm"

  local run_root="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e14-$(date -u +%Y%m%dT%H%M%SZ)-${mode}}"
  [[ ! -e "${run_root}" ]] || die_e14 "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  cp -- "${CONFIG}" "${run_root}/run_config.json"
  sha256sum "${CONFIG}" "${COLD_PLAN}" "${COLD_NAMES}" \
    "${WHOLE_NAMES}" "${WHOLE_PLAN}" \
    "${FIXTURE}" "${MODEL}/config.json" "${E14_BENCH}" "${E14_VERIFY}" \
    "${VLLM_MODELS_UTILS}" \
    "${E14_ROUTE_BUILDER}" \
    "${BASH_SOURCE[0]}" "${CUSTOM_UVM}" "${STOCK_UVM}" \
    >"${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  printf 'ordinal\tsymbol\tarm\tbatch_order\tdirectory\treturncode\n' \
    >"${run_root}/execution_status.tsv"

  local failures=0 index symbol arm batch_order case_dir rc
  for ((index = 0; index < ${#order}; ++index)); do
    symbol="${order:index:1}"
    arm="$(arm_name_e14 "${symbol}")"
    batch_order="${BATCHES_ASC}"
    (( index % 2 == 1 )) && batch_order="${BATCHES_DESC}"
    case_dir="${run_root}/$(printf '%02d_%s' "$((index + 1))" "${arm}")"
    if run_case_e14 "${symbol}" "$((index + 1))" "${batch_order}" "${case_dir}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 180 "${case_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$((index + 1))" "${symbol}" \
      "${arm}" "${batch_order}" "$(basename "${case_dir}")" "${rc}" \
      >>"${run_root}/execution_status.tsv"
  done

  restore_stock || die_e14 "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  if (( failures == 0 )); then
    if [[ "${mode}" == route-profile ]]; then
      "${VLLM_PY}" "${E14_ROUTE_BUILDER}" \
        --result "${run_root}/01_distributed_cold29/result.json" \
        --routing-trace "${run_root}/01_distributed_cold29/expert_trace.jsonl" \
        --fixture "${FIXTURE}" \
        --output "${run_root}/request_specific_cold29_plan.json" \
        >"${run_root}/request_cold_plan_summary.json"
    else
      "${VLLM_PY}" "${E14_VERIFY}" "${run_root}" --write-outputs
    fi
  fi
  echo "RUN_ROOT=${run_root}"
  (( failures == 0 )) || die_e14 "${failures} E14 process(es) failed"
}

case "${1:-help}" in
  static-preflight) static_preflight_e14 ;;
  smoke) run_suite_e14 smoke ;;
  route-profile) run_suite_e14 route-profile ;;
  qualify|qualification) run_suite_e14 qualification ;;
  formal|run) run_suite_e14 formal ;;
  verify)
    [[ $# -eq 2 ]] || die_e14 "verify requires RUN_ROOT"
    "${VLLM_PY}" "${E14_VERIFY}" "$2" --write-outputs
    ;;
  *)
    echo "usage: $0 {static-preflight|route-profile|smoke|qualify|formal|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
