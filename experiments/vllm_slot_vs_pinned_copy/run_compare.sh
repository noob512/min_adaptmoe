#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
BENCHMARK="${REPO_ROOT}/experiments/broad_offload_compare/benchmark_custom_offload.py"
PREPARE="${SCRIPT_DIR}/prepare_inputs.py"
VERIFY="${SCRIPT_DIR}/verify_compare.py"
COMBINED_PROBE="${REPO_ROOT}/experiments/uvm_slot_transition_shared_64k_probe/run_probe.py"
CUSTOM_MODULE="${CUSTOM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
TRANSITION_FAST_INDEX_PATH="${TRANSITION_FAST_INDEX_PATH:-/sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_fast_index}"
DIRECT_CUTOVER_PATH="${DIRECT_CUTOVER_PATH:-/sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_direct_cutover}"
PLAN_GENERATION_CONTRACT="module_lifetime_monotonic_nonzero_v1"
DEFAULT_PINNED_ANCHOR_MAX_DRIFT_PCT="3.0"

DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"

COMMAND="${1:-help}"
ACTIVE_MONITOR_PID=""

cleanup_monitor() {
  if [[ -n "${ACTIVE_MONITOR_PID}" ]]; then
    kill "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    wait "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    ACTIVE_MONITOR_PID=""
  fi
}
trap cleanup_monitor EXIT INT TERM

die() {
  echo "error: $*" >&2
  exit 1
}

snapshot_gpu() {
  local output="$1"
  nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader,nounits > "${output}"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits >> "${output}" 2>&1 || true
}

wait_clean_gpu() {
  local output="$1"
  local gpu_index="$2"
  local max_mib="$3"
  local timeout_sec="$4"
  local used_mib pids
  for ((elapsed = 0; elapsed <= timeout_sec; ++elapsed)); do
    used_mib="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
        && (( used_mib <= max_mib )) \
        && [[ -z "${pids}" ]]; then
      snapshot_gpu "${output}"
      return 0
    fi
    sleep 1
  done
  snapshot_gpu "${output}" || true
  die "GPU is not clean (used=${used_mib:-unknown} MiB pids=${pids:-none})"
}

static_preflight() {
  [[ -x "${VLLM_PY}" ]] || die "missing vLLM Python: ${VLLM_PY}"
  [[ -f "${BENCHMARK}" && -f "${PREPARE}" && -f "${VERIFY}" \
      && -f "${COMBINED_PROBE}" ]] \
    || die "comparison scripts are incomplete"
  [[ -f "${CUSTOM_MODULE}" ]] || die "missing custom UVM module"
  command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
  command -v timeout >/dev/null || die "timeout unavailable"
  command -v taskset >/dev/null || die "taskset unavailable"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile \
    "${BENCHMARK}" "${PREPARE}" "${VERIFY}" "${COMBINED_PROBE}"
  "${VLLM_PY}" "${BENCHMARK}" --help | rg -q -- '--cpu-offload-gb'
  rg -q 'VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE' \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    || die "exact-name copy gate is missing"
  rg -q 'VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD' \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    || die "formal slot selection gate is missing"
  rg -q 'REUSE_BACKING_REQUIRED' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    || die "fixed-bank reuse contract is missing"
  rg -q 'restore_uvm_expert_groups_to_slots' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "layer slot-restore batching is missing"
  rg -q 'register_uvm_slot_direct_layer_plans' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "registered compact layer plans are missing"
  rg -q 'slot_plan_register_calls' \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c" \
    || die "registered-plan driver telemetry is missing"
  echo "static preflight: OK"
}

runtime_preflight() {
  static_preflight
  local module_file_srcversion
  local expected_loaded_srcversion
  module_file_srcversion="$(modinfo -F srcversion "${CUSTOM_MODULE}")"
  expected_loaded_srcversion="${EXPECTED_LOADED_SRCVERSION:-${module_file_srcversion}}"
  [[ "${expected_loaded_srcversion}" =~ ^[0-9A-F]+$ ]] \
    || die "EXPECTED_LOADED_SRCVERSION is invalid"
  [[ "${expected_loaded_srcversion}" \
      == "$(sed -n '1p' /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)" ]] \
    || die "loaded UVM is not the requested custom module"
  [[ "$(tr -d '[:space:]' < /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)" == "1" ]] \
    || die "slot PTE remap is disabled"
  [[ -r "${STATS_PATH}" ]] || die "slot latency stats unavailable"
  local plan_counter
  for plan_counter in slot_plan_register_calls slot_plan_execute_calls \
      slot_plan_destroy_calls slot_legacy_ioctl_calls \
      slot_legacy_transition_calls slot_session_abort_calls \
      slot_session_abort_successes slot_session_abort_failures \
      slot_session_abort_released_slots slot_session_abort_released_bytes \
      slot_session_abort_remaining_slots; do
    rg -q "^${plan_counter}[[:space:]]" "${STATS_PATH}" \
      || die "loaded UVM lacks ${plan_counter} telemetry"
  done
  local direct_cutover="${SLOT_DIRECT_CUTOVER:-0}"
  [[ "${direct_cutover}" == "0" || "${direct_cutover}" == "1" ]] \
    || die "SLOT_DIRECT_CUTOVER must be 0 or 1"
  if [[ "${SLOT_LAYER_TRANSITION:-0}" == "1" || "${direct_cutover}" == "1" ]]; then
    [[ -r "${DIRECT_CUTOVER_PATH}" ]] \
      || die "slot direct-cutover parameter is unavailable"
    [[ "$(tr -d '[:space:]' < "${DIRECT_CUTOVER_PATH}")" == "${direct_cutover}" ]] \
      || die "loaded direct-cutover parameter does not match SLOT_DIRECT_CUTOVER"
  fi
}

run_arm() {
  local arm="$1"
  local ordinal="$2"
  local arm_dir="$3"
  local model="$4"
  local fixture="$5"
  local plan="$6"
  local names_file="$7"
  local gpu_index="$8"
  local cpuset="$9"
  local cpu_threads="${10}"
  local timeout_sec="${11}"
  local clean_mib="${12}"
  local clean_timeout="${13}"
  local profile_label mechanism_class cpu_offload_gb
  local slot_trace_target="${arm_dir}/slot_trace.jsonl"
  local slot_control_target=""
  local copy_trace_enable=1
  local -a arm_env

  if [[ "${TRACE_ENABLE:-1}" == "0" ]]; then
    slot_trace_target=""
    copy_trace_enable=0
  elif [[ "${TRACE_ENABLE:-1}" != "1" ]]; then
    die "TRACE_ENABLE must be 0 or 1"
  fi

  if [[ "${arm}" == "copy" ]]; then
    profile_label="pinned_copy_on_forward_exact"
    mechanism_class="vllm_pinned_copy_on_forward_exact_l0_l3"
    cpu_offload_gb="3.0"
    arm_env=(
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=0
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=restore
      VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD=0
      VLLM_CPU_OFFLOAD_BACKEND=copy
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${names_file}"
      VLLM_CPU_OFFLOAD_STATIC_BANK=0
      VLLM_CPU_OFFLOAD_COPY_TRACE="${copy_trace_enable}"
      VLLM_CPU_OFFLOAD_COPY_SYNC="${COPY_SYNC:-0}"
    )
  elif [[ "${arm}" == "slot" ]]; then
    slot_control_target="${arm_dir}/slot_control_summary.json"
    if [[ "${SLOT_REUSE_BANK:-0}" == "1" ]]; then
      profile_label="slot_reuse_bank_exact_l0_l3_all_cold"
      mechanism_class="vllm_slot_reuse_bank_exact_l0_l3_all_cold_576m_cache"
    else
      profile_label="slot_exact_l0_l3_all_cold"
      mechanism_class="vllm_slot_exact_l0_l3_all_cold_576m_cache"
    fi
    cpu_offload_gb="0.0"
    arm_env=(
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=slot_restore
      VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD=1
      VLLM_UVM_SLOT_DIRECT_REUSE_BANK="${SLOT_REUSE_BANK:-0}"
      VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES="${SLOT_PRECOMPUTED_LANES:-1}"
      VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION="${SLOT_LAYER_TRANSITION:-0}"
      VLLM_UVM_SLOT_DIRECT_INIT_BANK="${SLOT_INIT_BANK:-0}"
      VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN="${SLOT_REGISTERED_PLAN:-0}"
      VLLM_UVM_SLOT_DIRECT_SHARED_PUSH="${SLOT_SHARED_PUSH:-0}"
      VLLM_UVM_SLOT_DIRECT_64K_PTE="${SLOT_64K_PTE:-0}"
      VLLM_CPU_OFFLOAD_BACKEND=uva
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=
      VLLM_CPU_OFFLOAD_STATIC_BANK=0
      VLLM_CPU_OFFLOAD_COPY_TRACE=0
      VLLM_CPU_OFFLOAD_COPY_SYNC=0
    )
  else
    die "unknown arm ${arm}"
  fi

  mkdir "${arm_dir}"
  : > "${arm_dir}/slot_trace.jsonl"
  : > "${arm_dir}/cpu_offload_manifest.jsonl"
  wait_clean_gpu "${arm_dir}/gpu.before.csv" "${gpu_index}" "${clean_mib}" "${clean_timeout}"
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.before.txt"

  local -a common_env=(
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
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1
    VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
    VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE=off
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=
    VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}"
    VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
    VLLM_UVM_SLOT_DIRECT_CACHE_BYTES=603979776
    VLLM_UVM_SLOT_DIRECT_MAX_SLOTS=1152
      VLLM_UVM_SLOT_DIRECT_TRACE_FILE="${slot_trace_target}"
      VLLM_UVM_SLOT_DIRECT_CONTROL_SUMMARY_FILE="${slot_control_target}"
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${arm_dir}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${cpu_threads}"
    MKL_NUM_THREADS="${cpu_threads}"
    CUDA_VISIBLE_DEVICES="${gpu_index}"
  )
  printf '%s\n' "${common_env[@]}" "${arm_env[@]}" > "${arm_dir}/environment.txt"
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${common_env[@]}" "${arm_env[@]}"
    printf 'timeout %qs taskset -c %q %q %q ' \
      "${timeout_sec}" "${cpuset}" "${VLLM_PY}" "${BENCHMARK}"
    printf '%q ' \
      --model "${model}" \
      --evaluation-fixture "${fixture}" \
      --output "${arm_dir}/result.json" \
      --profile-label "${profile_label}" \
      --mechanism-class "${mechanism_class}" \
      --concurrency 1 --output-tokens 1 --warmup-requests 0 \
      --warmup-output-tokens 1 --kv-cache-memory-bytes 1288490189 \
      --max-model-len 544 --max-num-batched-tokens 2048 --max-num-seqs 32 \
      --gpu-memory-utilization 0.97 --cpu-offload-gb "${cpu_offload_gb}" \
      --seed 0 --require-prompt-token-ids
    printf '\n'
  } > "${arm_dir}/command.txt"
  {
    echo "ordinal=${ordinal}"
    echo "arm=${arm}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${arm_dir}/execution.env"

  nvidia-smi dmon -i "${gpu_index}" -s pucvmet -d 1 -o DT \
    > "${arm_dir}/gpu_dmon.log" 2>&1 &
  ACTIVE_MONITOR_PID="$!"
  local rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${common_env[@]}" "${arm_env[@]}" \
      /usr/bin/time -v -o "${arm_dir}/resource_time.txt" \
      timeout "${timeout_sec}s" taskset -c "${cpuset}" \
      "${VLLM_PY}" "${BENCHMARK}" \
      --model "${model}" \
      --evaluation-fixture "${fixture}" \
      --output "${arm_dir}/result.json" \
      --profile-label "${profile_label}" \
      --mechanism-class "${mechanism_class}" \
      --concurrency 1 --output-tokens 1 --warmup-requests 0 \
      --warmup-output-tokens 1 --kv-cache-memory-bytes 1288490189 \
      --max-model-len 544 --max-num-batched-tokens 2048 --max-num-seqs 32 \
      --gpu-memory-utilization 0.97 --cpu-offload-gb "${cpu_offload_gb}" \
      --seed 0 --require-prompt-token-ids
  ) > "${arm_dir}/run.log" 2>&1
  rc=$?
  set -e
  if (( rc == 0 )); then
    if [[ "${arm}" == "slot" && ! -s "${arm_dir}/slot_control_summary.json" ]]; then
      echo "error: slot process did not write mode8 control summary" >> "${arm_dir}/run.log"
      rc=1
    elif [[ "${arm}" == "copy" && -e "${arm_dir}/slot_control_summary.json" ]]; then
      echo "error: copy process unexpectedly wrote mode8 control summary" >> "${arm_dir}/run.log"
      rc=1
    fi
  fi
  cleanup_monitor
  printf '%s\n' "${rc}" > "${arm_dir}/returncode.txt"
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.after.txt"
  {
    echo "ended_at=$(date --iso-8601=seconds)"
    echo "returncode=${rc}"
  } >> "${arm_dir}/execution.env"
  wait_clean_gpu "${arm_dir}/gpu.after.csv" "${gpu_index}" "${clean_mib}" "${clean_timeout}"
  return "${rc}"
}

run_compare() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] || die "set ALLOW_GPU_RUN=1"
  runtime_preflight
  local order="${ORDER:-CS}"
  [[ "${order}" =~ ^[CS]+$ ]] || die "ORDER must contain only C(copy) and S(slot)"
  [[ "${order}" == *C* && "${order}" == *S* ]] || die "ORDER must include both arms"
  [[ "${COPY_SYNC:-0}" == "0" || "${COPY_SYNC:-0}" == "1" ]] \
    || die "COPY_SYNC must be 0 or 1"
  [[ "${TRACE_ENABLE:-1}" == "0" || "${TRACE_ENABLE:-1}" == "1" ]] \
    || die "TRACE_ENABLE must be 0 or 1"
  [[ "${SLOT_REUSE_BANK:-0}" == "0" || "${SLOT_REUSE_BANK:-0}" == "1" ]] \
    || die "SLOT_REUSE_BANK must be 0 or 1"
  [[ "${SLOT_PRECOMPUTED_LANES:-1}" == "0" || "${SLOT_PRECOMPUTED_LANES:-1}" == "1" ]] \
    || die "SLOT_PRECOMPUTED_LANES must be 0 or 1"
  [[ "${SLOT_LAYER_TRANSITION:-0}" == "0" || "${SLOT_LAYER_TRANSITION:-0}" == "1" ]] \
    || die "SLOT_LAYER_TRANSITION must be 0 or 1"
  [[ "${SLOT_INIT_BANK:-0}" == "0" || "${SLOT_INIT_BANK:-0}" == "1" ]] \
    || die "SLOT_INIT_BANK must be 0 or 1"
  [[ "${SLOT_REGISTERED_PLAN:-0}" == "0" || "${SLOT_REGISTERED_PLAN:-0}" == "1" ]] \
    || die "SLOT_REGISTERED_PLAN must be 0 or 1"
  [[ "${SLOT_SHARED_PUSH:-0}" == "0" || "${SLOT_SHARED_PUSH:-0}" == "1" ]] \
    || die "SLOT_SHARED_PUSH must be 0 or 1"
  [[ "${SLOT_64K_PTE:-0}" == "0" || "${SLOT_64K_PTE:-0}" == "1" ]] \
    || die "SLOT_64K_PTE must be 0 or 1"
  [[ "${SLOT_DIRECT_CUTOVER:-0}" == "0" || "${SLOT_DIRECT_CUTOVER:-0}" == "1" ]] \
    || die "SLOT_DIRECT_CUTOVER must be 0 or 1"
  [[ "${SLOT_REQUIRE_COMBINED:-0}" == "0" || "${SLOT_REQUIRE_COMBINED:-0}" == "1" ]] \
    || die "SLOT_REQUIRE_COMBINED must be 0 or 1"
  [[ "${REQUIRE_ABBA:-0}" == "0" || "${REQUIRE_ABBA:-0}" == "1" ]] \
    || die "REQUIRE_ABBA must be 0 or 1"
  local expected_mode8_fd_candidate_ioctl_calls="${SLOT_EXPECTED_MODE8_FD_CANDIDATE_IOCTL_CALLS:-}"
  if [[ -n "${expected_mode8_fd_candidate_ioctl_calls}" ]]; then
    [[ "${expected_mode8_fd_candidate_ioctl_calls}" =~ ^[0-9]+$ ]] \
      || die "SLOT_EXPECTED_MODE8_FD_CANDIDATE_IOCTL_CALLS must be a non-negative integer"
  elif [[ "${TRACE_ENABLE:-1}" == "0" ]]; then
    die "trace-off requires SLOT_EXPECTED_MODE8_FD_CANDIDATE_IOCTL_CALLS from a trace-on diagnostic"
  fi
  if [[ "${SLOT_INIT_BANK:-0}" == "1" ]]; then
    [[ "${SLOT_REUSE_BANK:-0}" == "1" \
        && "${SLOT_LAYER_TRANSITION:-0}" == "1" ]] \
      || die "SLOT_INIT_BANK requires fixed-bank reuse and layer transition"
  fi
  if [[ "${SLOT_REGISTERED_PLAN:-0}" == "1" ]]; then
    [[ "${SLOT_INIT_BANK:-0}" == "1" \
        && "${SLOT_REUSE_BANK:-0}" == "1" \
        && "${SLOT_PRECOMPUTED_LANES:-1}" == "1" \
        && "${SLOT_LAYER_TRANSITION:-0}" == "1" \
        && "${SLOT_SHARED_PUSH:-0}" == "1" \
      && "${SLOT_64K_PTE:-0}" == "1" ]] \
      || die "SLOT_REGISTERED_PLAN requires init-bank and the combined transition path"
  fi
  if [[ "${SLOT_REQUIRE_COMBINED:-0}" == "1" ]]; then
    [[ "${SLOT_REUSE_BANK:-0}" == "1" \
        && "${SLOT_PRECOMPUTED_LANES:-1}" == "1" \
        && "${SLOT_LAYER_TRANSITION:-0}" == "1" \
        && "${SLOT_SHARED_PUSH:-0}" == "1" \
        && "${SLOT_64K_PTE:-0}" == "1" ]] \
      || die "combined mechanism verification requires every SLOT optimization"
    [[ -r "${TRANSITION_FAST_INDEX_PATH}" \
        && "$(tr -d '[:space:]' < "${TRANSITION_FAST_INDEX_PATH}")" == "1" ]] \
      || die "combined mechanism verification requires transition fast-index=1"
  fi
  if [[ "${SLOT_DIRECT_CUTOVER:-0}" == "1" ]]; then
    [[ "${SLOT_REUSE_BANK:-0}" == "1" \
        && "${SLOT_LAYER_TRANSITION:-0}" == "1" ]] \
      || die "direct cutover requires fixed-bank reuse and layer transition"
    [[ -r "${DIRECT_CUTOVER_PATH}" \
        && "$(tr -d '[:space:]' < "${DIRECT_CUTOVER_PATH}")" == "1" ]] \
      || die "direct cutover was requested but the loaded module parameter is not 1"
  fi
  if [[ "${REQUIRE_ABBA:-0}" == "1" ]]; then
    [[ "${SLOT_REQUIRE_COMBINED:-0}" == "1" ]] \
      || die "REQUIRE_ABBA requires SLOT_REQUIRE_COMBINED=1"
    [[ "${order}" == "CSSC" || "${order}" == "SCCS" ]] \
      || die "REQUIRE_ABBA requires ORDER=CSSC or ORDER=SCCS"
  fi
  local model="${MODEL:-${DEFAULT_MODEL}}"
  local fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local gpu_index="${GPU_INDEX:-0}"
  local cpuset="${CPUSET:-0-13}"
  local cpu_threads="${CPU_THREADS:-14}"
  local timeout_sec="${TIMEOUT_SEC:-3600}"
  local clean_mib="${GPU_CLEAN_MAX_MIB:-0}"
  local clean_timeout="${GPU_CLEAN_TIMEOUT_SEC:-240}"
  local pinned_anchor_max_drift_pct="${PINNED_ANCHOR_MAX_DRIFT_PCT:-${DEFAULT_PINNED_ANCHOR_MAX_DRIFT_PCT}}"
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${order}}"
  "${VLLM_PY}" -c \
    'import math, sys; value = float(sys.argv[1]); assert math.isfinite(value) and value >= 0.0' \
    "${pinned_anchor_max_drift_pct}" \
    || die "PINNED_ANCHOR_MAX_DRIFT_PCT must be a finite non-negative number"
  [[ -d "${model}" && -f "${model}/config.json" ]] || die "model unavailable"
  [[ -f "${fixture}" ]] || die "fixture unavailable"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT exists"
  mkdir -p "${run_root}/inputs"
  run_root="$(cd "${run_root}" && pwd)"
  "${VLLM_PY}" "${PREPARE}" --output-dir "${run_root}/inputs" \
    > "${run_root}/prepare_inputs.stdout.json"
  local plan="${run_root}/inputs/exact_l0_l3_plan.json"
  local names_file="${run_root}/inputs/exact_parameter_names.txt"
  sha256sum "${CUSTOM_MODULE}" "${fixture}" "${model}/config.json" \
    "${plan}" "${names_file}" > "${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short > "${run_root}/git_status.txt"
  snapshot_gpu "${run_root}/gpu.initial.csv"
  {
    local module_file_srcversion
    local expected_loaded_srcversion
    module_file_srcversion="$(modinfo -F srcversion "${CUSTOM_MODULE}")"
    expected_loaded_srcversion="${EXPECTED_LOADED_SRCVERSION:-${module_file_srcversion}}"
    echo "loaded_srcversion=$(sed -n '1p' /sys/module/nvidia_uvm/srcversion)"
    echo "custom_srcversion=${expected_loaded_srcversion}"
    echo "custom_module_file_srcversion=${module_file_srcversion}"
    if [[ "${module_file_srcversion}" == "${expected_loaded_srcversion}" ]]; then
      echo "custom_module_artifact_matches_loaded=1"
    else
      echo "custom_module_artifact_matches_loaded=0"
    fi
    echo "slot_enable=$(tr -d '[:space:]' < /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)"
    echo "order=${order}"
    echo "trace_enable=${TRACE_ENABLE:-1}"
    echo "copy_sync=${COPY_SYNC:-0}"
    echo "slot_reuse_bank=${SLOT_REUSE_BANK:-0}"
    echo "slot_precomputed_lanes=${SLOT_PRECOMPUTED_LANES:-1}"
    echo "slot_layer_transition=${SLOT_LAYER_TRANSITION:-0}"
    echo "slot_init_bank=${SLOT_INIT_BANK:-0}"
    echo "slot_registered_plan=${SLOT_REGISTERED_PLAN:-0}"
    if [[ -n "${expected_mode8_fd_candidate_ioctl_calls}" ]]; then
      echo "slot_expected_mode8_fd_candidate_ioctl_calls=${expected_mode8_fd_candidate_ioctl_calls}"
    fi
    echo "slot_plan_generation_contract=${PLAN_GENERATION_CONTRACT}"
    echo "pinned_anchor_max_drift_pct=${pinned_anchor_max_drift_pct}"
    echo "slot_shared_push=${SLOT_SHARED_PUSH:-0}"
    echo "slot_64k_pte=${SLOT_64K_PTE:-0}"
    echo "slot_direct_cutover_requested=${SLOT_DIRECT_CUTOVER:-0}"
    if [[ -r "${DIRECT_CUTOVER_PATH}" ]]; then
      echo "slot_direct_cutover=$(tr -d '[:space:]' < "${DIRECT_CUTOVER_PATH}")"
    else
      echo "slot_direct_cutover=unavailable"
    fi
    echo "slot_require_combined=${SLOT_REQUIRE_COMBINED:-0}"
    echo "require_abba=${REQUIRE_ABBA:-0}"
    if [[ -r "${TRANSITION_FAST_INDEX_PATH}" ]]; then
      echo "slot_transition_fast_index=$(tr -d '[:space:]' < "${TRANSITION_FAST_INDEX_PATH}")"
    else
      echo "slot_transition_fast_index=unavailable"
    fi
    echo "slot_layer_batch=1"
  } > "${run_root}/run_config.env"
  printf 'ordinal\tarm\tdirectory\treturncode\n' > "${run_root}/execution_status.tsv"

  exec 9>"/tmp/vllm_slot_vs_pinned_copy.lock"
  flock -n 9 || die "another comparison holds the lock"
  local failures=0 ordinal symbol arm arm_dir rc
  for ((index = 0; index < ${#order}; ++index)); do
    ordinal=$((index + 1))
    symbol="${order:index:1}"
    arm="copy"
    [[ "${symbol}" == "S" ]] && arm="slot"
    arm_dir="${run_root}/$(printf '%02d_%s' "${ordinal}" "${arm}")"
    if run_arm "${arm}" "${ordinal}" "${arm_dir}" "${model}" "${fixture}" \
        "${plan}" "${names_file}" "${gpu_index}" "${cpuset}" "${cpu_threads}" \
        "${timeout_sec}" "${clean_mib}" "${clean_timeout}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 160 "${arm_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\t%s\n' "${ordinal}" "${arm}" "$(basename "${arm_dir}")" "${rc}" \
      >> "${run_root}/execution_status.tsv"
  done
  {
    if [[ -r "${TRANSITION_FAST_INDEX_PATH}" ]]; then
      echo "slot_transition_fast_index_final=$(tr -d '[:space:]' < "${TRANSITION_FAST_INDEX_PATH}")"
    else
      echo "slot_transition_fast_index_final=unavailable"
    fi
    if [[ -r "${DIRECT_CUTOVER_PATH}" ]]; then
      echo "slot_direct_cutover_final=$(tr -d '[:space:]' < "${DIRECT_CUTOVER_PATH}")"
    else
      echo "slot_direct_cutover_final=unavailable"
    fi
    echo "loaded_srcversion_final=$(sed -n '1p' /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  } >> "${run_root}/run_config.env"
  snapshot_gpu "${run_root}/gpu.final.csv"
  local verify_rc=0
  if "${VLLM_PY}" "${VERIFY}" "${run_root}" \
      > >(tee "${run_root}/verify.log") \
      2> >(tee "${run_root}/verify.err" >&2); then
    verify_rc=0
  else
    verify_rc=$?
  fi
  echo "RUN_ROOT=${run_root}"
  (( failures == 0 && verify_rc == 0 )) \
    || die "comparison failed: arms=${failures} verifier=${verify_rc}"
}

case "${COMMAND}" in
  static-preflight|preflight)
    static_preflight
    ;;
  run)
    run_compare
    ;;
  verify)
    [[ $# -eq 2 ]] || die "verify requires a run root"
    "${VLLM_PY}" "${VERIFY}" "$2"
    ;;
  help|-h|--help)
    echo "usage: run_compare.sh static-preflight|run|verify RUN_ROOT"
    ;;
  *)
    die "unknown command ${COMMAND}"
    ;;
esac
