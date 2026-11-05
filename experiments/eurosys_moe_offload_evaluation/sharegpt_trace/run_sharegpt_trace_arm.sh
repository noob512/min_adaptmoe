#!/usr/bin/env bash
set -euo pipefail

# Replay one frozen ShareGPT open-loop trace against one Qwen mechanism.
# This launcher deliberately keeps the service lifecycle and replay artifact
# together so TTFT/TPOT/QPS never get mixed across runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
MODEL="${MODEL:?Set MODEL to the local Qwen3-30B-A3B-FP8 snapshot directory}"
TRACE_ROOT="${SCRIPT_DIR}/../inputs/sharegpt_trace_20261013"
REPLAY="${SCRIPT_DIR}/replay_sharegpt_trace.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
EXACT_NAMES="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e3-calibration-20260825T091247Z/calibration_assets/exact_parameter_names.txt"
ADAPT_PLAN="${ADAPT_PLAN:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/provenance/plans/cold11.json}"
KT_PLAN_LOW="${KT_PLAN_LOW:-${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json}"
KT_GPU_EXPERTS="${KT_GPU_EXPERTS:-118}"
KT_PLAN_MID="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json"
KT_PLAN_HIGH="${REPO_ROOT}/experiments/vllm_uvm_e10_multilevel_pressure/inputs/cold89.json"
RUN_ROOT=""
SERVER_PID=""
CUSTOM_MODULE_ACTIVE=0
PORT="${PORT:-30521}"
MECHANISM="${1:-}"
LOAD="${2:-low}"
LIMIT="${LIMIT:-}"
WARMUP="${WARMUP:-4}"
WARMUP_SELECTION="${WARMUP_SELECTION:-trace_prefix}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-2147483648}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
NATIVE_OFFLOAD_GIB="${NATIVE_OFFLOAD_GIB:-2.322}"
NATIVE_EXACT_NAMES_FILE="${NATIVE_EXACT_NAMES_FILE:-}"
KT_MAX_TOTAL_TOKENS="${KT_MAX_TOTAL_TOKENS:-21840}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-7200}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"
ADAPT_PHASE_MODE="${ADAPT_PHASE_MODE:-auto}"
PARK_TOKEN_THRESHOLD="${PARK_TOKEN_THRESHOLD:-16}"
RESUME_TOKEN_THRESHOLD="${RESUME_TOKEN_THRESHOLD:-128}"
ARRIVAL_SCALE="${ARRIVAL_SCALE:-1.0}"
TRACE_FILE="${TRACE_FILE:-}"
REPLAY_MAX_CONNECTIONS="${REPLAY_MAX_CONNECTIONS:-64}"
SCHEDULER_TRACE_FILE="${SCHEDULER_TRACE_FILE:-${MIXED_SCHEDULER_TRACE:-}}"
unset MIXED_SCHEDULER_TRACE
ADAPT_DYNAMIC_TIERS="${ADAPT_DYNAMIC_TIERS:-0}"
ADAPT_LEGACY_INCREMENTAL_EXPERT="${ADAPT_LEGACY_INCREMENTAL_EXPERT:-0}"
ADAPT_DYNAMIC_FORCE_SYNC_MIGRATION="${ADAPT_DYNAMIC_FORCE_SYNC_MIGRATION:-0}"
# 2,048 physical blocks leave 2,047 policy-visible blocks: enough for the K3
# tier without reserving K4 capacity that this QPS=1 workload does not need.
ADAPT_DYNAMIC_KV_BYTES="${ADAPT_DYNAMIC_KV_BYTES:-3221225472}"
ADAPT_DYNAMIC_INITIAL_BLOCKS="${ADAPT_DYNAMIC_INITIAL_BLOCKS:-1364}"
ADAPT_DYNAMIC_POST_WARMUP_BLOCKS="${ADAPT_DYNAMIC_POST_WARMUP_BLOCKS:-${ADAPT_DYNAMIC_INITIAL_BLOCKS}}"
ADAPT_DYNAMIC_MAX_BLOCKS="${ADAPT_DYNAMIC_MAX_BLOCKS:-2047}"
ADAPT_DYNAMIC_MIN_BLOCKS="${ADAPT_DYNAMIC_MIN_BLOCKS:-${ADAPT_DYNAMIC_INITIAL_BLOCKS}}"
ADAPT_DYNAMIC_HIGH_WATERMARK="${ADAPT_DYNAMIC_HIGH_WATERMARK:-0.8}"
ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW="${ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW:-0}"
ADAPT_DYNAMIC_LOW_PRESSURE_STREAK="${ADAPT_DYNAMIC_LOW_PRESSURE_STREAK:-3}"
ADAPT_KV_POLICY_LABEL="${ADAPT_KV_POLICY_LABEL:-dynamic_k2_k3}"
ADAPT_DYNAMIC_BASE_PLAN="${ADAPT_DYNAMIC_BASE_PLAN:-${SCRIPT_DIR}/inputs/balanced_dynamic_plans_cold14_cold19/cold14.json}"
ADAPT_DYNAMIC_PLAN_CLASSES="${ADAPT_DYNAMIC_PLAN_CLASSES:-${SCRIPT_DIR}/plan_classes_k2_k3_balanced.json}"
ADAPT_DYNAMIC_INIT_PLAN="${ADAPT_DYNAMIC_INIT_PLAN:-${ADAPT_DYNAMIC_BASE_PLAN}}"
ADAPT_DYNAMIC_BACKUP_TARGET_MAP="${ADAPT_DYNAMIC_BACKUP_TARGET_MAP:-}"
ADAPT_DYNAMIC_BACKUP_GROW_BLOCKS="${ADAPT_DYNAMIC_BACKUP_GROW_BLOCKS:-683}"
ADAPT_DYNAMIC_PLAN_CLASS_MAP="${ADAPT_DYNAMIC_PLAN_CLASS_MAP:-}"
ADAPT_DYNAMIC_ALLOWED_TARGET_BLOCKS="${ADAPT_DYNAMIC_ALLOWED_TARGET_BLOCKS:-1364,2047}"
ADAPT_DYNAMIC_RESIZE_GRANULARITY_BLOCKS="${ADAPT_DYNAMIC_RESIZE_GRANULARITY_BLOCKS:-683}"
# Preserve the established alias granularity unless explicitly overridden for
# a diagnostic experiment. Formal K2/K5 reruns use the default 512 KiB.
ADAPT_ALIAS_REGION_BYTES="${ADAPT_ALIAS_REGION_BYTES:-524288}"

# Keep JSON defaults outside parameter expansion: an unescaped closing brace in
# ${var:-word} terminates the expansion and otherwise appends a stray `}` when
# callers provide an override.
if [[ -z "${ADAPT_DYNAMIC_BACKUP_TARGET_MAP}" ]]; then
  ADAPT_DYNAMIC_BACKUP_TARGET_MAP='{"1364":16,"2047":16}'
fi
if [[ -z "${ADAPT_DYNAMIC_PLAN_CLASS_MAP}" ]]; then
  ADAPT_DYNAMIC_PLAN_CLASS_MAP='{"1364":"cold14","2047":"cold19"}'
fi

die() { echo "sharegpt arm: $*" >&2; exit 1; }
[[ -n "${MECHANISM}" && -n "${LOAD}" ]] || die "usage: $0 {adaptmoe|native_uva|stock_uvm|native_copy|ktransformers} {low|medium|high} [run_root]"
case "${MECHANISM}" in adaptmoe|native_uva|stock_uvm|native_copy|ktransformers) ;; *) die "unknown mechanism ${MECHANISM}" ;; esac
case "${LOAD}" in low|medium|high) ;; *) die "unknown load ${LOAD}" ;; esac
case "${CUDA_GRAPH}" in 0|1) ;; *) die "CUDA_GRAPH must be 0 or 1" ;; esac
case "${ADAPT_DYNAMIC_TIERS}" in 0|1) ;; *) die "ADAPT_DYNAMIC_TIERS must be 0 or 1" ;; esac
case "${ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW}" in 0|1) ;; *) die "ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW must be 0 or 1" ;; esac
case "${ADAPT_LEGACY_INCREMENTAL_EXPERT}" in 0|1) ;; *) die "ADAPT_LEGACY_INCREMENTAL_EXPERT must be 0 or 1" ;; esac
case "${ADAPT_DYNAMIC_FORCE_SYNC_MIGRATION}" in 0|1) ;; *) die "ADAPT_DYNAMIC_FORCE_SYNC_MIGRATION must be 0 or 1" ;; esac
case "${ADAPT_PHASE_MODE}" in auto|prefill_remote_decode|prefill_graph_remote_decode|token_threshold) ;; *) die "unknown ADAPT_PHASE_MODE ${ADAPT_PHASE_MODE}" ;; esac
case "${WARMUP_SELECTION}" in trace_prefix|longest_prompt) ;; *) die "unknown WARMUP_SELECTION ${WARMUP_SELECTION}" ;; esac
[[ "${PARK_TOKEN_THRESHOLD}" =~ ^[1-9][0-9]*$ ]] || die "PARK_TOKEN_THRESHOLD must be positive"
[[ "${RESUME_TOKEN_THRESHOLD}" =~ ^[1-9][0-9]*$ ]] || die "RESUME_TOKEN_THRESHOLD must be positive"
(( PARK_TOKEN_THRESHOLD < RESUME_TOKEN_THRESHOLD )) || die "PARK_TOKEN_THRESHOLD must be below RESUME_TOKEN_THRESHOLD"
if [[ "${ADAPT_PHASE_MODE}" == token_threshold && "${CUDA_GRAPH}" != 0 ]]; then
  die "ADAPT_PHASE_MODE=token_threshold requires CUDA_GRAPH=0"
fi
if [[ "${MECHANISM}" == adaptmoe && "${ADAPT_DYNAMIC_TIERS}" == 1 ]]; then
  [[ "${CUDA_GRAPH}" == 0 ]] || die "ADAPT_DYNAMIC_TIERS=1 requires CUDA_GRAPH=0"
  [[ -f "${ADAPT_DYNAMIC_BASE_PLAN}" ]] || die "dynamic base plan missing: ${ADAPT_DYNAMIC_BASE_PLAN}"
  [[ -f "${ADAPT_DYNAMIC_PLAN_CLASSES}" ]] || die "dynamic plan classes missing: ${ADAPT_DYNAMIC_PLAN_CLASSES}"
fi
if [[ "${MECHANISM}" == adaptmoe && "${ADAPT_DYNAMIC_TIERS}" == 0 ]]; then
  [[ -f "${ADAPT_PLAN}" ]] || die "fixed AdaptMoE plan missing: ${ADAPT_PLAN}"
fi
if [[ -n "${NATIVE_EXACT_NAMES_FILE}" ]]; then
  [[ -f "${NATIVE_EXACT_NAMES_FILE}" ]] || die "native exact-name list missing: ${NATIVE_EXACT_NAMES_FILE}"
fi
RUN_ROOT="${3:-${SCRIPT_DIR}/artifacts/sharegpt-${MECHANISM}-${LOAD}-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "${RUN_ROOT}" ]] || die "run root already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

if [[ "${MECHANISM}" == adaptmoe && "${ADAPT_DYNAMIC_TIERS}" == 1 ]]; then
  KV_CACHE_BYTES="${ADAPT_DYNAMIC_KV_BYTES}"
fi

trace="${TRACE_ROOT}/sharegpt_qwen_trace_${LOAD}.jsonl"
if [[ -n "${TRACE_FILE}" ]]; then trace="${TRACE_FILE}"; fi
[[ -f "${trace}" && -f "${MODEL}/config.json" && -x "${VLLM_PY}" ]] || die "trace/model/vLLM missing"
[[ -f "${REPLAY}" ]] || die "replay client missing"
if [[ -n "${LIMIT}" ]]; then limit_args=(--limit "${LIMIT}"); else limit_args=(); fi

gpu_clean() {
  local used pids
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
  [[ "${used}" == 0 && -z "${pids}" ]]
}
wait_gpu_clean() { for _ in $(seq 1 240); do gpu_clean && return 0; sleep 1; done; return 1; }

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

capture_kernel_diagnostics() {
  [[ "${CUSTOM_MODULE_ACTIVE}" == 1 ]] || return 0
  local diagnostic_name
  for diagnostic_name in phase_f_route_log phase_f_route_log_stats; do
    if [[ -r "/proc/driver/nvidia-uvm/${diagnostic_name}" ]]; then
      cp -- "/proc/driver/nvidia-uvm/${diagnostic_name}" \
        "${RUN_ROOT}/${diagnostic_name}.txt" || true
    fi
  done
  sudo -n dmesg >"${RUN_ROOT}/dmesg.after.txt" 2>/dev/null || true
}

cleanup_run() {
  capture_kernel_diagnostics
  cleanup_server
  if [[ "${CUSTOM_MODULE_ACTIVE}" == 1 ]]; then
    set +e
    capture_and_stop_exporters
    restore_stock
    restart_exporters
    CUSTOM_MODULE_ACTIVE=0
    set -e
  fi
}
trap 'cleanup_run' EXIT INT TERM

wait_ready() {
  for _ in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 200 "${RUN_ROOT}/server.log" >&2; return 1; }
    sleep 1
  done
  return 1
}

common_env=(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_OPENAI_METRICS=1)
compilation='{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,4,8,16],"max_cudagraph_capture_size":16}'
if [[ "${CUDA_GRAPH}" == 0 ]]; then
  compilation='{"mode":"NONE","cudagraph_mode":"NONE"}'
fi

write_provenance() {
  sha256sum "${trace}" "${MODEL}/config.json" "${REPLAY}" >"${RUN_ROOT}/inputs.sha256"
  printf '%s\n' "model=${MODEL}" "trace=${trace}" "trace_sha256=$(sha256sum "${trace}" | awk '{print $1}')" "mechanism=${MECHANISM}" "load=${LOAD}" "arrival_scale=${ARRIVAL_SCALE}" "warmup=${WARMUP}" "warmup_selection=${WARMUP_SELECTION}" "cuda_graph=${CUDA_GRAPH}" "max_num_seqs=${MAX_NUM_SEQS}" "max_batched_tokens=${MAX_BATCHED}" "kv_cache_bytes=${KV_CACHE_BYTES}" "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" "native_offload_gib=${NATIVE_OFFLOAD_GIB}" "kt_max_total_tokens=${KT_MAX_TOTAL_TOKENS}" "adapt_phase_mode=${ADAPT_PHASE_MODE}" "adapt_plan=${ADAPT_PLAN}" "park_token_threshold=${PARK_TOKEN_THRESHOLD}" "resume_token_threshold=${RESUME_TOKEN_THRESHOLD}" "adapt_dynamic_tiers=${ADAPT_DYNAMIC_TIERS}" "adapt_kv_policy_label=${ADAPT_KV_POLICY_LABEL}" "adapt_legacy_incremental_expert=${ADAPT_LEGACY_INCREMENTAL_EXPERT}" "adapt_dynamic_force_sync_migration=${ADAPT_DYNAMIC_FORCE_SYNC_MIGRATION}" "adapt_dynamic_kv_bytes=${ADAPT_DYNAMIC_KV_BYTES}" "adapt_dynamic_initial_blocks=${ADAPT_DYNAMIC_INITIAL_BLOCKS}" "adapt_dynamic_post_warmup_blocks=${ADAPT_DYNAMIC_POST_WARMUP_BLOCKS}" "adapt_dynamic_min_blocks=${ADAPT_DYNAMIC_MIN_BLOCKS}" "adapt_dynamic_max_blocks=${ADAPT_DYNAMIC_MAX_BLOCKS}" "adapt_dynamic_high_watermark=${ADAPT_DYNAMIC_HIGH_WATERMARK}" "adapt_dynamic_low_pressure_streak=${ADAPT_DYNAMIC_LOW_PRESSURE_STREAK}" "adapt_dynamic_base_plan=${ADAPT_DYNAMIC_BASE_PLAN}" "adapt_dynamic_plan_classes=${ADAPT_DYNAMIC_PLAN_CLASSES}" "adapt_dynamic_init_plan=${ADAPT_DYNAMIC_INIT_PLAN}" "adapt_dynamic_backup_target_map=${ADAPT_DYNAMIC_BACKUP_TARGET_MAP}" "adapt_dynamic_backup_grow_blocks=${ADAPT_DYNAMIC_BACKUP_GROW_BLOCKS}" "adapt_dynamic_plan_class_map=${ADAPT_DYNAMIC_PLAN_CLASS_MAP}" "adapt_dynamic_allowed_target_blocks=${ADAPT_DYNAMIC_ALLOWED_TARGET_BLOCKS}" "adapt_dynamic_resize_granularity_blocks=${ADAPT_DYNAMIC_RESIZE_GRANULARITY_BLOCKS}" "adapt_alias_region_bytes=${ADAPT_ALIAS_REGION_BYTES}" >"${RUN_ROOT}/configuration.txt"
  nvidia-smi -q >"${RUN_ROOT}/nvidia-smi.before.txt"
  printf '%s\n' "adapt_dynamic_active_high_watermark_grow=${ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW}" >>"${RUN_ROOT}/configuration.txt"
  printf '%s\n' "kt_plan=${KT_PLAN_LOW}" "kt_gpu_experts=${KT_GPU_EXPERTS}" >>"${RUN_ROOT}/configuration.txt"
  printf '%s\n' "native_exact_names_file=${NATIVE_EXACT_NAMES_FILE}" >>"${RUN_ROOT}/configuration.txt"
}

launch_vllm() {
  local backend="$1" offload="$2" phase="$3" plan="$4" case_dir="$5"
  local -a envs=("${common_env[@]}" VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_CPU_OFFLOAD_BACKEND="${backend}" VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/offload_manifest.jsonl" VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 VLLM_CPU_OFFLOAD_COPY_SYNC=0 VLLM_CPU_OFFLOAD_COPY_TRACE=0)
  if [[ -n "${SCHEDULER_TRACE_FILE:-}" ]]; then
    envs+=(MIXED_SCHEDULER_TRACE="${SCHEDULER_TRACE_FILE}")
  fi
  if [[ "${MECHANISM}" == stock_uvm ]]; then
    envs+=(
      VLLM_USE_UVM=1
      PYTORCH_ALLOC_CONF=expandable_segments:True
      VLLM_UVM_PREFETCH=0
      VLLM_UVM_VERBOSE=0
      VLLM_UVM_POLICY_ENABLE=0
      VLLM_UVM_POLICY_WARMUP_PREFETCH=0
      VLLM_UVM_POLICY_WARMUP_ADVISE_GPU=0
      VLLM_UVM_GAP_WATCH_ENABLE=0
      VLLM_UVM_DEVICE_DIRECT_ENABLE=0
      VLLM_UVM_KV_DEVICE_DIRECT_ENABLE=0
      VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0
      VLLM_UVM_SCRATCH_POOL_ENABLE=0
      VLLM_UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE=0
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
      VLLM_UVM_MATCHED_CPU_OFFLOAD_ENABLE=0
      VLLM_UVM_KV_RUNTIME_ENABLE=0
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
      VLLM_UVM_LOG_FILE="${case_dir}/uvm_allocations.log"
    )
  else
    envs+=(VLLM_USE_UVM=0)
  fi
  local arm_compilation="${compilation}"
  if [[ "${MECHANISM}" == adaptmoe && "${CUDA_GRAPH}" == 1 ]]; then
    arm_compilation='{"mode":"NONE","cudagraph_mode":"FULL_SEPARATE","cudagraph_capture_sizes":[1,4,8,16,128,512,1024,2048,4096,8192],"max_cudagraph_capture_size":8192}'
  fi
  local -a args=(--host 127.0.0.1 --port "${PORT}" --disable-frontend-multiprocessing --model "${MODEL}" --trust-remote-code --dtype auto --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --kv-cache-memory-bytes "${KV_CACHE_BYTES}" --cpu-offload-gb "${offload}" --max-num-batched-tokens "${MAX_BATCHED}" --max-num-seqs "${MAX_NUM_SEQS}" --no-enable-prefix-caching --generation-config vllm --compilation-config "${arm_compilation}")
  if [[ "${CUDA_GRAPH}" == 0 ]]; then args+=(--enforce-eager); fi
  if [[ "${phase}" != off ]]; then
    envs+=(VLLM_UVM_ORIGINAL_VA_PHASE_MODE="${phase}")
    if [[ "${phase}" == token_threshold ]]; then
      envs+=(
        VLLM_UVM_PHASE_BATCHED_SCHEDULING=0
        VLLM_UVM_PHASE_KV_AWARE_COHORTS=0
        VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD="${PARK_TOKEN_THRESHOLD}"
        VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD="${RESUME_TOKEN_THRESHOLD}"
        VLLM_UVM_TOKEN_THRESHOLD_FAST_TRANSITION=1
        VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=0
        VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0
        VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=0
        VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0
        VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT=0
        VLLM_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT=0
        VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_CAPTURE_WARMUP=0
      )
    else
      envs+=(VLLM_UVM_PHASE_BATCHED_SCHEDULING=1 VLLM_UVM_PHASE_KV_AWARE_COHORTS=1 VLLM_UVM_PHASE_COHORT_KV_RESERVE_BLOCKS=1)
    fi
    if [[ "${CUDA_GRAPH}" == 1 ]]; then
      envs+=(VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1 VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=1 VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1)
    elif [[ "${phase}" != token_threshold ]]; then
      # Eager phase serving still needs the audited all-remote startup
      # topology for vLLM's dummy kernel warmup.  The post-warmup residency
      # gate restores hot experts and activates the sealed alias topology
      # before serving traffic.
      envs+=(VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0 VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=0 VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1 VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=0 VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0 VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT=0)
    fi
  fi
  # Native Copy uses the regular byte-budget selector.  The calibration
  # exact-name list is a frozen expert-plan list for AdaptMoE and can exceed
  # a small native budget before the loader reaches the intended boundary.
  if [[ "${backend}" == copy && "${MECHANISM}" == adaptmoe ]]; then envs+=(VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${EXACT_NAMES}"); fi
  if [[ "${MECHANISM}" == native_uva || "${MECHANISM}" == native_copy ]]; then
    if [[ -n "${NATIVE_EXACT_NAMES_FILE}" ]]; then
      envs+=(VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NATIVE_EXACT_NAMES_FILE}")
    fi
  fi
  if [[ -n "${plan}" ]]; then
    envs+=(VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${plan}" VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES="${ADAPT_ALIAS_REGION_BYTES}" VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592)
    if [[ "${MECHANISM}" == adaptmoe && "${ADAPT_DYNAMIC_TIERS}" == 1 ]]; then
      envs+=(
        VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE="${ADAPT_DYNAMIC_PLAN_CLASSES}"
        VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES=1
        VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1
        VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1
        VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1
        VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE=1
        VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA=1
        VLLM_CPU_OFFLOAD_PLAN_CLASS_INCREMENTAL_DUAL_REFRESH=1
        VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW="${VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW:-0}"
        VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW="${VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW:-0}"
        VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE=1
        VLLM_CPU_OFFLOAD_PLAN_CLASS_FORCE_SYNC_MIGRATION="${ADAPT_DYNAMIC_FORCE_SYNC_MIGRATION}"
        # E13's plan-class transaction owns the initial complete dual view.
        # Runtime incremental waves remain enabled, but the E7 one-shot
        # Stage-M pass is intentionally disabled because it would release
        # records before the initial dual-view seal.
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
        VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE="${ADAPT_LEGACY_INCREMENTAL_EXPERT}"
        VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE="${ADAPT_LEGACY_INCREMENTAL_EXPERT}"
        VLLM_UVM_DYNAMIC_EXPERT_AUTO_REBALANCE_ENABLE=0
        VLLM_UVM_DYNAMIC_EXPERT_HEAT_EWMA_ALPHA=0.25
        VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
        VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
        VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read
        VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
        VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE=0
        VLLM_UVM_KV_CACHE_MANAGED=1
        VLLM_UVM_KV_CACHE_LAZY_RESERVE=1
        VLLM_UVM_KV_RUNTIME_ENABLE=1
        VLLM_UVM_KV_RUNTIME_MODE=enforce
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1
        VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE="${case_dir}/kv_dynamic_pool.control"
        VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${ADAPT_DYNAMIC_INITIAL_BLOCKS}"
        VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS="${ADAPT_DYNAMIC_POST_WARMUP_BLOCKS}"
        VLLM_UVM_KV_BACKUP_RESERVE_BLOCKS=16
        VLLM_UVM_KV_BACKUP_RESERVE_TARGET_MAP="${ADAPT_DYNAMIC_BACKUP_TARGET_MAP}"
        VLLM_UVM_KV_BACKUP_GROW_BLOCKS="${ADAPT_DYNAMIC_BACKUP_GROW_BLOCKS}"
        VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE=0
        VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0
        VLLM_UVM_KV_AUTO_SCALE_ENABLE=1
        VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE=1
        VLLM_UVM_KV_AUTO_SCALE_SYNCHRONIZED_TIER_STEPS=1
        VLLM_UVM_KV_AUTO_SCALE_BACKUP_ONLY=0
        VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP="${ADAPT_DYNAMIC_PLAN_CLASS_MAP}"
        VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS="${ADAPT_DYNAMIC_ALLOWED_TARGET_BLOCKS}"
        VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_BEFORE_GROW=1
        VLLM_UVM_KV_AUTO_SCALE_RESUME_PREFILL_BEFORE_PLAN_CLASS=1
        VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK="${ADAPT_DYNAMIC_LOW_PRESSURE_STREAK}"
        VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR=2
        VLLM_UVM_KV_AUTO_SCALE_SHRINK_DIVISOR=2
        VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS="${ADAPT_DYNAMIC_MIN_BLOCKS}"
        VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS="${ADAPT_DYNAMIC_MAX_BLOCKS}"
        VLLM_UVM_KV_AUTO_SCALE_SAFETY_BLOCKS=16
        VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK="${ADAPT_DYNAMIC_HIGH_WATERMARK}"
        VLLM_UVM_KV_AUTO_SCALE_ACTIVE_HIGH_WATERMARK_GROW="${ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW}"
        VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS=4
        VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS="${ADAPT_DYNAMIC_RESIZE_GRANULARITY_BLOCKS}"
        VLLM_UVM_KV_AUTO_SCALE_EXPERT_RESTORE_ENABLE=1
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1
        VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR=cpp_batch
        VLLM_UVM_KV_ADAPTIVE_FAIL_FAST=1
        VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1
        VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=0
        VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=0
        VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=0
        VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE=0
      )
    else
      envs+=(VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0)
    fi
  fi
  if [[ -n "${plan}" ]]; then
    if [[ "${MECHANISM}" == adaptmoe && "${ADAPT_DYNAMIC_TIERS}" == 1 ]]; then
      # E13's complete dual-view protocol keeps these diagnostic CUDA reads
      # disabled after sealing.  Re-enabling them here invalidates the
      # certified parent/PTE children before the first PARK.
      envs+=(VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=0 VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE=0)
    elif [[ "${phase}" == token_threshold ]]; then
      envs+=(VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=0 VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE=0)
    else
      envs+=(VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=1 VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE=1)
    fi
    envs+=(VLLM_FUSED_MOE_CHUNK_SIZE=16384 VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=1)
    if [[ "${MECHANISM}" == adaptmoe && "${ADAPT_DYNAMIC_TIERS}" == 1 ]]; then
      envs+=(VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=0)
    else
      envs+=(VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=2147483648)
    fi
  fi
  printf '%q ' env "${envs[@]}" taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server "${args[@]}" >"${RUN_ROOT}/server_command.txt"; printf '\n' >>"${RUN_ROOT}/server_command.txt"
  env "${envs[@]}" taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server "${args[@]}" >"${RUN_ROOT}/server.log" 2>&1 &
  SERVER_PID="$!"
}

launch_kt() {
  local plan="$1" gpu_experts="$2" case_dir="$3"
  local -a envs=("${common_env[@]}" PATH="${KT_ROOT}/.venv/bin:${PATH}" SGLANG_EXPERT_DISTRIBUTION_RECORDER_MODE=off)
  local -a args=(--model-path "${MODEL}" --host 127.0.0.1 --port "${PORT}" --tp-size 1 --context-length 8194 --max-running-requests "${MAX_NUM_SEQS}" --mem-fraction-static 0.97 --max-total-tokens "${KT_MAX_TOTAL_TOKENS}" --chunked-prefill-size "${MAX_BATCHED}" --max-prefill-tokens "${MAX_BATCHED}" --schedule-policy fcfs --stream-interval 1 --random-seed 20261013 --trust-remote-code --kt-weight-path "${MODEL}" --kt-method FP8 --kt-cpuinfer 7 --kt-threadpool-count 1 --kt-numa-nodes 0 --kt-num-gpu-experts "${gpu_experts}" --kt-expert-placement-strategy frequency --init-expert-location "${plan}" --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0 --attention-backend triton --cuda-graph-max-bs "${MAX_NUM_SEQS}" --cuda-graph-bs 1 4 8 16 --enable-mixed-chunk --disable-shared-experts-fusion --disable-radix-cache --enable-metrics)
  if [[ "${CUDA_GRAPH}" == 0 ]]; then args+=(--disable-cuda-graph); fi
  printf '%q ' env "${envs[@]}" taskset -c 0-13 "${KT_PY}" -m sglang.launch_server "${args[@]}" >"${RUN_ROOT}/server_command.txt"; printf '\n' >>"${RUN_ROOT}/server_command.txt"
  env "${envs[@]}" taskset -c 0-13 "${KT_PY}" -m sglang.launch_server "${args[@]}" >"${RUN_ROOT}/server.log" 2>&1 &
  SERVER_PID="$!"
}

write_provenance
wait_gpu_clean || die "GPU is not clean before run"
case "${MECHANISM}" in
  native_uva) launch_vllm uva "${NATIVE_OFFLOAD_GIB}" off '' "${RUN_ROOT}" ;;
  stock_uvm) launch_vllm copy 0 off '' "${RUN_ROOT}" ;;
  native_copy) launch_vllm copy "${NATIVE_OFFLOAD_GIB}" off '' "${RUN_ROOT}" ;;
  adaptmoe)
    export VLLM_PHASE_SWITCH_SOURCE_ONLY=1 ALLOW_GPU_RUN=1 \
      RUN_ROOT_ACTIVE="${RUN_ROOT}" SLOT_BUDGET_BYTES=1207959552 \
      ACCESS_COUNTER_MIGRATION_ENABLE=0 UVM_PREFETCH_ENABLE=0 \
      UVM_THRASHING_ENABLE=0 STATIC_ALIAS_DUAL_VIEW_ENABLE=1 \
      STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
    # shellcheck source=/dev/null
    source "${BASE_HARNESS}"
    # The sourced harness installs its own EXIT trap. Restore this launcher's
    # cleanup so renewed exporter UVM clients are stopped before module unload.
    trap 'cleanup_run' EXIT
    trap 'cleanup_run; exit 130' INT
    trap 'cleanup_run; exit 143' TERM
    runtime_preflight
    module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
    load_custom_module
    CUSTOM_MODULE_ACTIVE=1
    module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
    if [[ "${ADAPT_PHASE_MODE}" != auto ]]; then
      adapt_phase="${ADAPT_PHASE_MODE}"
    elif [[ "${CUDA_GRAPH}" == 1 ]]; then
      adapt_phase=prefill_graph_remote_decode
    else
      adapt_phase=prefill_remote_decode
    fi
    if [[ "${ADAPT_DYNAMIC_TIERS}" == 1 ]]; then
      adapt_plan="${ADAPT_DYNAMIC_BASE_PLAN}"
    else
      adapt_plan="${ADAPT_PLAN}"
    fi
    launch_vllm copy 30 "${adapt_phase}" "${adapt_plan}" "${RUN_ROOT}"
    ;;
  ktransformers)
    kt_converter="${REPO_ROOT}/experiments/ktransformers_cpu_expert_compare/cold_plan_to_kt_stats.py"
    # Keep the same near-equal 2 GiB placement for all load levels.  Changing
    # placement with offered load would confound the load-scaling comparison.
    kt_input="${KT_PLAN_LOW}"
    kt_gpu="${KT_GPU_EXPERTS}"
    kt_out="${RUN_ROOT}/kt_plan"
    mkdir -p "${kt_out}"
    "${KT_PY}" "${kt_converter}" "${kt_input}" --output-dir "${kt_out}" >"${RUN_ROOT}/kt_plan_conversion.json"
    kt_pt="${kt_out}/$(basename "${kt_input}" .json)_logical_count.pt"
    launch_kt "${kt_pt}" "${kt_gpu}" "${RUN_ROOT}"
    ;;
esac
wait_ready || die "server did not become ready"
set +e
"${VLLM_PY}" "${REPLAY}" --trace "${trace}" --base-url "http://127.0.0.1:${PORT}" --model "${MODEL}" --output "${RUN_ROOT}/replay.json" "${limit_args[@]}" --warmup "${WARMUP}" --warmup-selection "${WARMUP_SELECTION}" --max-connections "${REPLAY_MAX_CONNECTIONS}" --timeout-seconds "${TIMEOUT_SECONDS}" --arrival-scale "${ARRIVAL_SCALE}" >"${RUN_ROOT}/replay.stdout" 2>"${RUN_ROOT}/replay.stderr"
replay_rc=$?
set -e
capture_kernel_diagnostics
cleanup_server
if [[ "${MECHANISM}" == adaptmoe ]]; then
  # Exporters may be restarted by their service policy during a long replay;
  # close any renewed UVM client before cycling back to the stock module.
  capture_and_stop_exporters
  restore_stock || die "failed to restore stock UVM"
  CUSTOM_MODULE_ACTIVE=0
  module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
fi
wait_gpu_clean || die "GPU did not clean after run"
nvidia-smi -q >"${RUN_ROOT}/nvidia-smi.after.txt"
if [[ "${MECHANISM}" == adaptmoe ]]; then
  restart_exporters
fi
echo "run_root=${RUN_ROOT} replay_rc=${replay_rc}"
[[ -f "${RUN_ROOT}/replay.json" ]] && jq '{status,request_count,successful_requests,qps,offered_qps,ttft_ms,tpot_ms,e2e_ms}' "${RUN_ROOT}/replay.json" || true
exit "${replay_rc}"
