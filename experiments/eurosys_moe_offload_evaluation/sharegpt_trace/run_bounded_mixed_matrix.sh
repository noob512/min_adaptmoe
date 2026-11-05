#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MATRIX_ROOT="${1:?new absolute artifact root required}"
[[ ! -e "${MATRIX_ROOT}" ]] || { echo 'Matrix root already exists' >&2; exit 1; }
mkdir -p "${MATRIX_ROOT}"
MATRIX_ROOT="$(cd "${MATRIX_ROOT}" && pwd)"
PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
"${PY}" "${SCRIPT_DIR}/build_bounded_mixed_trace.py" \
  --source "${SCRIPT_DIR}/../inputs/sharegpt_trace_20261013/sharegpt_qwen_trace_low.jsonl" \
  --output-dir "${MATRIX_ROOT}/traces" >"${MATRIX_ROOT}/trace_generation.json"
export CUDA_GRAPH=0 ADAPT_PHASE_MODE=token_threshold ARRIVAL_SCALE=1
export PHASE_F_ROUTE_LOG_ENABLE=0
export MAX_NUM_SEQS=64 MAX_BATCHED=8192 MAX_MODEL_LEN=2048
export REPLAY_MAX_CONNECTIONS=96 WARMUP=4 TIMEOUT_SECONDS=3600
export KV_CACHE_BYTES=4295491584 NATIVE_OFFLOAD_GIB=5.907
export KT_MAX_TOTAL_TOKENS=43696 KT_GPU_EXPERTS=100
export KT_PLAN_LOW="${SCRIPT_DIR}/inputs/balanced_dynamic_plans_cold14_cold28/cold28.json"
export VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW=1 VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW=1
export ADAPT_DYNAMIC_TIERS=1 ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW=1
export ADAPT_DYNAMIC_KV_BYTES=4295491584
export ADAPT_DYNAMIC_INITIAL_BLOCKS=1364 ADAPT_DYNAMIC_POST_WARMUP_BLOCKS=1364
export ADAPT_DYNAMIC_RESIZE_GRANULARITY_BLOCKS=1366 ADAPT_DYNAMIC_BACKUP_GROW_BLOCKS=1366
export ADAPT_DYNAMIC_BACKUP_TARGET_MAP='{"1364":16,"2730":16}'
export ADAPT_DYNAMIC_BASE_PLAN="${SCRIPT_DIR}/inputs/balanced_dynamic_plans_cold14_cold28/cold14.json"
export ADAPT_DYNAMIC_PLAN_CLASSES="${SCRIPT_DIR}/plan_classes_k2_k4_cold28.json"
export ADAPT_DYNAMIC_PLAN_CLASS_MAP='{"1364":"cold14","2730":"cold28"}'
export PYTHONPATH="${SCRIPT_DIR}/mixed_instrumentation${PYTHONPATH:+:${PYTHONPATH}}"
for trace_name in short medium long; do
  export TRACE_FILE="${MATRIX_ROOT}/traces/${trace_name}.jsonl"
  for arm in dynamic fixed_k2 fixed_k4 native_uva native_copy ktransformers; do
    export ADAPT_DYNAMIC_MIN_BLOCKS=1364 ADAPT_DYNAMIC_MAX_BLOCKS=2730
    export ADAPT_DYNAMIC_ALLOWED_TARGET_BLOCKS=1364,2730
    export ADAPT_DYNAMIC_HIGH_WATERMARK=0.8 ADAPT_DYNAMIC_LOW_PRESSURE_STREAK=3
    export ADAPT_KV_POLICY_LABEL="${arm}"
    mechanism="${arm}"
    case "${arm}" in
      dynamic) mechanism=adaptmoe ;;
      fixed_k2)
        mechanism=adaptmoe
        export ADAPT_DYNAMIC_MAX_BLOCKS=1364 ADAPT_DYNAMIC_ALLOWED_TARGET_BLOCKS=1364
        ;;
      fixed_k4)
        mechanism=adaptmoe
        # Grow using the validated transaction during warmup, then hold K4.
        export ADAPT_DYNAMIC_HIGH_WATERMARK=0.01 ADAPT_DYNAMIC_LOW_PRESSURE_STREAK=1000000
        ;;
    esac
    run_dir="${MATRIX_ROOT}/${trace_name}/${arm}"
    mkdir -p "${MATRIX_ROOT}/${trace_name}"
    export SCHEDULER_TRACE_FILE="${run_dir}/scheduler_steps.jsonl"
    printf '%s starting %s/%s\n' "$(date -Is)" "${trace_name}" "${arm}" | tee -a "${MATRIX_ROOT}/progress.log"
    if ! bash "${SCRIPT_DIR}/run_sharegpt_trace_arm.sh" "${mechanism}" low "${run_dir}" >"${MATRIX_ROOT}/${trace_name}/${arm}.launcher.log" 2>&1; then
      printf '%s FAILED %s/%s; matrix stopped\n' "$(date -Is)" "${trace_name}" "${arm}" | tee -a "${MATRIX_ROOT}/progress.log"
      exit 1
    fi
    printf '%s completed %s/%s\n' "$(date -Is)" "${trace_name}" "${arm}" | tee -a "${MATRIX_ROOT}/progress.log"
  done
done
