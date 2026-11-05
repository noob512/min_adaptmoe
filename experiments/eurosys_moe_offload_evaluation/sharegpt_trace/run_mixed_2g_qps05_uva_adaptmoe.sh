#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MATRIX_ROOT="${1:?new absolute artifact root required}"
[[ ! -e "${MATRIX_ROOT}" ]] || { echo 'Matrix root already exists' >&2; exit 1; }
mkdir -p "${MATRIX_ROOT}"
MATRIX_ROOT="$(cd "${MATRIX_ROOT}" && pwd)"
PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
OFFERED_QPS="${OFFERED_QPS:-0.5}"
read -r -a trace_names <<<"${TRACE_NAMES:-short medium long}"
"${PY}" "${SCRIPT_DIR}/build_bounded_mixed_trace.py" \
  --source "${SCRIPT_DIR}/../inputs/sharegpt_trace_20261013/sharegpt_qwen_trace_low.jsonl" \
  --output-dir "${MATRIX_ROOT}/traces" --qps "${OFFERED_QPS}" \
  --traces "${trace_names[@]}" >"${MATRIX_ROOT}/trace_generation.json"
export CUDA_GRAPH=0 ADAPT_PHASE_MODE=token_threshold ARRIVAL_SCALE=1
export PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128
export PHASE_F_ROUTE_LOG_ENABLE=0
export MAX_NUM_SEQS=64 MAX_BATCHED=8192 MAX_MODEL_LEN=2048
export REPLAY_MAX_CONNECTIONS=96 WARMUP=4 TIMEOUT_SECONDS=3600
# Match actual selected bytes, not just the nominal "2 GiB" budget.
export KV_CACHE_BYTES=1610612736 NATIVE_OFFLOAD_GIB=2.109375
export NATIVE_EXACT_NAMES_FILE="${SCRIPT_DIR}/inputs/native_uva_cold10_matched_names.txt"
# One fixed plan and one fixed KV target. The dual-view initialization is
# retained, while no capacity or expert-tier change is permitted.
export ADAPT_DYNAMIC_TIERS=1 ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW=0
export ADAPT_DYNAMIC_KV_BYTES=1610612736
export ADAPT_DYNAMIC_INITIAL_BLOCKS=1023 ADAPT_DYNAMIC_POST_WARMUP_BLOCKS=1023
export ADAPT_DYNAMIC_MIN_BLOCKS=1023 ADAPT_DYNAMIC_MAX_BLOCKS=1023
export ADAPT_DYNAMIC_ALLOWED_TARGET_BLOCKS=1023
export ADAPT_DYNAMIC_RESIZE_GRANULARITY_BLOCKS=1 ADAPT_DYNAMIC_BACKUP_GROW_BLOCKS=1
export ADAPT_DYNAMIC_BACKUP_TARGET_MAP='{"1023":16}'
export ADAPT_DYNAMIC_BASE_PLAN="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json"
export ADAPT_DYNAMIC_PLAN_CLASSES="${SCRIPT_DIR}/plan_class_fixed_cold10.json"
export ADAPT_DYNAMIC_PLAN_CLASS_MAP='{"1023":"cold10"}'
export ADAPT_DYNAMIC_HIGH_WATERMARK=0.8 ADAPT_DYNAMIC_LOW_PRESSURE_STREAK=1000000
export ADAPT_KV_POLICY_LABEL=fixed_k1_5_cold10
export PYTHONPATH="${SCRIPT_DIR}/mixed_instrumentation${PYTHONPATH:+:${PYTHONPATH}}"
cp -- "${BASH_SOURCE[0]}" "${MATRIX_ROOT}/matrix_script.sh"
cp -- "${SCRIPT_DIR}/run_sharegpt_trace_arm.sh" "${MATRIX_ROOT}/arm_script.sh"
cp -- "${NATIVE_EXACT_NAMES_FILE}" "${MATRIX_ROOT}/native_exact_names.txt"
cp -- "${ADAPT_DYNAMIC_BASE_PLAN}" "${MATRIX_ROOT}/adapt_plan.json"
cp -- "${ADAPT_DYNAMIC_PLAN_CLASSES}" "${MATRIX_ROOT}/adapt_plan_class.json"
for trace_name in "${trace_names[@]}"; do
  export TRACE_FILE="${MATRIX_ROOT}/traces/${trace_name}.jsonl"
  for mechanism in native_uva adaptmoe; do
    run_dir="${MATRIX_ROOT}/${trace_name}/${mechanism}"
    mkdir -p "${MATRIX_ROOT}/${trace_name}"
    export SCHEDULER_TRACE_FILE="${run_dir}/scheduler_steps.jsonl"
    printf '%s starting %s/%s\n' "$(date -Is)" "${trace_name}" "${mechanism}" | tee -a "${MATRIX_ROOT}/progress.log"
    if ! bash "${SCRIPT_DIR}/run_sharegpt_trace_arm.sh" "${mechanism}" low "${run_dir}" >"${MATRIX_ROOT}/${trace_name}/${mechanism}.launcher.log" 2>&1; then
      printf '%s FAILED %s/%s; matrix stopped\n' "$(date -Is)" "${trace_name}" "${mechanism}" | tee -a "${MATRIX_ROOT}/progress.log"
      exit 1
    fi
    printf '%s completed %s/%s\n' "$(date -Is)" "${trace_name}" "${mechanism}" | tee -a "${MATRIX_ROOT}/progress.log"
  done
done
"${PY}" "${SCRIPT_DIR}/analyze_mixed_2g.py" "${MATRIX_ROOT}" >"${MATRIX_ROOT}/analysis.stdout.json"
