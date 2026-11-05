#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K5_COLD_EXPERTS="${K5_COLD_EXPERTS:-29}"
case "${K5_COLD_EXPERTS}" in
  29) K5_CLASSES=plan_classes_k2_k5_hbm_balanced.json ;;
  30) K5_CLASSES=plan_classes_k2_k5_hbm_headroom.json ;;
  32) K5_CLASSES=plan_classes_k2_k5_hbm_safe.json ;;
  34) K5_CLASSES=plan_classes_k2_k5_hbm_safe_cold34.json ;;
  *) echo "Unsupported cold tier: ${K5_COLD_EXPERTS}" >&2; exit 1 ;;
esac
export CUDA_GRAPH=0 ADAPT_PHASE_MODE=token_threshold ADAPT_DYNAMIC_TIERS=1
export VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW=1
export ARRIVAL_SCALE=0.13185410310639004
export ADAPT_DYNAMIC_KV_BYTES=5369757696
export ADAPT_DYNAMIC_INITIAL_BLOCKS=1364 ADAPT_DYNAMIC_POST_WARMUP_BLOCKS=1364
export ADAPT_DYNAMIC_MIN_BLOCKS=1364 ADAPT_DYNAMIC_MAX_BLOCKS=3413
export ADAPT_DYNAMIC_ALLOWED_TARGET_BLOCKS=1364,3413
export ADAPT_DYNAMIC_RESIZE_GRANULARITY_BLOCKS=2049
export ADAPT_DYNAMIC_BACKUP_GROW_BLOCKS=2049
export ADAPT_DYNAMIC_BACKUP_TARGET_MAP='{"1364":16,"3413":16}'
export ADAPT_DYNAMIC_BASE_PLAN="${SCRIPT_DIR}/inputs/balanced_dynamic_plans_cold14_cold${K5_COLD_EXPERTS}/cold14.json"
export ADAPT_DYNAMIC_PLAN_CLASSES="${SCRIPT_DIR}/${K5_CLASSES}"
export ADAPT_DYNAMIC_INIT_PLAN="${ADAPT_DYNAMIC_BASE_PLAN}"
export ADAPT_DYNAMIC_PLAN_CLASS_MAP="{\"1364\":\"cold14\",\"3413\":\"cold${K5_COLD_EXPERTS}\"}"
export ADAPT_KV_POLICY_LABEL="dynamic_k2_k5_cold${K5_COLD_EXPERTS}_qps2"
exec bash "${SCRIPT_DIR}/run_sharegpt_trace_arm.sh" adaptmoe low "${1:?run root required}"
