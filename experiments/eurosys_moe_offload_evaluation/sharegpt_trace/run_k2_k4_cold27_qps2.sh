#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Retain K2/K4 and QPS=2; release three more experts per layer at K4 to
# accommodate the physical hot-expert loss observed with cold24.
export CUDA_GRAPH=0 ADAPT_PHASE_MODE=token_threshold ADAPT_DYNAMIC_TIERS=1
export VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW=1
export VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW=1
export ARRIVAL_SCALE=0.13185410310639004
export ADAPT_DYNAMIC_KV_BYTES=4295491584
export ADAPT_DYNAMIC_INITIAL_BLOCKS=1364 ADAPT_DYNAMIC_POST_WARMUP_BLOCKS=1364
export ADAPT_DYNAMIC_MIN_BLOCKS=1364 ADAPT_DYNAMIC_MAX_BLOCKS=2730
export ADAPT_DYNAMIC_ALLOWED_TARGET_BLOCKS=1364,2730
export ADAPT_DYNAMIC_RESIZE_GRANULARITY_BLOCKS=1366
export ADAPT_DYNAMIC_BACKUP_GROW_BLOCKS=1366
export ADAPT_DYNAMIC_BACKUP_TARGET_MAP='{"1364":16,"2730":16}'
export ADAPT_DYNAMIC_BASE_PLAN="${SCRIPT_DIR}/inputs/balanced_dynamic_plans_cold14_cold27/cold14.json"
export ADAPT_DYNAMIC_PLAN_CLASSES="${SCRIPT_DIR}/plan_classes_k2_k4_cold27.json"
export ADAPT_DYNAMIC_INIT_PLAN="${ADAPT_DYNAMIC_BASE_PLAN}"
export ADAPT_DYNAMIC_PLAN_CLASS_MAP='{"1364":"cold14","2730":"cold27"}'
export ADAPT_KV_POLICY_LABEL=dynamic_k2_k4_cold27_qps2
exec bash "${SCRIPT_DIR}/run_sharegpt_trace_arm.sh" adaptmoe low "${1:?run root required}"
