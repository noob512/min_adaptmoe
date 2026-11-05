#!/usr/bin/env bash
set -euo pipefail

DEEP_LARGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEP_REPO_ROOT="$(cd "${DEEP_LARGE_DIR}/../.." && pwd)"
DEEP_VLLM_PY="${DEEP_REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
DEEP_ASSET_ROOT="${DEEP_REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v1-20260823/deepseek_v2_lite_bf16"
DEEP_CALIBRATION="${DEEP_ASSET_ROOT}/calibration_assets_attempt3/routing/routing_trace.jsonl"
DEEP_PLAN="${DEEP_LARGE_DIR}/inputs/deepseek_cold_plan_uniform32.json"

"${DEEP_VLLM_PY}" "${DEEP_LARGE_DIR}/prepare_deepseek_large_plan.py" \
  --trace "${DEEP_CALIBRATION}" \
  --output "${DEEP_PLAN}" \
  --experts-per-layer 32 >/dev/null

export VERIFY="${DEEP_LARGE_DIR}/verify_deepseek_experiment.py"
export MODEL="/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
export FIXTURE="${DEEP_ASSET_ROOT}/evaluation_ordinary_chat.json"
export NAMES="${DEEP_ASSET_ROOT}/calibration_assets_attempt3/plans/exact_parameter_names.txt"
export EXPERT_PLAN="${DEEP_PLAN}"
export ATTENTION_BACKEND=TRITON_MLA
export CUDAGRAPH_CAPTURE_SIZES=1,512
export MAX_MODEL_LEN=640
export MAX_NUM_BATCHED_TOKENS=640
export EXPECTED_PARAMETER_COUNT=52
export EXPECTED_LAYER_COUNT=26
export EXPECTED_COLD_EXPERTS_PER_LAYER=32
export EXPECTED_FULL_PARAMETER_BYTES=28789702656
export EXPECTED_SELECTED_BYTES=14394851328
export EXPECTED_TOTAL_MODEL_BYTES=31412968448
export KV_BLOCK_BYTES=497664
export KV_MAX_BYTES=10737598464
export KV_2G_BLOCKS=4314
export KV_4G_BLOCKS=8629
export KV_8G_BLOCKS=17260
export KV_10G_BLOCKS=21575
export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-2147483648}"
export MIN_RAW_HEADROOM_BYTES="${MIN_RAW_HEADROOM_BYTES:-4294967296}"
export CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-30.0}"
export VLLM_LARGE_KV_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${DEEP_LARGE_DIR}/run_experiment.sh"
unset VLLM_LARGE_KV_SOURCE_ONLY

case "${1:-help}" in
  static-preflight)
    static_preflight_large_kv
    ;;
  smoke)
    CASE_LABELS=direct_2g VERIFY_AFTER_RUN=0 run_qualification_large_kv
    ;;
  qualify|qualification|run)
    run_qualification_large_kv
    ;;
  verify)
    [[ $# -eq 2 ]] || die_large_kv "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2" --output "$2/verification.json"
    ;;
  *)
    echo "usage: $0 {static-preflight|smoke|qualify|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
