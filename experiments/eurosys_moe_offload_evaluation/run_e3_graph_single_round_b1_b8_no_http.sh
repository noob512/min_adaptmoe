#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_formal_e3.sh"
TOP_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-graph-single-round-b1-b8-no-http-$(date -u +%Y%m%dT%H%M%SZ)}"
GROUP_FILTER="${GROUP_FILTER:-tier2_main,tier2_medium_b1,dual_tier8_medium_b8,dual_tier8_rag_b1,dual_cold63_rag_b8}"
SERVING_NATIVE_HOLE_SENTINEL_BYTES="${SERVING_NATIVE_HOLE_SENTINEL_BYTES:-268435456}"

PLAN_2G="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json"
PLAN_8G="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json"
PLAN_COLD63="${SCRIPT_DIR}/inputs/frozen/e3_qwen_cold63_20260828/cold_decode_plan_2g.json"

die() { echo "E3 Graph single round: $*" >&2; exit 1; }
contains_csv() { local needle="$1" haystack=",$2,"; [[ "${haystack}" == *",${needle},"* ]]; }

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ ! -e "${TOP_ROOT}" ]] || die "TOP_ROOT already exists: ${TOP_ROOT}"
for path in "${BASE_RUNNER}" "${PLAN_2G}" "${PLAN_8G}" "${PLAN_COLD63}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done
mkdir -p "${TOP_ROOT}/groups" "${TOP_ROOT}/provenance"
TOP_ROOT="$(cd "${TOP_ROOT}" && pwd)"

run_group() {
  local name="$1" workloads="$2" batches="$3" plan="$4" selected="$5" backing="$6" moe_chunk="$7" autotune_limit="$8"
  contains_csv "${name}" "${GROUP_FILTER}" || return 0
  local group_root="${TOP_ROOT}/groups/${name}"
  printf '%q ' env ALLOW_GPU_RUN=1 PROCESS_BLOCKS=1 RUN_KT=0 \
    E3_CURRENT_TIER_DUAL_VIEW=1 \
    WORKLOAD_FILTER="${workloads}" BATCH_FILTER="${batches}" \
    CUSTOM_ARMS=G,B,C ARM_FILTER=G,B,C CUDAGRAPH_CAPTURE_SIZES=per_batch \
    RUNTIME_HOT_RESIDENCY_AUDIT=0 CUSTOM_PREFILL_NA_THRESHOLD=0 \
    FUSED_MOE_CHUNK_SIZE="${moe_chunk}" \
    FLASHINFER_AUTOTUNE_MAX_TOKENS="${autotune_limit}" \
    VLLM_UVM_POST_KV_HOT_RECONCILE=1 \
    PRE_KV_NATIVE_HOLE_SENTINEL=auto \
    SERVING_NATIVE_HOLE_SENTINEL="${SERVING_NATIVE_HOLE_SENTINEL_BYTES}" \
    CUSTOM_KV_LIMIT_BYTES=7000000000 STOP_ON_CUSTOM_ERROR=1 \
    PLAN="${plan}" SELECTED_BYTES="${selected}" BACKING_BYTES="${backing}" \
    RUN_ROOT="${group_root}" bash "${BASE_RUNNER}" \
    >"${TOP_ROOT}/provenance/${name}.command.txt"
  printf '\n' >>"${TOP_ROOT}/provenance/${name}.command.txt"
  env ALLOW_GPU_RUN=1 PROCESS_BLOCKS=1 RUN_KT=0 \
    E3_CURRENT_TIER_DUAL_VIEW=1 \
    WORKLOAD_FILTER="${workloads}" BATCH_FILTER="${batches}" \
    CUSTOM_ARMS=G,B,C ARM_FILTER=G,B,C CUDAGRAPH_CAPTURE_SIZES=per_batch \
    RUNTIME_HOT_RESIDENCY_AUDIT=0 CUSTOM_PREFILL_NA_THRESHOLD=0 \
    FUSED_MOE_CHUNK_SIZE="${moe_chunk}" \
    FLASHINFER_AUTOTUNE_MAX_TOKENS="${autotune_limit}" \
    VLLM_UVM_POST_KV_HOT_RECONCILE=1 \
    PRE_KV_NATIVE_HOLE_SENTINEL=auto \
    SERVING_NATIVE_HOLE_SENTINEL="${SERVING_NATIVE_HOLE_SENTINEL_BYTES}" \
    CUSTOM_KV_LIMIT_BYTES=7000000000 STOP_ON_CUSTOM_ERROR=1 \
    PLAN="${plan}" SELECTED_BYTES="${selected}" BACKING_BYTES="${backing}" \
    RUN_ROOT="${group_root}" bash "${BASE_RUNNER}"
}

run_dual_graph_group() {
  local name="$1" workloads="$2" batches="$3" plan="$4" selected="$5" backing="$6" reserve_tokens="$7"
  contains_csv "${name}" "${GROUP_FILTER}" || return 0
  local group_root="${TOP_ROOT}/groups/${name}"
  printf '%q ' env ALLOW_GPU_RUN=1 PROCESS_BLOCKS=1 RUN_KT=0 \
    E3_CURRENT_TIER_DUAL_VIEW=1 \
    WORKLOAD_FILTER="${workloads}" BATCH_FILTER="${batches}" \
    CUSTOM_ARMS=G,B,D ARM_FILTER=D CUDAGRAPH_CAPTURE_SIZES=per_batch \
    RUNTIME_HOT_RESIDENCY_AUDIT=0 CUSTOM_PREFILL_NA_THRESHOLD=0 \
    FUSED_MOE_CHUNK_SIZE=4096 FLASHINFER_AUTOTUNE_MAX_TOKENS=4096 \
    VLLM_UVM_POST_KV_HOT_RECONCILE=1 \
    PRE_KV_NATIVE_HOLE_SENTINEL=auto CUSTOM_KV_LIMIT_BYTES=7000000000 \
    SERVING_NATIVE_HOLE_SENTINEL="${SERVING_NATIVE_HOLE_SENTINEL_BYTES}" \
    STOP_ON_CUSTOM_ERROR=1 PLAN="${plan}" SELECTED_BYTES="${selected}" \
    BACKING_BYTES="${backing}" \
    VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
    VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1 \
    VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1 \
    VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592 \
    VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS="${reserve_tokens}" \
    RUN_ROOT="${group_root}" bash "${BASE_RUNNER}" \
    >"${TOP_ROOT}/provenance/${name}.command.txt"
  printf '\n' >>"${TOP_ROOT}/provenance/${name}.command.txt"
  env ALLOW_GPU_RUN=1 PROCESS_BLOCKS=1 RUN_KT=0 \
    E3_CURRENT_TIER_DUAL_VIEW=1 \
    WORKLOAD_FILTER="${workloads}" BATCH_FILTER="${batches}" \
    CUSTOM_ARMS=G,B,D ARM_FILTER=D CUDAGRAPH_CAPTURE_SIZES=per_batch \
    RUNTIME_HOT_RESIDENCY_AUDIT=0 CUSTOM_PREFILL_NA_THRESHOLD=0 \
    FUSED_MOE_CHUNK_SIZE=4096 FLASHINFER_AUTOTUNE_MAX_TOKENS=4096 \
    VLLM_UVM_POST_KV_HOT_RECONCILE=1 \
    PRE_KV_NATIVE_HOLE_SENTINEL=auto CUSTOM_KV_LIMIT_BYTES=7000000000 \
    SERVING_NATIVE_HOLE_SENTINEL="${SERVING_NATIVE_HOLE_SENTINEL_BYTES}" \
    STOP_ON_CUSTOM_ERROR=1 PLAN="${plan}" SELECTED_BYTES="${selected}" \
    BACKING_BYTES="${backing}" \
    VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
    VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1 \
    VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1 \
    VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592 \
    VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS="${reserve_tokens}" \
    RUN_ROOT="${group_root}" bash "${BASE_RUNNER}"
}

# Small/ordinary/long and Medium-B1 retain the frozen cold10 (about 2 GiB)
# geometry.  Larger one-step Prefill cells get additional HBM headroom while
# all three arms within a cell still use the exact same expert coordinates.
run_group tier2_main "short_dialogue,ordinary_chat,long_generation" "1,8" \
  "${PLAN_2G}" 2264924160 94371840 16384 0
run_group tier2_medium_b1 "medium_context" "1" \
  "${PLAN_2G}" 2264924160 94371840 16384 0
run_dual_graph_group dual_tier8_medium_b8 "medium_context" "8" \
  "${PLAN_8G}" 8606711808 358612992 16384
run_dual_graph_group dual_tier8_rag_b1 "rag_summary" "1" \
  "${PLAN_8G}" 8606711808 358612992 8192
run_dual_graph_group dual_cold63_rag_b8 "rag_summary" "8" \
  "${PLAN_COLD63}" 14269022208 594542592 20480

jq -n \
  --arg schema eurosys_e3_graph_single_round_b1_b8_no_http_v1 \
  --arg group_filter "${GROUP_FILTER}" \
  --arg serving_hole "${SERVING_NATIVE_HOLE_SENTINEL_BYTES}" \
  '{schema:$schema,process_blocks:1,batches:[1,8],
    comparison_arms:["G","B","C"],large_prefill_qualification_arm:"D",
    transport:"none",all_requests_submitted_before_first_step:true,
    mixed_prefill_decode:false,
    graph_modes:["FULL_DECODE_ONLY","FULL_SEPARATE"],
    cudagraph_capture_sizes:"current_batch_only",hot_residency_audit:false,
    post_kv_hot_reconcile:true,
    current_tier_dual_view:true,
    serving_native_hole_sentinel_bytes:($serving_hole|tonumber),
    group_filter:$group_filter}' \
  >"${TOP_ROOT}/configuration.json"

echo "TOP_ROOT=${TOP_ROOT}"
