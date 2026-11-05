#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_formal_e3.sh"
TOP_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-d-vs-c-paired-$(date -u +%Y%m%dT%H%M%SZ)}"
SERVING_NATIVE_HOLE_SENTINEL_BYTES="${SERVING_NATIVE_HOLE_SENTINEL_BYTES:-268435456}"

PLAN_8G="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json"
PLAN_COLD63="${SCRIPT_DIR}/inputs/frozen/e3_qwen_cold63_20260828/cold_decode_plan_2g.json"

die() { echo "E3 paired D/C: $*" >&2; exit 1; }

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ ! -e "${TOP_ROOT}" ]] || die "TOP_ROOT already exists: ${TOP_ROOT}"
for path in "${BASE_RUNNER}" "${PLAN_8G}" "${PLAN_COLD63}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done
mkdir -p "${TOP_ROOT}/groups" "${TOP_ROOT}/provenance"
TOP_ROOT="$(cd "${TOP_ROOT}" && pwd)"

run_pair() {
  local name="$1" workload="$2" batch="$3" plan="$4"
  local selected="$5" backing="$6" reserve_tokens="$7" arm_order="$8"
  local group_root="${TOP_ROOT}/groups/${name}"
  local common_env=(
    ALLOW_GPU_RUN=1 PROCESS_BLOCKS=1 RUN_KT=0
    E3_CURRENT_TIER_DUAL_VIEW=1
    WORKLOAD_FILTER="${workload}" BATCH_FILTER="${batch}"
    CUSTOM_ARMS="${arm_order}" ARM_FILTER=D,C
    ATTENTION_BACKEND=TRITON_ATTN CUDAGRAPH_CAPTURE_SIZES=per_batch
    RUNTIME_HOT_RESIDENCY_AUDIT=0 CUSTOM_PREFILL_NA_THRESHOLD=0
    FUSED_MOE_CHUNK_SIZE=4096 FLASHINFER_AUTOTUNE_MAX_TOKENS=4096
    VLLM_UVM_POST_KV_HOT_RECONCILE=1
    PRE_KV_NATIVE_HOLE_SENTINEL=auto
    SERVING_NATIVE_HOLE_SENTINEL="${SERVING_NATIVE_HOLE_SENTINEL_BYTES}"
    CUSTOM_KV_LIMIT_BYTES=7000000000 STOP_ON_CUSTOM_ERROR=1
    PLAN="${plan}" SELECTED_BYTES="${selected}" BACKING_BYTES="${backing}"
    VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1
    VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1
    VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592
    VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS="${reserve_tokens}"
    RUN_ROOT="${group_root}"
  )
  printf '%q ' env "${common_env[@]}" bash "${BASE_RUNNER}" \
    >"${TOP_ROOT}/provenance/${name}.command.txt"
  printf '\n' >>"${TOP_ROOT}/provenance/${name}.command.txt"
  env "${common_env[@]}" bash "${BASE_RUNNER}"
}

jq -n \
  --arg serving_hole "${SERVING_NATIVE_HOLE_SENTINEL_BYTES}" \
  '{schema:"eurosys_e3_d_vs_c_paired_v1",qualification_only:true,
    retries:0,process_blocks:1,
    cells:["medium_context/B8","rag_summary/B1","rag_summary/B8"],
    arms:{D:"GPU-cache Prefill Graph -> remote Decode Graph",
          C:"eager remote Prefill -> remote Decode Graph"},
    request_protocol:"frozen original E3",
    attention_backend:"TRITON_ATTN",
    D_cudagraph_mode:"FULL_SEPARATE",
    C_cudagraph_mode:"FULL_DECODE_ONLY",
    current_tier_dual_view_for_D:true,
    serving_native_hole_sentinel_bytes:($serving_hole|tonumber)}' \
  >"${TOP_ROOT}/configuration.json"

# Frozen original-E3 request shapes and capacity plans. Each arm is a fresh
# vLLM process. Alternate order to avoid assigning the same order to one arm.
run_pair medium_b8 medium_context 8 "${PLAN_8G}" \
  8606711808 358612992 16384 D,C
run_pair rag_b1 rag_summary 1 "${PLAN_8G}" \
  8606711808 358612992 8192 C,D
run_pair rag_b8 rag_summary 8 "${PLAN_COLD63}" \
  14269022208 594542592 20480 D,C

echo "TOP_ROOT=${TOP_ROOT}"
