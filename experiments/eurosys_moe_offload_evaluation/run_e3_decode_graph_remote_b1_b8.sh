#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_formal_e3.sh"
TOP_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-decode-graph-remote-b1-b8-$(date -u +%Y%m%dT%H%M%SZ)}"
PLAN_10="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json"
PLAN_38="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json"
PLAN_63="${SCRIPT_DIR}/inputs/frozen/e3_qwen_cold63_20260828/cold_decode_plan_2g.json"

die() { echo "E3 Decode Graph remote: $*" >&2; exit 1; }

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ ! -e "${TOP_ROOT}" ]] || die "output already exists: ${TOP_ROOT}"
for path in "${BASE_RUNNER}" "${PLAN_10}" "${PLAN_38}" "${PLAN_63}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done

gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/')"
[[ "${gpu_used}" == 0 && -z "${compute_pids}" ]] || die "GPU is not initially clean"

mkdir -p "${TOP_ROOT}/groups" "${TOP_ROOT}/na" "${TOP_ROOT}/provenance"
TOP_ROOT="$(cd "${TOP_ROOT}" && pwd)"

run_group() {
  local name="$1" workloads="$2" batches="$3" plan="$4"
  local selected="$5" backing="$6" chunk="$7" autotune="$8"
  local group_root="${TOP_ROOT}/groups/${name}"
  local -a command=(
    env ALLOW_GPU_RUN=1 PROCESS_BLOCKS=1 RUN_CUSTOM=1 RUN_KT=0
    ATTENTION_BACKEND=FLASH_ATTN
    WORKLOAD_FILTER="${workloads}" BATCH_FILTER="${batches}"
    CUSTOM_ARMS=G,B,C ARM_FILTER=C
    CUDAGRAPH_CAPTURE_SIZES=per_batch
    CUSTOM_PREFILL_NA_THRESHOLD=0 CUSTOM_KV_LIMIT_BYTES=7000000000
    STOP_ON_CUSTOM_ERROR=1 RUNTIME_HOT_RESIDENCY_AUDIT=0
    REMOTE_READ_FULL_TLB_INVALIDATE=1
    FUSED_MOE_CHUNK_SIZE="${chunk}"
    FLASHINFER_AUTOTUNE_MAX_TOKENS="${autotune}"
    PRE_KV_NATIVE_HOLE_SENTINEL=auto
    PLAN="${plan}" SELECTED_BYTES="${selected}" BACKING_BYTES="${backing}"
    RUN_ROOT="${group_root}" bash "${BASE_RUNNER}"
  )
  printf '%q ' "${command[@]}" >"${TOP_ROOT}/provenance/${name}.command.txt"
  printf '\n' >>"${TOP_ROOT}/provenance/${name}.command.txt"
  "${command[@]}"
}

record_na() {
  local workload="$1" batch="$2" prompt_tokens="$3" kv_bytes="$4"
  local attempts="$5" reason="$6"
  jq -n --arg workload "${workload}" --argjson batch "${batch}" \
    --argjson prompt_tokens "${prompt_tokens}" --argjson kv_bytes "${kv_bytes}" \
    --arg attempts "${attempts}" --arg reason "${reason}" \
    '{status:"N/A",phase:"conditioning_prefill",workload:$workload,
      batch_size:$batch,prompt_tokens:$prompt_tokens,kv_cache_bytes:$kv_bytes,
      decode_sample_obtained:false,attempts:$attempts,reason:$reason}' \
    >"${TOP_ROOT}/na/${workload}_b${batch}.json"
}

jq -n '{
  schema:"eurosys_e3_decode_graph_remote_b1_b8_v1",
  model:"Qwen3-30B-A3B-FP8",
  process_blocks:1,
  workloads:["short_dialogue","ordinary_chat","medium_context","rag_summary","long_generation"],
  batches:[1,8],
  custom_arm:"C",
  prefill:"remote eager, conditioning only",
  decode:"remote-read FULL_DECODE_ONLY CUDA Graph",
  attention_backend:"FLASH_ATTN",
  transport:"none",
  mixed_prefill_decode:false,
  runtime_hot_residency_audit:false,
  pre_registered_na:["medium_context_b8","rag_summary_b8"]
}' >"${TOP_ROOT}/configuration.json"

run_group cold10_main "short_dialogue,ordinary_chat" "1,8" \
  "${PLAN_10}" 2264924160 94371840 16384 0
run_group cold10_long_b1 "long_generation" "1" \
  "${PLAN_10}" 2264924160 94371840 16384 0
run_group cold38_long_b8 "long_generation" "8" \
  "${PLAN_38}" 8606711808 358612992 1024 2048
run_group cold10_medium_b1 "medium_context" "1" \
  "${PLAN_10}" 2264924160 94371840 16384 0
run_group cold63_rag_b1 "rag_summary" "1" \
  "${PLAN_63}" 14269022208 594542592 1024 2048

record_na medium_context 8 16384 1723858944 \
  "cold38; cold63; cold63 with MoE chunk 1024 and autotune cap 2048" \
  "The single-cohort 16384-token conditioning Prefill faults in the custom original-VA path before Decode; cold63 leaves ample free HBM, so increasing offload is not a valid repair."
record_na rag_summary 8 65536 6555697152 \
  "cold63 with MoE chunk 1024 and autotune cap 2048" \
  "The single-cohort 65536-token conditioning Prefill faults before Decode; splitting B8 or enabling asymmetric chunk admission would change the experiment contract."

gpu_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/')"
[[ "${gpu_used}" == 0 && -z "${compute_pids}" ]] || die "GPU is not clean after the matrix"

echo "TOP_ROOT=${TOP_ROOT}"
