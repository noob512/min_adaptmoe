#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FORMAL_E3="${SCRIPT_DIR}/run_formal_e3.sh"
NATIVE_E3="${SCRIPT_DIR}/run_e3_native_copy_vs_uva_phase.sh"
PLAN_LOW="${E3_PLAN_LOW:-${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json}"
PLAN_MID="${E3_PLAN_MID:-${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json}"
PLAN_HIGH="${E3_PLAN_HIGH:-${REPO_ROOT}/experiments/vllm_uvm_e10_multilevel_pressure/inputs/cold89.json}"
COLD_WIDTH_LOW="${E3_COLD_WIDTH_LOW:-10}"
COLD_WIDTH_MID="${E3_COLD_WIDTH_MID:-38}"
COLD_WIDTH_HIGH="${E3_COLD_WIDTH_HIGH:-89}"
TOP_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-c1-c16-uva-single-round-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NATIVE_BASELINES="${E3_RUN_NATIVE_BASELINES:-1}"
RUN_KTRANSFORMERS="${E3_RUN_KTRANSFORMERS:-1}"
OURS_CELL_FILTER="${E3_OURS_CELL_FILTER:-all}"

die() { echo "E3 C1/C16 UVA: $*" >&2; exit 1; }
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${RUN_NATIVE_BASELINES}" == 0 || "${RUN_NATIVE_BASELINES}" == 1 ]] \
  || die "E3_RUN_NATIVE_BASELINES must be 0 or 1"
[[ "${RUN_KTRANSFORMERS}" == 0 || "${RUN_KTRANSFORMERS}" == 1 ]] \
  || die "E3_RUN_KTRANSFORMERS must be 0 or 1"
[[ ! -e "${TOP_ROOT}" ]] || die "output already exists: ${TOP_ROOT}"
for width in "${COLD_WIDTH_LOW}" "${COLD_WIDTH_MID}" "${COLD_WIDTH_HIGH}"; do
  [[ "${width}" =~ ^[1-9][0-9]*$ ]] || die "invalid cold width: ${width}"
  (( width <= 128 )) || die "cold width exceeds 128: ${width}"
done
(( COLD_WIDTH_LOW < COLD_WIDTH_MID && COLD_WIDTH_MID < COLD_WIDTH_HIGH )) \
  || die "cold widths must be strictly increasing"
for path in "${FORMAL_E3}" "${NATIVE_E3}" "${PLAN_LOW}" "${PLAN_MID}" "${PLAN_HIGH}"; do
  [[ -f "${path}" ]] || die "missing file: ${path}"
done
mkdir -p "${TOP_ROOT}/ours" "${TOP_ROOT}/provenance"
TOP_ROOT="$(cd "${TOP_ROOT}" && pwd)"
printf 'section\tarm\tworkload\tbatch\troot\tlauncher_rc\n' >"${TOP_ROOT}/execution.tsv"

record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" \
    >>"${TOP_ROOT}/execution.tsv"
}

failed=0
native_root="${TOP_ROOT}/native_copy_uva"
if [[ "${RUN_NATIVE_BASELINES}" == 1 ]]; then
  set +e
  ALLOW_GPU_RUN=1 ATTENTION_BACKEND=FLASH_ATTN BATCH_FILTER=1,16 \
    CONDITIONING_REPEATS=1 MEASUREMENT_REPEATS=1 \
    bash "${NATIVE_E3}" "${native_root}"
  native_rc=$?
  set -e
  record native copy_uva all 1,16 "${native_root}" "${native_rc}"
  (( native_rc == 0 )) || failed=1
else
  record native copy_uva all 1,16 "${native_root}" skipped_reused
fi

run_ours_d() {
  local workload="$1" batch="$2" plan="$3" selected="$4" backing="$5"
  local reserve="$6" chunk="$7" autotune="$8" name root rc
  name="${workload}_b${batch}_D"
  if [[ "${OURS_CELL_FILTER}" != all && ",${OURS_CELL_FILTER}," != *",${workload}_b${batch},"* ]]; then
    return 0
  fi
  root="${TOP_ROOT}/ours/${name}"
  set +e
  env ALLOW_GPU_RUN=1 PROCESS_BLOCKS=1 RUN_CUSTOM=1 RUN_KT=0 \
    E3_CURRENT_TIER_DUAL_VIEW=1 WORKLOAD_FILTER="${workload}" BATCH_FILTER="${batch}" \
    CUSTOM_ARMS=D ARM_FILTER=D CUDAGRAPH_CAPTURE_SIZES=per_batch \
    ATTENTION_BACKEND=TRITON_ATTN CUSTOM_PREFILL_NA_THRESHOLD=0 \
    CUSTOM_KV_LIMIT_BYTES=16000000000 STOP_ON_CUSTOM_ERROR=1 \
    RUNTIME_HOT_RESIDENCY_AUDIT=0 PRE_KV_NATIVE_HOLE_SENTINEL=auto \
    FUSED_MOE_CHUNK_SIZE="${chunk}" FLASHINFER_AUTOTUNE_MAX_TOKENS="${autotune}" \
    PLAN="${plan}" SELECTED_BYTES="${selected}" BACKING_BYTES="${backing}" \
    VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
    VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1 \
    VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1 \
    VLLM_CUDAGRAPH_DISPATCH_AUDIT="${VLLM_CUDAGRAPH_DISPATCH_AUDIT:-1}" \
    VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592 \
    VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS="${reserve}" \
    RUN_ROOT="${root}" bash "${FORMAL_E3}"
  rc=$?
  set -e
  # run_formal_e3 historically returned success after a failed custom child
  # when its own post-processing completed. Treat the child receipt as the
  # authoritative result so a CUDA, residency, or protocol failure cannot be
  # hidden by a successful launcher exit.
  if (( rc == 0 )); then
    if [[ ! -f "${root}/custom/execution.tsv" ]] || ! awk -F '\t' '
      NR > 1 && $9 == "ok" && $10 == "0" { ok = 1 }
      END { exit(ok ? 0 : 1) }
    ' "${root}/custom/execution.tsv"; then
      rc=1
    fi
  fi
  record ours D "${workload}" "${batch}" "${root}" "${rc}"
  (( rc == 0 )) || failed=1
}

# One frozen cold tier per shape. Every D arm uses capture-before-hot-restore
# plus an explicit retained allocator warmup; large shapes cap that eager
# warmup at the already qualified 20,480-token envelope while the full Graph
# capture itself owns the complete shape-specific pool.
selected_low=$((COLD_WIDTH_LOW * 48 * 4718592))
selected_mid=$((COLD_WIDTH_MID * 48 * 4718592))
selected_high=$((COLD_WIDTH_HIGH * 48 * 4718592))
backing_low=$((COLD_WIDTH_LOW * 48 * 196608))
backing_mid=$((COLD_WIDTH_MID * 48 * 196608))
backing_high=$((COLD_WIDTH_HIGH * 48 * 196608))

run_ours_d short_dialogue 1  "${PLAN_LOW}" "${selected_low}" "${backing_low}" 128   16384 0
run_ours_d short_dialogue 16 "${PLAN_LOW}" "${selected_low}" "${backing_low}" 2048  16384 0
run_ours_d ordinary_chat 1   "${PLAN_LOW}" "${selected_low}" "${backing_low}" 512   16384 0
run_ours_d ordinary_chat 16  "${PLAN_MID}" "${selected_mid}" "${backing_mid}" 8192 4096 4096
run_ours_d medium_context 1  "${PLAN_LOW}" "${selected_low}" "${backing_low}" 2048  16384 0
run_ours_d medium_context 16 "${PLAN_MID}" "${selected_mid}" "${backing_mid}" 20480 4096 4096
run_ours_d rag_summary 1     "${PLAN_MID}" "${selected_mid}" "${backing_mid}" 8192 4096 4096
# RAG/C16 retains a 12.21 GiB KV pool and a 131,072-token Prefill Graph
# workspace. cold63 leaves 14.72 GiB of hot experts, which cannot be restored
# after capture (only about 10.17 GiB is free). cold89 keeps the request shape
# unchanged while reducing the hot set to 8.23 GiB and preserving >1 GiB of
# post-reconcile headroom.
run_ours_d rag_summary 16    "${PLAN_HIGH}" "${selected_high}" "${backing_high}" 20480 4096 4096
run_ours_d long_generation 1 "${PLAN_LOW}" "${selected_low}" "${backing_low}" 512   16384 0
run_ours_d long_generation 16 "${PLAN_MID}" "${selected_mid}" "${backing_mid}" 8192 4096 4096

kt_root="${TOP_ROOT}/ktransformers"
if [[ "${RUN_KTRANSFORMERS}" == 1 ]]; then
  set +e
  env ALLOW_GPU_RUN=1 PROCESS_BLOCKS=1 RUN_CUSTOM=0 RUN_KT=1 \
    BATCH_FILTER=1,16 WORKLOAD_FILTER=short_dialogue,ordinary_chat,medium_context,rag_summary,long_generation \
    RUN_ROOT="${kt_root}" bash "${FORMAL_E3}"
  kt_rc=$?
  set -e
  record native ktransformers all 1,16 "${kt_root}" "${kt_rc}"
  (( kt_rc == 0 )) || failed=1
else
  record native ktransformers all 1,16 "${kt_root}" skipped_reused
fi

jq -n --argjson launcher_failed "${failed}" \
  --argjson native_baselines_run "${RUN_NATIVE_BASELINES}" \
  --argjson ktransformers_run "${RUN_KTRANSFORMERS}" \
  --arg ours_cell_filter "${OURS_CELL_FILTER}" \
  --argjson cold_width_low "${COLD_WIDTH_LOW}" \
  --argjson cold_width_mid "${COLD_WIDTH_MID}" \
  --argjson cold_width_high "${COLD_WIDTH_HIGH}" '{
  schema:"eurosys_e3_c1_c16_uva_single_round_v1",
  batches:[1,16],
  workloads:["short_dialogue","ordinary_chat","medium_context","rag_summary","long_generation"],
  arms:{D:"ours Prefill/Decode Graph with retained workspace",copy:"native vLLM copy-on-forward",uva:"native vLLM pinned UVA remote-read",ktransformers:"exact-coordinate KTransformers"},
  custom_uvm_always_remote_excluded:true,
  custom_uvm_always_remote_replacement:"native vLLM pinned UVA",
  repeats:1,
  native_baselines_run:($native_baselines_run == 1),
  ktransformers_run:($ktransformers_run == 1),
  ours_cell_filter:$ours_cell_filter,
  ours_cold_widths:{low:$cold_width_low,mid:$cold_width_mid,high:$cold_width_high},
  rag_c16_capacity_tier:("cold" + ($cold_width_high|tostring)),
  skipped_arms_reuse_previous_single_round:true,
  launcher_failed:($launcher_failed == 1)
}' >"${TOP_ROOT}/configuration.json"

echo "TOP_ROOT=${TOP_ROOT}"
exit "${failed}"
