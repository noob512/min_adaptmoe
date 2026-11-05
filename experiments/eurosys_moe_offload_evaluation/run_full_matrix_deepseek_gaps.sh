#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_e2_c16_direct_batch_matrix.sh"
C1_COPY_PROFILE="${SCRIPT_DIR}/configs/e2_c1_near_equal_native_copy_full_graph.json"
C16_COPY_PROFILE="${SCRIPT_DIR}/configs/e2_c16_near_equal_native_copy_full_graph.json"
C16_UVA_PROFILE="${SCRIPT_DIR}/configs/e2_c16_near_equal_offload_plus_one_expert.json"
TOP_ROOT="${1:-${SCRIPT_DIR}/artifacts/full-matrix-deepseek-gaps-$(date -u +%Y%m%dT%H%M%SZ)}"

die() { echo "DeepSeek full-matrix gaps: $*" >&2; exit 1; }

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
for path in "${RUNNER}" "${C1_COPY_PROFILE}" "${C16_COPY_PROFILE}" \
  "${C16_UVA_PROFILE}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done
if [[ -e "${TOP_ROOT}" && "${FULL_MATRIX_RESUME:-0}" != 1 ]]; then
  die "refusing to overwrite ${TOP_ROOT}; set FULL_MATRIX_RESUME=1 to continue"
fi
mkdir -p "${TOP_ROOT}"
TOP_ROOT="$(cd "${TOP_ROOT}" && pwd)"
EXECUTION="${TOP_ROOT}/execution.tsv"
if [[ ! -f "${EXECUTION}" ]]; then
  printf 'section\troot\treturncode\n' >"${EXECUTION}"
fi

section_done() {
  awk -F '\t' -v section="$1" 'NR > 1 && $1 == section {found=1} END {exit !found}' \
    "${EXECUTION}"
}

record() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"${EXECUTION}"
}

run_adapt_prefill() {
  local concurrency="$1" profile="$2" name="adaptmoe_prefill_c${1}"
  local root="${TOP_ROOT}/${name}" rc
  section_done "${name}" && return 0
  set +e
  ALLOW_GPU_RUN=1 E2_C16_PROFILE="${profile}" E2_C16_SYSTEMS=ours \
    E2_C16_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
    E2_DIRECT_CONCURRENCY="${concurrency}" E2_MEASUREMENT_OUTPUT_TOKENS=1 \
    E2_SKIP_DUAL_VIEW_VERIFY=1 E2_C16_CASE_TIMEOUT_SECONDS=7200 \
    bash "${RUNNER}" "${root}"
  rc=$?
  set -e
  record "${name}" "${root}" "${rc}"
}

run_native_copy() {
  local concurrency="$1" profile="$2" graph_mode="$3" label="$4"
  local name="native_copy_${label}_c${concurrency}"
  local root="${TOP_ROOT}/${name}" rc
  section_done "${name}" && return 0
  set +e
  ALLOW_GPU_RUN=1 E2_C16_PROFILE="${profile}" \
    E2_C16_SYSTEMS=native_copy_on_forward \
    E2_C16_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
    E2_DIRECT_CONCURRENCY="${concurrency}" \
    E2_C16_CUDAGRAPH_MODE="${graph_mode}" E2_C16_CASE_TIMEOUT_SECONDS=7200 \
    bash "${RUNNER}" "${root}"
  rc=$?
  set -e
  record "${name}" "${root}" "${rc}"
}

run_native_uva_graph_c16() {
  local name=native_uva_dual_graph_c16
  local root="${TOP_ROOT}/${name}" rc
  section_done "${name}" && return 0
  set +e
  ALLOW_GPU_RUN=1 E2_C16_PROFILE="${C16_UVA_PROFILE}" \
    E2_C16_SYSTEMS=native_pinned_uva \
    E2_C16_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
    E2_DIRECT_CONCURRENCY=16 E2_C16_CUDAGRAPH_MODE=FULL_SEPARATE \
    E2_C16_CASE_TIMEOUT_SECONDS=7200 bash "${RUNNER}" "${root}"
  rc=$?
  set -e
  record "${name}" "${root}" "${rc}"
}

run_adapt_prefill 1 "${C1_COPY_PROFILE}"
run_adapt_prefill 16 "${C16_COPY_PROFILE}"
run_native_copy 1 "${C1_COPY_PROFILE}" NONE eager
run_native_copy 16 "${C16_COPY_PROFILE}" NONE eager
run_native_uva_graph_c16
run_native_copy 1 "${C1_COPY_PROFILE}" FULL_SEPARATE dual_graph
run_native_copy 16 "${C16_COPY_PROFILE}" FULL_SEPARATE dual_graph

echo "TOP_ROOT=${TOP_ROOT}"
if awk -F '\t' 'NR > 1 && $3 != 0 {bad=1} END {exit !bad}' "${EXECUTION}"; then
  exit 1
fi
