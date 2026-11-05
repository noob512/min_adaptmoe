#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_e2_c16_direct_batch_matrix.sh"
C1_PROFILE="${SCRIPT_DIR}/configs/e2_c1_near_equal_native_copy_full_graph.json"
C16_PROFILE="${SCRIPT_DIR}/configs/e2_c16_near_equal_native_copy_full_graph.json"
TOP_ROOT="${1:-${SCRIPT_DIR}/artifacts/full-matrix-deepseek-retries-$(date -u +%Y%m%dT%H%M%SZ)}"

die() { echo "DeepSeek full-matrix retries: $*" >&2; exit 1; }
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
for path in "${RUNNER}" "${C1_PROFILE}" "${C16_PROFILE}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done
[[ ! -e "${TOP_ROOT}" ]] || die "refusing to overwrite ${TOP_ROOT}"
mkdir -p "${TOP_ROOT}"
TOP_ROOT="$(cd "${TOP_ROOT}" && pwd)"
EXECUTION="${TOP_ROOT}/execution.tsv"
printf 'section\troot\treturncode\n' >"${EXECUTION}"

run_section() {
  local section="$1"
  shift
  local root="${TOP_ROOT}/${section}" rc
  set +e
  "$@" "${root}"
  rc=$?
  set -e
  printf '%s\t%s\t%s\n' "${section}" "${root}" "${rc}" >>"${EXECUTION}"
}

run_section adaptmoe_prefill_rag_c1 \
  env ALLOW_GPU_RUN=1 E2_C16_PROFILE="${C1_PROFILE}" E2_C16_SYSTEMS=ours \
  E2_C16_WORKLOADS=rag_summary E2_DIRECT_CONCURRENCY=1 \
  E2_MEASUREMENT_OUTPUT_TOKENS=1 E2_SKIP_DUAL_VIEW_VERIFY=1 \
  E2_C16_CASE_TIMEOUT_SECONDS=7200 bash "${RUNNER}"

run_section adaptmoe_prefill_ordinary_c16_output2 \
  env ALLOW_GPU_RUN=1 E2_C16_PROFILE="${C16_PROFILE}" E2_C16_SYSTEMS=ours \
  E2_C16_WORKLOADS=ordinary_chat E2_DIRECT_CONCURRENCY=16 \
  E2_MEASUREMENT_OUTPUT_TOKENS=2 E2_C16_CASE_TIMEOUT_SECONDS=7200 \
  bash "${RUNNER}"

run_section native_copy_eager_c1 \
  env ALLOW_GPU_RUN=1 E2_C16_PROFILE="${C1_PROFILE}" \
  E2_C16_SYSTEMS=native_copy_on_forward \
  E2_C16_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
  E2_DIRECT_CONCURRENCY=1 E2_C16_CUDAGRAPH_MODE=NONE \
  E2_C16_CASE_TIMEOUT_SECONDS=7200 bash "${RUNNER}"

run_section native_copy_eager_c16_short \
  env ALLOW_GPU_RUN=1 E2_C16_PROFILE="${C16_PROFILE}" \
  E2_C16_SYSTEMS=native_copy_on_forward E2_C16_WORKLOADS=short_dialogue \
  E2_DIRECT_CONCURRENCY=16 E2_C16_CUDAGRAPH_MODE=NONE \
  E2_C16_CASE_TIMEOUT_SECONDS=7200 bash "${RUNNER}"

run_section native_copy_dual_graph_c1 \
  env ALLOW_GPU_RUN=1 E2_C16_PROFILE="${C1_PROFILE}" \
  E2_C16_SYSTEMS=native_copy_on_forward \
  E2_C16_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
  E2_DIRECT_CONCURRENCY=1 E2_C16_CUDAGRAPH_MODE=FULL_SEPARATE \
  E2_C16_CASE_TIMEOUT_SECONDS=7200 bash "${RUNNER}"

run_section native_copy_dual_graph_c16_short \
  env ALLOW_GPU_RUN=1 E2_C16_PROFILE="${C16_PROFILE}" \
  E2_C16_SYSTEMS=native_copy_on_forward E2_C16_WORKLOADS=short_dialogue \
  E2_DIRECT_CONCURRENCY=16 E2_C16_CUDAGRAPH_MODE=FULL_SEPARATE \
  E2_C16_CASE_TIMEOUT_SECONDS=7200 bash "${RUNNER}"

echo "TOP_ROOT=${TOP_ROOT}"
if awk -F '\t' 'NR > 1 && $3 != 0 {bad=1} END {exit !bad}' "${EXECUTION}"; then
  exit 1
fi
