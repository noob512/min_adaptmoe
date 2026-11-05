#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e2-native-uva-full-graph-matched-protocol-$(date -u +%Y%m%dT%H%M%SZ)}"
C1_PROFILE="${SCRIPT_DIR}/configs/e2_c1_near_equal_offload_plus_one_expert.json"
C16_PROFILE="${SCRIPT_DIR}/configs/e2_c16_near_equal_offload_plus_one_expert.json"

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || {
  echo "GPU execution requires ALLOW_GPU_RUN=1" >&2
  exit 1
}
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "refusing to overwrite ${RUN_ROOT}" >&2
  exit 1
}

mkdir -p "${RUN_ROOT}/provenance"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
cp "${C1_PROFILE}" "${RUN_ROOT}/provenance/c1.profile.json"
cp "${C16_PROFILE}" "${RUN_ROOT}/provenance/c16.profile.json"
sha256sum "${BASH_SOURCE[0]}" "${C1_PROFILE}" "${C16_PROFILE}" \
  "${SCRIPT_DIR}/run_c1_mechanism_runtime_hot_audit_off_matrix.sh" \
  "${SCRIPT_DIR}/run_e2_c16_direct_batch_matrix.sh" \
  >"${RUN_ROOT}/provenance/runner.sha256"
printf 'section\treturncode\n' >"${RUN_ROOT}/execution.tsv"

set +e
MATRIX_PROFILE_CONFIG="${C1_PROFILE}" \
MATRIX_SYSTEMS=native_pinned_uva \
MATRIX_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
MATRIX_SINGLE_ATTEMPT=1 \
MATRIX_NATIVE_UVA_CUDAGRAPH_MODE=FULL_SEPARATE \
bash "${SCRIPT_DIR}/run_c1_mechanism_runtime_hot_audit_off_matrix.sh" \
  "${RUN_ROOT}/c1" >"${RUN_ROOT}/c1.launcher.log" 2>&1
c1_rc=$?
printf 'c1\t%s\n' "${c1_rc}" >>"${RUN_ROOT}/execution.tsv"

E2_C16_PROFILE="${C16_PROFILE}" \
E2_C16_SYSTEMS=native_pinned_uva \
E2_C16_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
E2_C16_CUDAGRAPH_MODE=FULL_SEPARATE \
bash "${SCRIPT_DIR}/run_e2_c16_direct_batch_matrix.sh" \
  "${RUN_ROOT}/c16" >"${RUN_ROOT}/c16.launcher.log" 2>&1
c16_rc=$?
printf 'c16\t%s\n' "${c16_rc}" >>"${RUN_ROOT}/execution.tsv"
set -e

nvidia-smi --query-gpu=index,memory.used,memory.free \
  --format=csv,noheader,nounits >"${RUN_ROOT}/gpu.after.csv"

echo "RUN_ROOT=${RUN_ROOT}"
echo "C1_RC=${c1_rc} C16_RC=${c16_rc}"
(( c1_rc == 0 && c16_rc == 0 ))
