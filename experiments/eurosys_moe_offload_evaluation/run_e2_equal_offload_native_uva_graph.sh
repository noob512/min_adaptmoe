#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
C1_PROFILE="${E2_EQUAL_C1_PROFILE:-${SCRIPT_DIR}/configs/e2_c1_equal_offload_native_uva_graph.json}"
C16_PROFILE="${E2_EQUAL_C16_PROFILE:-${SCRIPT_DIR}/configs/e2_c16_equal_offload_native_uva_graph.json}"
VERIFY="${SCRIPT_DIR}/verify_e2_equal_offload_native_uva_graph.py"
VERIFY_MODE="${E2_EQUAL_VERIFY_MODE:-native_requested_matches_ours}"
NATIVE_GRAPH_MODE="${E2_EQUAL_NATIVE_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e2-equal-offload-native-uva-graph-$(date -u +%Y%m%dT%H%M%SZ)}"

die() { echo "E2 equal-offload: $*" >&2; exit 1; }

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${NATIVE_GRAPH_MODE}" == FULL_DECODE_ONLY \
   || "${NATIVE_GRAPH_MODE}" == FULL_SEPARATE ]] \
  || die "unsupported E2_EQUAL_NATIVE_CUDAGRAPH_MODE=${NATIVE_GRAPH_MODE}"
[[ ! -e "${RUN_ROOT}" ]] || die "refusing to overwrite ${RUN_ROOT}"
for path in "${VLLM_PY}" "${C1_PROFILE}" "${C16_PROFILE}" "${VERIFY}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
  | head -1 | tr -d '[:space:]')"
[[ "${used}" == 0 ]] || die "GPU is not clean before the run: ${used} MiB"

mkdir -p "${RUN_ROOT}/provenance"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
kernel_epoch="$(date +%s)"
git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/repo.commit"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
cp "${C1_PROFILE}" "${RUN_ROOT}/provenance/c1.profile.json"
cp "${C16_PROFILE}" "${RUN_ROOT}/provenance/c16.profile.json"
sha256sum \
  "${BASH_SOURCE[0]}" \
  "${SCRIPT_DIR}/run_c1_mechanism_runtime_hot_audit_off_matrix.sh" \
  "${SCRIPT_DIR}/run_e2_c16_direct_batch_matrix.sh" \
  "${SCRIPT_DIR}/direct_batch_c16_vllm_runner.py" \
  "${SCRIPT_DIR}/verify_direct_batch_c16_case.py" \
  "${C1_PROFILE}" "${C16_PROFILE}" "${VERIFY}" \
  >"${RUN_ROOT}/provenance/runner.sha256"
printf 'section\treturncode\n' >"${RUN_ROOT}/execution.tsv"

set +e
ALLOW_GPU_RUN=1 \
MATRIX_PROFILE_CONFIG="${C1_PROFILE}" \
MATRIX_SYSTEMS=native_pinned_uva,ours \
MATRIX_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
MATRIX_SINGLE_ATTEMPT=1 \
MATRIX_NATIVE_UVA_CUDAGRAPH_MODE="${NATIVE_GRAPH_MODE}" \
MATRIX_OURS_CUDAGRAPH_MODE=FULL_SEPARATE \
bash "${SCRIPT_DIR}/run_c1_mechanism_runtime_hot_audit_off_matrix.sh" \
  "${RUN_ROOT}/c1" >"${RUN_ROOT}/c1.launcher.log" 2>&1
c1_rc=$?
printf 'c1\t%s\n' "${c1_rc}" >>"${RUN_ROOT}/execution.tsv"

ALLOW_GPU_RUN=1 \
E2_C16_PROFILE="${C16_PROFILE}" \
E2_C16_SYSTEMS=native_pinned_uva,ours \
E2_C16_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
E2_C16_CUDAGRAPH_MODE="${NATIVE_GRAPH_MODE}" \
bash "${SCRIPT_DIR}/run_e2_c16_direct_batch_matrix.sh" \
  "${RUN_ROOT}/c16" >"${RUN_ROOT}/c16.launcher.log" 2>&1
c16_rc=$?
printf 'c16\t%s\n' "${c16_rc}" >>"${RUN_ROOT}/execution.tsv"
set -e

sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err" || true
nvidia-smi --query-gpu=index,memory.used,memory.free \
  --format=csv,noheader,nounits >"${RUN_ROOT}/gpu.after.csv"

set +e
"${VLLM_PY}" "${VERIFY}" "${RUN_ROOT}" --mode "${VERIFY_MODE}" \
  --native-graph-mode "${NATIVE_GRAPH_MODE}" \
  >"${RUN_ROOT}/verify.stdout.txt" 2>"${RUN_ROOT}/verify.stderr.txt"
verify_rc=$?
set -e
printf 'verification\t%s\n' "${verify_rc}" >>"${RUN_ROOT}/execution.tsv"

echo "RUN_ROOT=${RUN_ROOT}"
echo "C1_RC=${c1_rc} C16_RC=${c16_rc} VERIFY_RC=${verify_rc}"
(( c1_rc == 0 && c16_rc == 0 && verify_rc == 0 ))
