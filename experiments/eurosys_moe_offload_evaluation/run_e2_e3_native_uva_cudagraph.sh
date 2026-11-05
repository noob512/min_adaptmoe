#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
VERIFY="${SCRIPT_DIR}/verify_e2_e3_native_uva_cudagraph.py"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e2-e3-native-uva-cudagraph-$(date -u +%Y%m%dT%H%M%SZ)}"

die() { echo "E2/E3 Native UVA Graph: $*" >&2; exit 1; }

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ ! -e "${RUN_ROOT}" ]] || die "refusing to overwrite ${RUN_ROOT}"
for path in "${VLLM_PY}" "${VERIFY}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
[[ "${used}" == 0 ]] || die "GPU is not clean before the run: ${used} MiB"

mkdir -p "${RUN_ROOT}/provenance"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
kernel_epoch="$(date +%s)"
git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/repo.commit"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
sha256sum \
  "${BASH_SOURCE[0]}" \
  "${SCRIPT_DIR}/run_c1_mechanism_runtime_hot_audit_off_matrix.sh" \
  "${SCRIPT_DIR}/run_e2_c16_direct_batch_matrix.sh" \
  "${SCRIPT_DIR}/direct_batch_c16_vllm_runner.py" \
  "${SCRIPT_DIR}/verify_direct_batch_c16_case.py" \
  "${SCRIPT_DIR}/run_e3_native_copy_vs_uva_phase.sh" \
  "${REPO_ROOT}/experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py" \
  "${VERIFY}" >"${RUN_ROOT}/provenance/runner.sha256"

printf 'section\treturncode\n' >"${RUN_ROOT}/execution.tsv"

set +e
ALLOW_GPU_RUN=1 \
MATRIX_SYSTEMS=native_pinned_uva \
MATRIX_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
MATRIX_SINGLE_ATTEMPT=1 \
MATRIX_NATIVE_UVA_CUDAGRAPH_MODE=FULL_DECODE_ONLY \
bash "${SCRIPT_DIR}/run_c1_mechanism_runtime_hot_audit_off_matrix.sh" \
  "${RUN_ROOT}/e2/c1" >"${RUN_ROOT}/e2-c1.launcher.log" 2>&1
e2_c1_rc=$?
printf 'e2_c1\t%s\n' "${e2_c1_rc}" >>"${RUN_ROOT}/execution.tsv"

ALLOW_GPU_RUN=1 \
E2_C16_PROFILE="${SCRIPT_DIR}/configs/e2_c16_single_attempt_matrix.json" \
E2_C16_SYSTEMS=native_pinned_uva \
E2_C16_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
E2_C16_CUDAGRAPH_MODE=FULL_DECODE_ONLY \
bash "${SCRIPT_DIR}/run_e2_c16_direct_batch_matrix.sh" \
  "${RUN_ROOT}/e2/c16" >"${RUN_ROOT}/e2-c16.launcher.log" 2>&1
e2_c16_rc=$?
printf 'e2_c16\t%s\n' "${e2_c16_rc}" >>"${RUN_ROOT}/execution.tsv"

ALLOW_GPU_RUN=1 \
ARM_FILTER=uva \
ATTENTION_BACKEND=TRITON_ATTN \
CUDAGRAPH_MODE=FULL_SEPARATE \
BATCH_FILTER=1,16 \
CONDITIONING_REPEATS=1 \
MEASUREMENT_REPEATS=1 \
bash "${SCRIPT_DIR}/run_e3_native_copy_vs_uva_phase.sh" \
  "${RUN_ROOT}/e3" >"${RUN_ROOT}/e3.launcher.log" 2>&1
e3_rc=$?
printf 'e3\t%s\n' "${e3_rc}" >>"${RUN_ROOT}/execution.tsv"
set -e

sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err" || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  >"${RUN_ROOT}/gpu.after.csv"

set +e
"${VLLM_PY}" "${VERIFY}" "${RUN_ROOT}" \
  >"${RUN_ROOT}/verify.stdout.txt" 2>"${RUN_ROOT}/verify.stderr.txt"
verify_rc=$?
set -e
printf 'verification\t%s\n' "${verify_rc}" >>"${RUN_ROOT}/execution.tsv"

echo "RUN_ROOT=${RUN_ROOT}"
echo "E2_C1_RC=${e2_c1_rc} E2_C16_RC=${e2_c16_rc} E3_RC=${e3_rc} VERIFY_RC=${verify_rc}"
(( e2_c1_rc == 0 && e2_c16_rc == 0 && e3_rc == 0 && verify_rc == 0 ))
