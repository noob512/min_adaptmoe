#!/usr/bin/env bash
set -euo pipefail

# Historical filename retained for compatibility.  The current protocol runs one round.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_e3_ktransformers_c1_c16_baseline.sh"
VERIFY="${SCRIPT_DIR}/verify_e3_ktransformers_tuned_three_round.py"
KT_PY="${SCRIPT_DIR}/../../KTransformers/.venv/bin/python"
PREFIX="${1:-${SCRIPT_DIR}/artifacts/e3-ktransformers-tuned-single-round-$(date -u +%Y%m%dT%H%M%SZ)}"
UVA_SUMMARY="${UVA_SUMMARY:-${SCRIPT_DIR}/artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/summary.json}"
COPY_SUMMARY="${COPY_SUMMARY:-${SCRIPT_DIR}/artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/summary.json}"
OLD_KT="${OLD_KT:-${SCRIPT_DIR}/artifacts/e3-ktransformers-c1-c16-cudagraph-20260902-attempt-01/verification.json}"

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || {
  echo "GPU execution requires ALLOW_GPU_RUN=1" >&2
  exit 1
}
for required in "${RUN_ONE}" "${VERIFY}" "${KT_PY}" \
  "${UVA_SUMMARY}" "${COPY_SUMMARY}" "${OLD_KT}"; do
  [[ -f "${required}" ]] || { echo "missing file: ${required}" >&2; exit 1; }
done

root="${PREFIX}"
CPU_THREADS=7 CPUSET=0-13 ATTENTION_BACKEND=flashinfer \
  ATTENTION_BACKEND_RAG_B16=triton \
  DECODE_ATTENTION_BACKEND= GPU_PREFILL_TOKEN_THRESHOLD=0 \
  GPU_PREFILL_TOKEN_THRESHOLD_MEDIUM_B16=16384 \
  GPU_PREFILL_TOKEN_THRESHOLD_RAG_B16=4096 \
  DISABLE_OVERLAP_SCHEDULE=0 E3_KT_SKIP_VERIFY=1 ALLOW_GPU_RUN=1 \
  bash "${RUN_ONE}" "${root}"

"${KT_PY}" "${VERIFY}" \
  --kt-root "${root}" \
  --native-uva-summary "${UVA_SUMMARY}" \
  --native-copy-summary "${COPY_SUMMARY}" \
  --old-kt-verification "${OLD_KT}" \
  --output "${PREFIX}-verification.json"

echo "PREFIX=${PREFIX}"
echo "VERIFICATION=${PREFIX}-verification.json"
