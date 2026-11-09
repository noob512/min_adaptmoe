#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
PLAN_PREP="${SCRIPT_DIR}/prepare_e3_near_equal_offload_plans.py"
NATIVE_RUNNER="${SCRIPT_DIR}/run_e3_native_copy_vs_uva_phase.sh"
OURS_RUNNER="${SCRIPT_DIR}/run_e3_c1_c16_uva_single_round.sh"
VERIFY="${SCRIPT_DIR}/verify_e3_near_equal_offload_plus_one_expert.py"
SOURCE="${E3_CALIBRATION_SOURCE:-${SCRIPT_DIR}/inputs/frozen/qwen_decode_calibration_counts.json}"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-near-equal-offload-plus-one-expert-$(date -u +%Y%m%dT%H%M%SZ)}"

die() { echo "E3 near-equal offload: $*" >&2; exit 1; }

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ ! -e "${RUN_ROOT}" ]] || die "refusing to overwrite ${RUN_ROOT}"
for path in "${VLLM_PY}" "${PLAN_PREP}" "${NATIVE_RUNNER}" "${OURS_RUNNER}" "${VERIFY}" "${SOURCE}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
[[ "${used}" == 0 ]] || die "GPU is not clean before the run: ${used} MiB"

mkdir -p "${RUN_ROOT}/provenance/plans"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
kernel_epoch="$(date +%s)"
git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/repo.commit"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
"${VLLM_PY}" "${PLAN_PREP}" --source "${SOURCE}" \
  --output-dir "${RUN_ROOT}/provenance/plans" \
  >"${RUN_ROOT}/provenance/plan-generation.stdout.json"
sha256sum "${BASH_SOURCE[0]}" "${PLAN_PREP}" "${NATIVE_RUNNER}" \
  "${OURS_RUNNER}" "${VERIFY}" "${RUN_ROOT}/provenance/plans"/*.json \
  >"${RUN_ROOT}/provenance/runner-and-plans.sha256"
printf 'section\treturncode\n' >"${RUN_ROOT}/execution.tsv"

set +e
ALLOW_GPU_RUN=1 ARM_FILTER=uva ATTENTION_BACKEND=TRITON_ATTN \
  CUDAGRAPH_MODE=FULL_SEPARATE BATCH_FILTER=1,16 \
  CONDITIONING_REPEATS=1 MEASUREMENT_REPEATS=1 \
  bash "${NATIVE_RUNNER}" "${RUN_ROOT}/native" \
  >"${RUN_ROOT}/native.launcher.log" 2>&1
native_rc=$?
printf 'native\t%s\n' "${native_rc}" >>"${RUN_ROOT}/execution.tsv"

ALLOW_GPU_RUN=1 E3_RUN_NATIVE_BASELINES=0 E3_RUN_KTRANSFORMERS=0 \
  E3_PLAN_LOW="${RUN_ROOT}/provenance/plans/cold11.json" \
  E3_PLAN_MID="${RUN_ROOT}/provenance/plans/cold39.json" \
  E3_PLAN_HIGH="${RUN_ROOT}/provenance/plans/cold90.json" \
  E3_COLD_WIDTH_LOW=11 E3_COLD_WIDTH_MID=39 E3_COLD_WIDTH_HIGH=90 \
  bash "${OURS_RUNNER}" "${RUN_ROOT}/ours" \
  >"${RUN_ROOT}/ours.launcher.log" 2>&1
ours_rc=$?
printf 'ours\t%s\n' "${ours_rc}" >>"${RUN_ROOT}/execution.tsv"
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
echo "NATIVE_RC=${native_rc} OURS_RC=${ours_rc} VERIFY_RC=${verify_rc}"
(( native_rc == 0 && ours_rc == 0 && verify_rc == 0 ))
