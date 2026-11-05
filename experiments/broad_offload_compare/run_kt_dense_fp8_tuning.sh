#!/usr/bin/env bash
set -euo pipefail

# GPU entry point for the reviewed, CPU-generated workload M plan.  This script
# never installs configs into the pinned checkout; installation is a separate
# explicit and reversible command.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${KT_ROOT:-${WORKSPACE_ROOT}/KTransformers}"
SGLANG_ROOT="${KT_ROOT}/third_party/sglang"
KT_PYTHON="${KT_PYTHON:-${KT_ROOT}/.venv/bin/python}"
HELPER="${SCRIPT_DIR}/kt_dense_fp8_tuning.py"
EXPECTED_KT_COMMIT="eb9b70c4115cff151ace2cae5b0fc9db3690e31e"
EXPECTED_SGLANG_COMMIT="bc7f0058fafe6c05dc6ad52d6b1c854af2a673b2"

usage() {
  cat <<'EOF'
Usage:
  run_kt_dense_fp8_tuning.sh PLAN_JSON OUTPUT_ROOT

The plan must contain formal frozen-input-ID traces for every requested
concurrency. OUTPUT_ROOT must not already contain fragments or a bundle.

Environment:
  KT_ROOT=/path/KTransformers
  KT_PYTHON=/path/KTransformers/.venv/bin/python
  KT_TUNE_GPU=0              physical GPU index exposed to the tuner
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi
[[ $# -eq 2 ]] || { usage; exit 2; }
PLAN="$1"
OUTPUT_ROOT="$2"
GPU_ID="${KT_TUNE_GPU:-0}"

[[ -f "${PLAN}" ]] || die "missing plan: ${PLAN}"
[[ -x "${KT_PYTHON}" ]] || die "missing KT Python: ${KT_PYTHON}"
[[ -f "${HELPER}" ]] || die "missing helper: ${HELPER}"
[[ "$(git -C "${KT_ROOT}" rev-parse HEAD)" == "${EXPECTED_KT_COMMIT}" ]] \
  || die "KTransformers commit mismatch"
[[ "$(git -C "${SGLANG_ROOT}" rev-parse HEAD)" == "${EXPECTED_SGLANG_COMMIT}" ]] \
  || die "SGLang commit mismatch"
[[ ! -e "${OUTPUT_ROOT}/fragments" && ! -e "${OUTPUT_ROOT}/bundle" ]] \
  || die "refusing to mix with an existing tuning output root"

mapfile -t COMPUTE_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d'
)
(( ${#COMPUTE_PIDS[@]} == 0 )) \
  || die "GPU already has compute processes: ${COMPUTE_PIDS[*]}"

# jobs also enforces qualification_ready=true before any CUDA process starts.
mapfile -t JOBS < <("${KT_PYTHON}" "${HELPER}" jobs --plan "${PLAN}")
(( ${#JOBS[@]} > 0 )) || die "plan produced no tuning jobs"

mkdir -p "${OUTPUT_ROOT}/fragments" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/metadata"
cp -- "${PLAN}" "${OUTPUT_ROOT}/metadata/workload_m_plan.json"
sha256sum "${PLAN}" > "${OUTPUT_ROOT}/metadata/workload_m_plan.sha256"
git -C "${KT_ROOT}" rev-parse HEAD > "${OUTPUT_ROOT}/metadata/ktransformers.commit"
git -C "${SGLANG_ROOT}" rev-parse HEAD > "${OUTPUT_ROOT}/metadata/sglang.commit"
"${KT_PYTHON}" -m pip freeze > "${OUTPUT_ROOT}/metadata/pip_freeze.txt"
nvidia-smi -q > "${OUTPUT_ROOT}/metadata/nvidia_smi_before.txt"
lscpu > "${OUTPUT_ROOT}/metadata/lscpu.txt"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONHASHSEED=0

for job in "${JOBS[@]}"; do
  IFS=$'\t' read -r n k m <<< "${job}"
  stem="N${n}_K${k}_M${m}"
  fragment="${OUTPUT_ROOT}/fragments/${stem}.fragment.json"
  log="${OUTPUT_ROOT}/logs/${stem}.log"
  echo "tuning ${stem}"
  set +e
  "${KT_PYTHON}" "${HELPER}" tune-one \
    --sglang-root "${SGLANG_ROOT}" \
    --N "${n}" --K "${k}" --M "${m}" \
    --seed 0 --output "${fragment}" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  (( rc == 0 )) || die "${stem} failed with return code ${rc}"
done

"${KT_PYTHON}" "${HELPER}" merge \
  --plan "${PLAN}" \
  --fragments-dir "${OUTPUT_ROOT}/fragments" \
  --bundle-dir "${OUTPUT_ROOT}/bundle" \
  | tee "${OUTPUT_ROOT}/logs/merge.log"

nvidia-smi -q > "${OUTPUT_ROOT}/metadata/nvidia_smi_after.txt"
sha256sum "${OUTPUT_ROOT}/bundle/configs/"*.json \
  > "${OUTPUT_ROOT}/metadata/bundle_configs.sha256"
echo "review bundle: ${OUTPUT_ROOT}/bundle/manifest.json"
echo "configs were not installed"
