#!/usr/bin/env bash
set -euo pipefail

# Resume only an interrupted, manifest-pinned dense FP8 tuning run. Completed
# fragments are validated before they are skipped; partial logs are archived.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${KT_ROOT:-${REPO_ROOT}/KTransformers}"
SGLANG_ROOT="${KT_ROOT}/third_party/sglang"
KT_PY="${KT_PY:-${KT_ROOT}/.venv/bin/python}"
HELPER="${SCRIPT_DIR}/kt_dense_fp8_tuning.py"
EXPECTED_KT_COMMIT="eb9b70c4115cff151ace2cae5b0fc9db3690e31e"
EXPECTED_SGLANG_COMMIT="bc7f0058fafe6c05dc6ad52d6b1c854af2a673b2"

die() {
  echo "error: $*" >&2
  exit 1
}

[[ $# -eq 2 ]] || die "usage: $0 PLAN_JSON EXISTING_OUTPUT_ROOT"
PLAN="$(realpath "$1")"
OUTPUT_ROOT="$(realpath "$2")"
[[ -f "${PLAN}" ]] || die "missing plan: ${PLAN}"
[[ -d "${OUTPUT_ROOT}/fragments" && -d "${OUTPUT_ROOT}/logs" ]] \
  || die "output root is not an initialized interrupted tuning run"
[[ ! -e "${OUTPUT_ROOT}/bundle" ]] || die "bundle already exists; resume is unnecessary"
[[ -x "${KT_PY}" && -f "${HELPER}" ]] || die "tuning Python/helper is missing"
[[ "$(git -C "${KT_ROOT}" rev-parse HEAD)" == "${EXPECTED_KT_COMMIT}" ]] \
  || die "KTransformers commit drift"
[[ "$(git -C "${SGLANG_ROOT}" rev-parse HEAD)" == "${EXPECTED_SGLANG_COMMIT}" ]] \
  || die "SGLang commit drift"

recorded_plan="${OUTPUT_ROOT}/metadata/workload_m_plan.json"
recorded_sha="${OUTPUT_ROOT}/metadata/workload_m_plan.sha256"
[[ -f "${recorded_plan}" && -f "${recorded_sha}" ]] \
  || die "original plan metadata is missing"
[[ "$(sha256sum "${PLAN}" | awk '{print $1}')" == \
    "$(sha256sum "${recorded_plan}" | awk '{print $1}')" ]] \
  || die "resume plan differs from the recorded plan"
sha256sum --status -c "${recorded_sha}" \
  || die "recorded plan checksum no longer validates"

mapfile -t COMPUTE_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk '$1 ~ /^[0-9]+$/ {print $1}'
)
(( ${#COMPUTE_PIDS[@]} == 0 )) \
  || die "GPU has compute processes: ${COMPUTE_PIDS[*]}"
used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
  | head -n 1 | tr -d ' ')"
[[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= 128 )) \
  || die "GPU memory is not clean: ${used_mib:-unknown} MiB"

mapfile -t JOBS < <("${KT_PY}" "${HELPER}" jobs --plan "${PLAN}")
(( ${#JOBS[@]} > 0 )) || die "plan has no jobs"

validate_fragment() {
  local fragment="$1" n="$2" k="$3" m="$4"
  "${KT_PY}" - "${fragment}" "${n}" "${k}" "${m}" \
    "${EXPECTED_SGLANG_COMMIT}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
n, k, m = map(int, sys.argv[2:5])
expected_commit = sys.argv[5]
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema") != "kt-dense-fp8-tuning-v1" or value.get("kind") != "tuning-fragment":
    raise SystemExit(f"{path}: schema/kind mismatch")
shape = value.get("shape", {})
expected_shape = {
    "M": m,
    "N": n,
    "K": k,
    "block_shape": [128, 128],
    "input_dtype": "float8_e4m3fn",
    "output_dtype": "bfloat16",
}
if shape != expected_shape:
    raise SystemExit(f"{path}: shape mismatch")
config = value.get("best_config")
if not isinstance(config, dict) or config.get("BLOCK_SIZE_K") != 128:
    raise SystemExit(f"{path}: runtime-inexact K tile")
if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in config.values()):
    raise SystemExit(f"{path}: invalid best config")
search = value.get("search", {})
if search.get("candidate_count") != 640 or not isinstance(search.get("elapsed_seconds"), (int, float)):
    raise SystemExit(f"{path}: search provenance mismatch")
environment = value.get("environment", {})
if environment.get("device_name") != "NVIDIA GeForce RTX 5090":
    raise SystemExit(f"{path}: device mismatch")
if environment.get("compute_capability") != [12, 0]:
    raise SystemExit(f"{path}: compute capability mismatch")
if environment.get("sglang_commit") != expected_commit:
    raise SystemExit(f"{path}: SGLang commit mismatch")
for key in ("tuner_sha256", "fp8_kernel_sha256", "candidate_sha256"):
    candidate = environment.get(key) if key != "candidate_sha256" else search.get(key)
    if not isinstance(candidate, str) or len(candidate) != 64:
        raise SystemExit(f"{path}: invalid {key}")
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
resume_dir="${OUTPUT_ROOT}/logs/interrupted_${stamp}"
mkdir -p "${resume_dir}"
manifest="${OUTPUT_ROOT}/metadata/resume_${stamp}.json"
"${KT_PY}" - "${manifest}" "${PLAN}" "${OUTPUT_ROOT}" "${stamp}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "kt-dense-fp8-tuning-resume-v1",
    "created_at_utc": sys.argv[4],
    "plan": str(Path(sys.argv[2]).resolve()),
    "output_root": str(Path(sys.argv[3]).resolve()),
    "reason": "previous process ended without a tuner exception or complete fragment",
    "policy": "validate and skip complete fragments; archive partial logs; merge only after every planned job succeeds",
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${KT_TUNE_GPU:-0}"
export PYTHONHASHSEED=0

for job in "${JOBS[@]}"; do
  IFS=$'\t' read -r n k m <<< "${job}"
  stem="N${n}_K${k}_M${m}"
  fragment="${OUTPUT_ROOT}/fragments/${stem}.fragment.json"
  if [[ -f "${fragment}" ]]; then
    validate_fragment "${fragment}" "${n}" "${k}" "${m}" \
      > "${OUTPUT_ROOT}/logs/${stem}.validated.sha256"
    echo "validated existing ${stem}"
    continue
  fi
  mapfile -t partial_logs < <(
    find "${OUTPUT_ROOT}/logs" -maxdepth 1 -type f \
      -name "${stem}*.log" -print | sort
  )
  for partial_log in "${partial_logs[@]}"; do
    mv -- "${partial_log}" "${resume_dir}/$(basename "${partial_log}")"
  done
  log="${OUTPUT_ROOT}/logs/${stem}.resume_${stamp}.log"
  echo "resuming ${stem}"
  set +e
  "${KT_PY}" "${HELPER}" tune-one \
    --sglang-root "${SGLANG_ROOT}" \
    --N "${n}" --K "${k}" --M "${m}" \
    --seed 0 --output "${fragment}" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  (( rc == 0 )) || die "${stem} failed with return code ${rc}"
  validate_fragment "${fragment}" "${n}" "${k}" "${m}" \
    > "${OUTPUT_ROOT}/logs/${stem}.validated.sha256"
done

"${KT_PY}" "${HELPER}" merge \
  --plan "${PLAN}" \
  --fragments-dir "${OUTPUT_ROOT}/fragments" \
  --bundle-dir "${OUTPUT_ROOT}/bundle" \
  | tee "${OUTPUT_ROOT}/logs/merge.resume_${stamp}.log"
nvidia-smi -q > "${OUTPUT_ROOT}/metadata/nvidia_smi_after_resume_${stamp}.txt"
sha256sum "${OUTPUT_ROOT}/bundle/configs/"*.json \
  > "${OUTPUT_ROOT}/metadata/bundle_configs.sha256"
echo "complete_bundle=${OUTPUT_ROOT}/bundle/manifest.json"
