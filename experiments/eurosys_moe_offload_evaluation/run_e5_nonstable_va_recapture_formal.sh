#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/experiments/vllm_uvm_dynamic_kv_plan_rebuild/run_experiment.sh"
VERIFY="${SCRIPT_DIR}/verify_e5_nonstable_va_recapture_formal.py"
NAMES="${SCRIPT_DIR}/inputs/e5_qwen_layers_0_3_exact_parameter_names.txt"
PLAN="${SCRIPT_DIR}/inputs/generated/e5_qwen_layers_0_3_full_expert_plan_20260828.json"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
PYTHON="${VLLM_ROOT}/.venv/bin/python"
CUSTOM_UVM="${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko"

RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e5-nonstable-va-recapture-$(date -u +%Y%m%dT%H%M%SZ)-formal}"
REPEATS="${REPEATS:-10}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || {
  echo "E5 non-stable-VA formal requires ALLOW_GPU_RUN=1" >&2
  exit 1
}
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "REPEATS must be a positive integer" >&2
  exit 1
}
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2
  exit 1
}
for path in "${HARNESS}" "${VERIFY}" "${NAMES}" "${PLAN}" "${PYTHON}" "${CUSTOM_UVM}"; do
  [[ -e "${path}" ]] || {
    echo "missing required input: ${path}" >&2
    exit 1
  }
done

mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
printf 'repeat\tstatus\trun_root\n' >"${RUN_ROOT}/execution_status.tsv"
"${PYTHON}" - "${RUN_ROOT}/run_config.json" "${REPEATS}" <<'PY'
import json, sys
output, repeats = sys.argv[1:]
payload = {
    "schema": "e5_nonstable_va_recapture_formal_v1",
    "repeats": int(repeats),
    "model": "Qwen3-30B-A3B-FP8",
    "layers": [0, 1, 2, 3],
    "parameter_count": 8,
    "selected_bytes": 2_415_919_104,
    "physical_kv_bytes": 134_217_728,
    "sequence": ["initial_b1", "reallocate_b4", "reallocate_b1"],
    "process_isolation": "one cold process per repeat",
    "parameter_va_contract": "all eight VAs change per generation; all Graphs recaptured",
}
with open(output, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

sha256sum \
  "${BASH_SOURCE[0]}" "${HARNESS}" "${VERIFY}" "${NAMES}" "${PLAN}" \
  "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
  "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
  "${VLLM_ROOT}/vllm/device_allocator/uvm_static_alias.py" \
  "${CUSTOM_UVM}" >"${RUN_ROOT}/input_sha256.txt"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/git_status.txt"

for repeat in $(seq 1 "${REPEATS}"); do
  repeat_label="$(printf '%02d' "${repeat}")"
  child_root="${RUN_ROOT}/repeat_${repeat_label}"
  printf '%s\n' \
    "VLLM_UVM_E5_NONSTABLE_VA_RECAPTURE=1" \
    "ALLOW_GPU_RUN=1" \
    "RUN_ROOT=${child_root}" \
    "NAMES=${NAMES}" \
    "PLAN_2G=${PLAN}" \
    "PLAN_4G=${PLAN}" \
    "PLAN_2G_SELECTED_BYTES=2415919104" \
    "PLAN_4G_SELECTED_BYTES=2415919104" \
    "EXPECTED_PARAMETER_COUNT=8" \
    "EXPECTED_FULL_PARAMETER_BYTES=2415919104" \
    "KV_CACHE_BYTES=134217728" \
    "SLOT_BUDGET_BYTES=1207959552" \
    "CPU_OFFLOAD_GB=3.0" \
    "TIMEOUT_SEC=${TIMEOUT_SEC}" \
    "bash ${HARNESS} qualify" >"${RUN_ROOT}/repeat_${repeat_label}.command.txt"

  set +e
  env \
    VLLM_UVM_E5_NONSTABLE_VA_RECAPTURE=1 \
    ALLOW_GPU_RUN=1 \
    RUN_ROOT="${child_root}" \
    NAMES="${NAMES}" \
    PLAN_2G="${PLAN}" \
    PLAN_4G="${PLAN}" \
    PLAN_2G_SELECTED_BYTES=2415919104 \
    PLAN_4G_SELECTED_BYTES=2415919104 \
    PLAN_2G_LABEL=matched_b1 \
    PLAN_4G_LABEL=matched_b4 \
    EXPECTED_PARAMETER_COUNT=8 \
    EXPECTED_FULL_PARAMETER_BYTES=2415919104 \
    KV_CACHE_BYTES=134217728 \
    SLOT_BUDGET_BYTES=1207959552 \
    CPU_OFFLOAD_GB=3.0 \
    TIMEOUT_SEC="${TIMEOUT_SEC}" \
    bash "${HARNESS}" qualify \
    >"${RUN_ROOT}/repeat_${repeat_label}.stdout.log" \
    2>"${RUN_ROOT}/repeat_${repeat_label}.stderr.log"
  rc=$?
  set -e
  if (( rc != 0 )); then
    printf '%s\tfailed:%s\t%s\n' "${repeat}" "${rc}" "${child_root}" \
      >>"${RUN_ROOT}/execution_status.tsv"
    echo "repeat ${repeat} failed with rc=${rc}" >&2
    exit "${rc}"
  fi
  printf '%s\tok\t%s\n' "${repeat}" "${child_root}" \
    >>"${RUN_ROOT}/execution_status.tsv"
done

"${PYTHON}" "${VERIFY}" "${RUN_ROOT}" \
  --expected-repeats "${REPEATS}" \
  --output "${RUN_ROOT}/verification.json"
echo "RUN_ROOT=${RUN_ROOT}"
echo "verification=${RUN_ROOT}/verification.json"
